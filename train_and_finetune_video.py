import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
# from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

import timm
from timm.models.layers import trunc_normal_
import timm.optim.optim_factory as optim_factory
from torchvision.datasets.folder import make_dataset

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
import models_vit

from engine_pretrain import train_one_epoch
from engine_finetune import train_one_epoch as finetune_one_epoch
from engine_finetune import evaluate
import wandb
from util.lars import LARS

import torchvision.datasets.video_utils
from torchvision.io import read_video

# from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset
from copy import deepcopy

import os
import random

from typing import List
import numpy as np
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

import ffmpeg

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255


def get_video_size(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    try:
        n_frames = int(video_info['nb_frames'])
    except KeyError:
        n_frames = float(video_info['duration']) * eval(video_info['r_frame_rate'])
    frame_rate = eval(video_info['r_frame_rate'])
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height, int(n_frames), frame_rate, float(video_info['duration'])


class Video(VisionDataset):

    def __init__(
        self,
        root,
        extensions=('mp4', 'avi'),
        transform=None,
    ):
        super(Video, self).__init__(root)
        extensions = extensions

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get random sample
        success = False
        while not success:
            try:
                path, target = self.samples[idx]
                _, _, _, _, duration = get_video_size(path)
                start = random.uniform(0., duration)
                frame, _, _ = torchvision.io.read_video(
                    path,
                    start_pts=start,
                    end_pts=start,
                    pts_unit='sec'
                )
                success = True
            except Exception as e:
                print(e)
                print('skipped idx', idx)
                idx = np.random.randint(self.__len__())
        # Seek and return frames
        frame = self.transform(frame)
        return frame, target


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--train_path', default='datasets/KTH_raw/', type=str,
                        help='dataset path')
    parser.add_argument('--finetune_path', default='datasets/imagenette_ffcv/', type=str,
                        help='dataset path')

    parser.add_argument('--eval_freq' , default=1, type=int)
    parser.add_argument('--eval_epochs', default=1, type=int)
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output/',
                        help='path where to tensorboard log')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.output_dir is None:
        args.output_dir = 'output/'

    exp_name = '-'.join([
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.model,
        str(args.input_size),
    ])
    args.output_dir = os.path.join(args.output_dir, exp_name)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # simple augmentation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    # print(dataset_train)
    dataset_train = Video(
        os.path.join(args.train_path, 'training'),
        # args.train_path,
        extensions=('mp4', 'avi'),
        transform=transform,
    )

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(
            'Training in distributed mode with multiple processes, ',
            '1 GPU per process. Process %d, total %d.'
            % (args.rank, args.world_size)
        )
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        print('Training with a single process on 1 GPUs.')
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    assert args.rank >= 0

    if args.rank == 0 and args.log_wandb:
        logger = wandb.init(project="imagenet", entity="jeffhernandez1995", config=args)
    else:
        logger = None

    # if args.rank == 0 and args.log_dir is not None:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = SummaryWriter(log_dir=args.log_dir)
    # else:
    #     log_writer = None

    train_decoder = RandomResizedCropRGBImageDecoder(
        (args.input_size, args.input_size),
    )
    image_pipeline: List[Operation] = [
        train_decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(args.device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(
            torch.device(args.device),
            non_blocking=True
        )
    ]

    order = OrderOption.RANDOM if args.distributed else OrderOption.QUASI_RANDOM
    loader_train = Loader(
        f"{args.finetune_path}/train_500_1_90.ffcv",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=order,
        os_cache=True,
        drop_last=True,
        pipelines={
            'image': image_pipeline,
            'label': label_pipeline
        },
        distributed=args.distributed,
        seed=args.seed,
    )

    res_tuple = (args.input_size, args.input_size)
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=0.75)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(torch.device(args.device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(
            torch.device(args.device),
            non_blocking=True
        )
    ]

    loader_eval = Loader(
        f"{args.finetune_path}/validation_500_1_90.ffcv",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=OrderOption.SEQUENTIAL,
        drop_last=True,
        pipelines={
            'image': image_pipeline,
            'label': label_pipeline
        },
        distributed=args.distributed
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = models_mae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=logger,
            args=args
        )

        if epoch % args.eval_freq == 0:
            class_model = models_vit.__dict__[args.model.replace('mae_', '')](
                num_classes=args.nb_classes,
                global_pool=False,
            )
            model_to_copy = model_without_ddp.state_dict()
            state_dict = class_model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in model_to_copy and model_to_copy[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del model_to_copy[k]
            # interpolate position embedding
            interpolate_pos_embed(class_model, model_to_copy)
            msg = class_model.load_state_dict(model_to_copy, strict=False)
            print(msg)
            trunc_normal_(class_model.head.weight, std=0.01)
            class_model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(class_model.head.in_features, affine=False, eps=1e-6), class_model.head)
            # freeze all but the head
            for _, p in class_model.named_parameters():
                p.requires_grad = False
            for _, p in class_model.head.named_parameters():
                p.requires_grad = True

            class_model.to(args.device)

            class_model_without_ddp = class_model

            args.fintune_lr = 0.1 * eff_batch_size / 256

            # print("Class Model = %s" % str(class_model_without_ddp))
            if args.distributed:
                class_model = torch.nn.parallel.DistributedDataParallel(class_model, device_ids=[args.gpu], find_unused_parameters=True)
                class_model_without_ddp = class_model.module
            class_optimizer = LARS(class_model_without_ddp.head.parameters(), lr=args.fintune_lr, weight_decay=0)
            print(class_optimizer)
            class_loss_scaler = NativeScaler()

            criterion = torch.nn.CrossEntropyLoss()

            print("criterion = %s" % str(criterion))
            max_accuracy = 0.0
            for _ in range(args.eval_epochs):
                train_stats = finetune_one_epoch(
                    class_model, criterion, loader_train,
                    class_optimizer, args.device, epoch, class_loss_scaler,
                    max_norm=None,
                    log_writer=logger,
                    args=args
                )
                test_stats = evaluate(loader_eval, class_model, args.device, epoch, log_writer=logger)
                print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
                max_accuracy = max(max_accuracy, test_stats["acc1"])
                print(f'Max accuracy: {max_accuracy:.2f}%')
                # if logger is not None:
                #     logger.log({
                #         'val_test_acc1': test_stats['acc1'],
                #         'val_test_acc5': test_stats['acc5'],
                #         'val_test_loss': test_stats['loss']
                #     }, step=epoch)

        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            # if log_writer is not None:
            #     log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
