import os
import random

import re
import torch.utils.data
import torchvision
import itertools
from torchvision import transforms as T


# Dataset are divided according to the instruction at:
# http://www.nada.kth.se/cvap/actions/00sequences.txt
TRAIN_PEOPLE_ID = [11, 12, 13, 14, 15, 16, 17, 18]
DEV_PEOPLE_ID = [19, 20, 21, 23, 24, 25, 1, 4]
TEST_PEOPLE_ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]


def parse_sequence_file(filename):
    print(f"Parsing ..{filename}")

    # Read 00sequences.txt file.
    with open(f'{filename}', 'r') as content_file:
        content = content_file.readlines()

    # Remove top of file with information about splits
    content = content[20:]
    content = [f.replace('\n', '') for f in content]

    filenames = ['{}_uncomp.avi'.format(line.split("\t")[0]) for line in content]
    frames = [line.split("\t")[-1].replace(' ', '').split(",") for line in content]
    frames = [
        [tuple(map(int, val.split("-"))) for val in f]
        for f in frames
    ]
    frames_idx = dict(zip(filenames, frames))
    return frames_idx


def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(dir, mode, class_to_idx, extensions):
    seq_filename = os.path.join(dir, "sequences.txt")
    frames_idx = parse_sequence_file(seq_filename)

    if mode == "training":
        IDS = TRAIN_PEOPLE_ID
    elif mode == "validation":
        IDS = DEV_PEOPLE_ID
    else:
        IDS = TEST_PEOPLE_ID

    dataset = []
    for filename, frames in frames_idx.items():
        if filename.endswith(extensions):
            person_id = int(re.findall(r'\d+', filename.split("_")[1])[0])
            if person_id not in IDS:
                continue
            dataset.append(
                (filename, class_to_idx[filename.split("/")[-2]], frames)
            )
    return dataset


def get_samples(root, mode, extensions=(".mp4", ".avi")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, mode, class_to_idx, extensions=extensions)


class KTHDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        root,
        mode,
        epoch_size=None,
        frame_transform=None,
        video_transform=None,
        clip_len=15
    ):
        super(KTHDataset).__init__()

        self.samples = get_samples(root, mode)

        # Allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size

        self.clip_len = clip_len
        self.frame_transform = frame_transform
        self.video_transform = video_transform

    def __iter__(self):
        for i in range(self.epoch_size):
            # Get random sample
            path, target, frames = random.choice(self.samples)
            # Get video object
            vid = torchvision.io.VideoReader(path, "video")
            metadata = vid.get_metadata()
            video_frames = []  # video frame buffer
            # Get random frame start and end
            start_f, end_f = random.choice(frames)
            # Seek and return frames
            min_seek = start_f / metadata["video"]['fps'][0]
            max_seek = (
                (end_f / metadata["video"]['fps'][0]) -
                (self.clip_len / metadata["video"]['fps'][0])
            )
            start = random.uniform(min_seek, max_seek)
            for frame in itertools.islice(vid.seek(start), self.clip_len):
                video_frames.append(self.frame_transform(frame['data']))
                current_pts = frame['pts']
            # Stack it into a tensor
            video = torch.stack(video_frames, 0)
            if self.video_transform:
                video = self.video_transform(video)
            output = {
                'path': path,
                'video': video,
                'target': target,
                'start': start,
                'end': current_pts
            }
            yield output


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = KTHDataset(
        root='datasets/KTH_raw',
        epoch_size=None,
        frame_transform=transform,
        video_transform=None,
        clip_len=15
    )

    for i in range(10):
        sample = next(iter(dataset))
        print(sample['path'])
        print(sample['video'].shape)
        print(sample['target'])
        print(sample['start'])
        print(sample['end'])
