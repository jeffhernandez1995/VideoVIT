# 8 GPU training (use only 1 for ResNet-18 training)
export CUDA_VISIBLE_DEVICES=0,1

# Set the visible GPUs according to the `world_size` configuration parameter
# Modify `data.in_memory` and `data.num_workers` based on your machine
python train_imagenet.py --config-file rn50_16_epochs.yaml --data.train_dataset=datasets/imagenet/train_500_1_90.ffcv --data.val_dataset=datasets/imagenet/validation_500_1_90.ffcv --data.num_workers=12 --data.in_memory=1 --logging.folder=output/ --logging.log_level=1