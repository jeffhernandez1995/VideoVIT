#/bin/bash!

# python train_and_finetune_video.py --batch_size 4 --epochs 10 --model mae_vit_small_patch16 --input_size 224 --mask_ratio 0.75 --train_path datasets/KTH_raw/ --output_dir output/ --log-wandb --num_workers 2 --finetune_path datasets/imagenette_ffcv/ --eval_freq 2 --nb_classes 10 --eval_epochs 1

python train_and_finetune_video.py --batch_size 256 --epochs 300 --model mae_vit_small_patch16 --input_size 224 --mask_ratio 0.75 --train_path datasets/Moments_in_Time_Raw/ --output_dir output/ --log-wandb --num_workers 12 --finetune_path datasets/imagenet/ --eval_freq 50 --nb_classes 1000 --eval_epochs 50 --world_size 4 --local_rank 0