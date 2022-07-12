#/bin/bash!

python train_and_finetune_video.py --batch_size 4 --epochs 10 --model mae_vit_small_patch16 --input_size 224 --mask_ratio 0.75 --train_path datasets/KTH_raw/ --output_dir output/ --log-wandb --num_workers 2 --finetune_path datasets/imagenette_ffcv/ --eval_freq 2 --nb_classes 10 --eval_epochs 1