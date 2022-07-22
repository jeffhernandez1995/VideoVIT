# python train_vit_using_timm.py datasets/imagenette_ffcv/ --model vit_base_patch16_224 --sched cosine --epochs 300 --opt adamw -j 4 --warmup-lr 1e-6 --model-ema --model-ema-decay 0.99996 --amp --lr 5e-4 --weight-decay .05 --drop 0.1 --drop-path .1 -b 16 --log-wandb --experiment ffcv_test

./distributed_train.sh 4 datasets/imagenet --model vit_small_patch16_224 --sched cosine --epochs 300 --opt adam -j 12 --model-ema --model-ema-decay 0.99996 --amp --lr 0.001 --weight-decay 0.0001 --drop 0.1 --drop-path .1 --clip-grad 1.0 -b 512 --log-wandb

# ./distributed_train.sh 2 datasets/imagenet --model resnet --sched cosine --epochs 300 --opt adamw -j 12 --warmup-lr 1e-6 --model-ema --model-ema-decay 0.99996 --amp --lr 5e-4 --weight-decay .05 --drop 0.1 --drop-path .1 -b 512 --log-wandb