#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_PRELOAD="/usr/local/lib/libtcmalloc.so"

train_dir="./train_multigpu"

python train.py --train_dir $train_dir \
    --num_gpu 4 \
    --batch_size 50 \
    --test_interval 500 \
    --test_iter 10 \
    --l2_weight 0.0001 \
    --initial_lr 0.001 \
    --lr_step_epoch 1.0,2.0 \
    --lr_decay 0.1 \
    --max_steps 19215 \
    --checkpoint_interval 3200 \
    --gpu_fraction 0.96 \
    --display 50 \
    #--log_device_placement True \
