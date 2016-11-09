#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_PRELOAD="/usr/local/lib/libtcmalloc.so"

checkpoint_dir="./$1"
test_output="$checkpoint_dir/eval_test.txt"
train_output="$checkpoint_dir/eval_train.txt"
batch_size=50
num_gpu=4
test_iter=250
train_iter=250
gpu_fraction=0.96

python eval.py --ckpt_path $checkpoint_dir \
               --output $test_output \
               --num_gpu $num_gpu \
               --batch_size $batch_size \
               --test_iter $test_iter \
               --gpu_fraction $gpu_fraction \
               --display 50

python eval.py --ckpt_path $checkpoint_dir \
               --output $train_output \
               --num_gpu $num_gpu \
               --batch_size $batch_size \
               --test_iter $train_iter \
               --gpu_fraction $gpu_fraction \
               --display 50 \
               --train_data True
