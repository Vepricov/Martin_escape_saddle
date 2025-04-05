#!/usr/bin/env bash
clear
for optimizer in adam-sania 
do
    CUDA_VISIBLE_DEVICES=4 python ./src/run_experiment.py \
        --dataset cifar10 \
        --model resnet18 \
        --eval_runs 1 \
        --tune_runs 100 \
        --optimizer $optimizer \
        --use_old_tune_params  \
        --wandb --not_augment #
done