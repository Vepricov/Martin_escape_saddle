#!/usr/bin/env bash
clear
CUDA_VISIBLE_DEVICES=4 python ./src/run_experiment.py \
    --config ./configs/config_cifar.json \
    --use_old_tune_params \
    --verbose 
