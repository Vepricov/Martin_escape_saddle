#!/usr/bin/env bash
clear
for optimizer_name in adam
do
    for scale in 0 1
    do
        CUDA_VISIBLE_DEVICES=4 python ./src/run_experiment.py \
            --config ./configs/config_mushrooms.json \
            --optimizer_name $optimizer_name \
            --scale $scale \
            --use_old_tune_params 
    done
done