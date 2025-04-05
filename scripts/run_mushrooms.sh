#!/usr/bin/env bash
clear
for optimizer in taia
do
    for transform in "" "--scale"
    do
        CUDA_VISIBLE_DEVICES=7 python ./src/run_experiment.py \
            --dataset mushrooms \
            --eval_runs 1 \
            --n_epoches 1 \
            --tune_runs 40 \
            --optimizer $optimizer \
            --hidden_dim 10 \
            $transform \
            --no_bias \
            --lmo frobenious \
            --use_old_tune_params \
            --momentum 0 \
            --wandb # 
    done
done