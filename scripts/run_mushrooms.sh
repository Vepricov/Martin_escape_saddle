#!/usr/bin/env bash
clear
for optimizer in adamw
do
    for transform in "" "--scale"
    do
        CUDA_VISIBLE_DEVICES=3 python ./src/run_experiment.py \
            --dataset mushrooms \
            --eval_runs 1 \
            --n_epoches 1 \
            --tune_runs 40 \
            --optimizer $optimizer \
            --hidden_dim 10 \
            $transform \
            --no_bias \
            --tune \
            --lmo frobenious \
            --precondition_type fisher \
            --momentum 0 \
            --dtype float64 \
            # --wandb # --use_old_tune_params \
    done
done