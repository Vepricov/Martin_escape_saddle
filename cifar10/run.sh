#!/usr/bin/env bash

clear
#source ~/dl/bin/activate
CUDA_VISIBLE_DEVICES=7 python run_cifar.py --config config.json --device 0 > result.txt
#--use_old_tune_params
