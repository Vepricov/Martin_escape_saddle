#!/usr/bin/env bash

clear
#source ~/dl/bin/activate
python unbalanced_dataset.py --config config.json --device 2 --tune --use_old_tune_params --balanced_test > results_all_final.txt
#python unbalanced_dataset.py --config config.json --device 2 --augment --tune > results_augment.txt
