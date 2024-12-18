CUDA_VISIBLE_DEVICES=1 python run_libsvm_experiment.py \
    --dataset_name libsvm \
    --task_name binary \
    --per_device_train_batch_size 1024 \
    --per_device_eval_batch_size 1024 \
    --hidden_dim 10 \
    --learning_rate 5e-2 \
    --num_epochs 10 \
    --optimizer_name new_soap \
    --beta 0.95 \
    --report_to none # none or wandb