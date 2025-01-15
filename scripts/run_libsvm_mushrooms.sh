for optimizer_name in adam soap new_soap shampoo sgd
# for optimizer_name in soap
do
    CUDA_VISIBLE_DEVICES=1 python run_libsvm_experiment.py \
        --dataset_name libsvm \
        --task_name mushrooms \
        --per_device_train_batch_size 1024 \
        --per_device_eval_batch_size 1024 \
        --hidden_dim 10 \
        --learning_rate 5e-2 \
        --num_epochs 10 \
        --use_scaling true \
        --optimizer_name $optimizer_name \
        --report_to wandb # none or wandb
done