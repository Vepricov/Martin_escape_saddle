for optimizer_name in adam soap new_soap shampoo sgd
# for optimizer_name in shampoo
do
    CUDA_VISIBLE_DEVICES=1 python run_libsvm_experiment.py \
        --dataset_name libsvm \
        --task_name make_classification \
        --per_device_train_batch_size 1024 \
        --per_device_eval_batch_size 1024 \
        --hidden_dim 10 \
        --n_samples 1000 \
        --n_informative 4 \
        --n_classes 2 \
        --learning_rate 5e-2 \
        --num_epochs 10 \
        --optimizer_name $optimizer_name \
        --report_to none # none or wandb
done