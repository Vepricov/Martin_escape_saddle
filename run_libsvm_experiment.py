import torch, gc, os, wandb
from transformers import HfArgumentParser
from src import config, my_trainer, utils


def main():
    for i in range(torch.cuda.device_count()):
        print("We will use the GPU:", torch.cuda.get_device_name(i))
    parser = HfArgumentParser((
        config.ModelArguments, 
        config.DataTrainingArguments, 
        config.TrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    utils.set_seed(training_args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ################# Model, Tokenizer and Dataset Downloading #################
    if data_args.dataset_name == "libsvm":
        train_dataset, eval_dataset, model = utils.libsvm_prepocess(
            data_args, 
            model_args,
            training_args,
            device
        )
    else:
        raise ValueError(f"Wrong dataset name: {data_args.dataset_name}!")
    ################################## Wandb ###################################
    run_name = f"[{training_args.optimizer_name}] {data_args.task_name}"
    # run_name = "[TEST]"
    training_args.output_dir = f"{training_args.output_dir}/{run_name}"
    training_args.benchmark_name = data_args.dataset_name
    training_args.tsk_name = data_args.task_name
    if "wandb" in training_args.report_to:
        wandb.init(
            project = "MARTIN_ESCAPE",
            tags    = [f"{data_args.dataset_name} {data_args.task_name}"],
            name    = run_name,
            config  = training_args.to_dict(),
            dir     = f"./train_outputs/{run_name}"
        )
    print("$"*(39-len(run_name)//2), run_name, "$"*(39-len(run_name)//2))
    print(model)
    print(f"Len of train / eval datasets = {training_args.train_len} / {training_args.eval_len}")
    ############################# Training #####################################
    trainer = my_trainer.Trainer(
        model=model,
        training_args=training_args,
        train_dataloader=train_dataset if training_args.do_train else None,
        eval_dataloader=eval_dataset if training_args.do_evaluate else None,
        device=device
    )
    if training_args.do_train:
        trainer.train(num_verbose = 1)
        if "wandb" in training_args.report_to:
            wandb.finish()
    print("$"*(80+len(run_name)%2))
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()