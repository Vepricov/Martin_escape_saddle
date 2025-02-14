import numpy as np
import json
import wandb
import torch
from argparse import ArgumentParser
from collections import defaultdict
import os.path
import optuna

from problems import get_problem
from my_trainer import train


def parse_args() -> None:
    parser = ArgumentParser(description="Main Experiment")
    parser.add_argument("--config", type=str, default="config.json", 
                        help="Config file path")
    parser.add_argument("--optimizer_name", type=str, default="adam", 
                        help="Name of the optimizer to use",
                        choices=["adam", "soap", "shampoo", "sgd"])
    parser.add_argument("--tune", action="store_true", 
                        help="Tune params")
    parser.add_argument("--use_old_tune_params", action="store_true", 
                        help="Use already tuned params. Thay must be in a folder ```./tuned_params/{taks_name}```")
    parser.add_argument("--augment", action="store_true", 
                        help="Use augmentation (for cifar)")
    parser.add_argument("--scale", type=int, default=0, choices=[0, 1],
                        help="Use or not scaling for the libsvm datasets (0 - false, 1-true)")
    parser.add_argument("--verbose", action="store_true", 
                        help="To print training resuls in the terminal")
    parser.add_argument("--results_file_name", default="results_tmp", 
                        help="File name to save results of the experiment")
    return parser.parse_args()

def run_optimization(config, metrics, tuning=False, verbose=False):
    (model, optimizer, train_dataloader, val_dataloader, test_dataloader,
     loss_fn, main_metric, device) = get_problem(config)
    _, val_results, test_results = train(
        model, optimizer, 
        train_dataloader, val_dataloader,
        test_dataloader, loss_fn, device, 
        config, tuning=tuning, verbose=verbose
    )
        
    if tuning:
        return np.max(val_results[f"val_{main_metric}"])
    idx = np.argmax(val_results[f"val_{main_metric}"])
    for metric in test_results:
        metrics[metric].append(test_results[metric][idx])
    return metrics 

def tune_params(config, name=None, use_old_tune_params=True, verbose=False):
    folder_name = f"tuned_params/{config["task"]}_{config["n_epoches_tune"]}_ep"
    if not os.path.exists(folder_name) and use_old_tune_params:
        os.makedirs(folder_name, exist_ok=True)
    f_name = f"{folder_name}/{name}.json"
    if os.path.exists(f_name) and use_old_tune_params:
        try:
            with open(f_name) as f:
                params = json.load(f)
                for key in params.keys():
                    config[key] = params[key]
            return config
        except json.decoder.JSONDecodeError:
            pass

    study = optuna.create_study(direction="maximize", 
                                study_name=f"{name}")
    def tune_function(trial):
        config["lr"] = trial.suggest_float("lr", 1e-6, 5e0, log=True)
        config["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        return(run_optimization(config, None, tuning=True, verbose=verbose))
        
    study.optimize(tune_function, n_trials=config["tune_runs"])
    with open(f_name, "w") as f:
        json.dump(study.best_trial.params, f)

    with open(f_name) as f:
        params = json.load(f)
        for key in params.keys():
            config[key] = params[key]
    
    return config

args = parse_args()
with open(args.config) as f:
    config = json.load(f)

os.makedirs("./results_raw", exist_ok=True)
os.makedirs("./data", exist_ok=True)
os.makedirs("./tuned_params", exist_ok=True)
config["augment"] = args.augment
if "report_to" not in config.keys(): config["report_to"] = "none"
for i in range(torch.cuda.device_count()):
    print("We will use the GPU:", torch.cuda.get_device_name(i))

metrics = {}
config["optimizer"] = args.optimizer_name
config["scale"] = bool(args.scale)
run_name = args.optimizer_name
if config["scale"]: run_name += "_scaled"
if run_name not in metrics.keys():
    metrics[run_name] = defaultdict(list)

for i, seed in enumerate(range(config["eval_runs"])):
    config["seed"] = seed
    if args.tune or args.use_old_tune_params:
        if i == 0:
            config = tune_params(config, run_name, 
                                 use_old_tune_params=args.use_old_tune_params,
                                 verbose=args.verbose)
        else: 
            config = tune_params(config, run_name, use_old_tune_params=True,
                                 verbose=args.verbose)
    if config["report_to"] == "wandb":
        wandb.init(
            project = "NEW_SOAP",
            tags    = [f"{config["task"]}"],
            name    = f"[{config["task"]}] {run_name}",
            config  = config
        )
    print(f"~~~~~~~~~~~ Run name {run_name} ~~~~~~~~~~~")
    metrics[run_name] = run_optimization(config, metrics[run_name], 
                                         tuning=False, verbose=args.verbose)
    if config["report_to"] == "wandb": wandb.finish()

with open(f"./results_raw/{args.results_file_name}.txt", "a") as f:
    f.write(f"~~~~~~~~~~~ Run name {run_name} ~~~~~~~~~~~\n")
    for metric_name in metrics[run_name].keys():
        res = np.array(metrics[run_name][metric_name])
        f.write(f"{metric_name}: {res.mean()}+-{res.std()}\n")