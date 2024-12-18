import numpy as np
import json
import wandb
from argparse import ArgumentParser
from collections import defaultdict
import os.path
import optuna

from problems import get_problem
from cifar_trainer import train


def parse_args() -> None:
    parser = ArgumentParser(description="Cifar10")
    parser.add_argument("--device", type=int, default=None, help="Device index for GPU")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--tune", action="store_true", help="Tune params")
    parser.add_argument("--use_old_tune_params", action="store_true", help="Use or old tunings")
    parser.add_argument("--augment", action="store_true", help="Use augmentation")
    return parser.parse_args()

def run_optimization(config, metrics, tuning=False):
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, \
        loss_fn, device = get_problem(config)
    _, val_results, test_results = train(model, optimizer, 
                                         train_dataloader, val_dataloader,
                                         test_dataloader, loss_fn, device, 
                                         config, tuning=tuning)
        
    if tuning:
        return np.max(val_results['f1'])
    idx = np.argmax(val_results['f1'])
    for metric in test_results:
        metrics[metric].append(test_results[metric][idx])
    return metrics 

def tune_params(config, name=None, use_old_tune_params=True):
    f_name = f'tuned_params/{name}.json'
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
        config["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        # config["lr"] = 1e-3
        config["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        return(run_optimization(config, None, tuning=True))
        
    study.optimize(tune_function, n_trials=config['tune_runs'])
    with open(f_name, 'w') as f:
        json.dump(study.best_trial.params, f)

    with open(f_name) as f:
        params = json.load(f)
        for key in params.keys():
            config[key] = params[key]
    
    config["lr"] = 1e-4
    return config

args = parse_args()
with open(args.config) as f:
    config = json.load(f)
config['device_id'] = args.device
config['augment'] = args.augment
if "report_to" not in config.keys(): config["report_to"] = "none"

# experiment_list = [
#     "adam", "soap", "new_soap", "sgd"
# ]
experiment_list = [
    "new_soap"
]

metrics = {}
for optimizer_name in experiment_list:
    config["optimizer"] = optimizer_name
    run_name = optimizer_name
    if run_name not in metrics.keys():
        metrics[run_name] = defaultdict(list)
    for i, seed in enumerate(range(config['eval_runs'])):
        config["seed"] = seed
        
        if args.tune:
            if i == 0:
                config = tune_params(config, run_name, args.use_old_tune_params)
            else: 
                config = tune_params(config, run_name, True)
        if config["report_to"] == "wandb":
            wandb.init(
                project = "MARTIN_ESCAPE",
                tags    = [f"CIFAR10 {config["model"]}"],
                name    = run_name,
                config  = config
            )
        metrics[run_name] = run_optimization(config, metrics[run_name])  
        if config["report_to"] == "wandb": wandb.finish()

    print(f'~~~~~~~~~~~ Run name {run_name} ~~~~~~~~~~~')
    for metric_name in metrics[run_name].keys():
        res = np.array(metrics[run_name][metric_name])
        print(f'{metric_name}: {res.mean()}+-{res.std()}')

print("===================== Config =====================")
print(config)
print("==========================================")