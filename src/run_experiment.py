import numpy as np
import json
import wandb
import torch
from collections import defaultdict
import os.path
import optuna

from problems import get_problem
from trainer import train
from utils import get_run_name
from config import parse_args

def run_optimization(args, metrics, tuning=False, verbose=False):
    (model, optimizer, train_dataloader, val_dataloader, test_dataloader,
     loss_fn, main_metric, device) = get_problem(args)
    _, val_results, test_results = train(
        model, optimizer, 
        train_dataloader, val_dataloader,
        test_dataloader, loss_fn, device, 
        args, tuning=tuning, verbose=verbose
    )
        
    if tuning:
        return np.max(val_results[f"val_{main_metric}"])
    idx = np.argmax(val_results[f"val_{main_metric}"])
    for metric in test_results:
        metrics[metric].append(test_results[metric][idx])
    return metrics 

def tune_params(args, parser, use_old_tune_params=True):
    tune_name = get_run_name(args, parser, tuning=True)
    f_name = f"{args.tune_path}/{tune_name}.json"
    if os.path.exists(f_name) and use_old_tune_params:
        try:
            with open(f_name) as f:
                params = json.load(f)
                for key in params.keys():
                    setattr(args, key, params[key])
            return args, params.keys()
        except json.decoder.JSONDecodeError:
            pass

    study = optuna.create_study(direction="maximize", 
                                study_name=f"{tune_name}")
    def tune_function(trial):
        args.lr = trial.suggest_float("lr", 1e-6, 5e0, log=True)
        if hasattr(args, "weight_decay"):
            args.weight_decay = trial.suggest_float("weight_decay", 
                                                    1e-6, 1e-2, log=True)

        return(run_optimization(args, None, tuning=True, verbose=False))
        
    study.optimize(tune_function, n_trials=args.tune_runs)
    with open(f_name, "w") as f:
        json.dump(study.best_trial.params, f)

    with open(f_name) as f:
        params = json.load(f)
        for key in params.keys():
            setattr(args, key, params[key])
    
    return args, params.keys()

def main(args, parser):
    os.makedirs(f"./{args.results_path}", exist_ok=True)
    os.makedirs(f"./{args.tune_path}", exist_ok=True)
    os.makedirs(f"./{args.data_path}", exist_ok=True)

    metrics = defaultdict(list)
    run_name = get_run_name(args, parser, tuning=False)
    if args.tune or args.use_old_tune_params:
        if args.use_old_tune_params is False:
            print("~~~~~~~~~~~ TUNING ~~~~~~~~~~~")
        args, tuned_params = tune_params(args, parser, 
                                         use_old_tune_params=args.use_old_tune_params)
        if len(tuned_params) > 0:
            print("~~~~~~~~~~~ TUNED PARAMS ~~~~~~~~~~~")
            for param_name in tuned_params:
                print(f"{param_name}: {getattr(args, param_name)}")
    
    for i, seed in enumerate(range(args.eval_runs)):
        args.seed = seed
        if args.wandb:
            wandb.init(
                project = args.wandb_project,
                tags    = [args.dataset, args.optimizer],
                name    = run_name,
                config  = args
            )

        print(f"~~~~~~~~~~~ TRAIN RUN {i+1}/{args.eval_runs} ~~~~~~~~~~~")
        metrics = run_optimization(args, metrics, 
                                   tuning=False, verbose=args.verbose)
        if args.wandb: wandb.finish()

    with open(f"./{args.results_path}/{run_name}.txt", "w") as f:
        f.write(f"~~~~~~~~~~~ {run_name} ~~~~~~~~~~~\n")
        for metric_name in metrics.keys():
            res = np.array(metrics[metric_name])
            f.write(f"{metric_name}: {res.mean()}+-{res.std()}\n")

    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

if __name__ == "__main__":
    print("~~~~~~~~~~~ GPU ~~~~~~~~~~~")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
    args, parser = parse_args()
    main(args, parser)