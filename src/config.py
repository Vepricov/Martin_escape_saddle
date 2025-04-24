from argparse import ArgumentParser
from termcolor import colored

def parse_args() -> None:
    parser1 = ArgumentParser(description="Main Experiment")

    ### Dataset and Model Arguments
    parser1.add_argument("--dataset", default=None, 
                         choices=["cifar10", 
                                  "mushrooms", 
                                  "binary"
                                  ])
    parser1.add_argument("--optimizer", type=str, default=None, 
                        help="Name of the optimizer to use",
                        choices=["adamw", 
                                 "soap", 
                                 "shampoo", 
                                 "sgd", 
                                 "adam-sania",
                                 "muon",
                                 "taia",
                                 "diag-hvp"
                                ])
    args1, _ = parser1.parse_known_args()
    parser = parser1
    if args1.dataset in ["cifar10"]:
        ### cifar10 Arguments
        model_default = "resnet18", 
        parser.add_argument("--not_augment", action="store_true", 
                            help="To not use the augmentation")
        batch_size_default, n_epoches_default, eval_runs_default = 64, 10, 5
        n_epoches_tune_default, tune_runs_default = 5, 100
    elif args1.dataset in ["mushrooms", "binary"]:
        ### Libsvm Arguments
        model_default = "linear-classifier"
        parser.add_argument("--hidden_dim", default=10, type=int,
                            help="Hidden dimatially of linear classifier")
        parser.add_argument("--no_bias", action="store_true",
                            help="No bias in the FCL of the linear classifier")
        parser.add_argument("--weight_init", default="uniform", 
                            choices=["zeroes", "uniform", "bad_scaled", "ones"],
                            help="Initial weights of the linear classifier")
        parser.add_argument("--scale", action="store_true",
                            help="Use or not scaling for the libsvm datasets")
        parser.add_argument("--scale_bound", default=20, type=int,
                            help="Scaling ~`exp[U(-scale_bound, scale_bound)]`")
        parser.add_argument("--rotate", action="store_true",
                            help="Use or not rotating for the libsvm datasets")
        batch_size_default, n_epoches_default, eval_runs_default = 64, 5, 3
        n_epoches_tune_default, tune_runs_default = 1, 20
        
    parser.add_argument("--model", default=model_default, help="Model name")

    ### Training Arguments
    parser.add_argument("--batch_size", default=batch_size_default, type=int)
    parser.add_argument("--n_epoches", default=n_epoches_default, type=int,
                        help="How many epochs to train")
    parser.add_argument("--eval_runs", default=eval_runs_default, type=int, 
                        help="Number of re-training model with diffirent seeds")
    parser.add_argument("--dtype", default="float32", type=str, 
                        help="Default type for torch")                 

    ### Tuning Arguments
    parser.add_argument("--tune", action="store_true", 
                        help="Tune params")
    parser.add_argument("--use_old_tune_params", action="store_true", 
                        help="Use already tuned params")
    parser.add_argument("--n_epoches_tune", default=n_epoches_tune_default, 
                        type=int, help="How many epochs to tune")
    parser.add_argument("--tune_runs", default=tune_runs_default, type=int, 
                        help="Number of optuna steps")

    ### Wandb Arguments
    parser.add_argument("--wandb", action="store_true", help="To use wandb")
    parser.add_argument("--run_prefix", default=None, help="To use wandb")
    parser.add_argument("--wandb_project", default="NEW_SOAP")
    parser.add_argument("--verbose", action="store_true", 
                        help="To print training resuls in the terminal")
    parser.add_argument("--seed", default=18, type=int)

    ### Saving Paths
    parser.add_argument("--results_path", default="results_raw", 
                        help="Path to save the results of the experiment")
    parser.add_argument("--tune_path", default="tuned_params", 
                        help="Path to save the tuned params")
    parser.add_argument("--data_path", default="data", 
                        help="Path to save the datasets")

    ### Otimizer Arguments
    parser.add_argument("--lr", default=None, type=float,
                        help="learning rate") # tuneed param
    parser.add_argument("--weight_decay", default=1e-6, type=float,
                        help="weight decay") # tuneed param
    if args1.optimizer not in ["shampoo", "sgd"]:
        parser.add_argument("--beta1", default=0.9, type=float,
                        help="First momentum")
        parser.add_argument("--beta2", default=0.999, type=float,
                            help="Second momentum")
        parser.add_argument("--eps", default=1e-8, type=float,
                            help="Epsolon for Adam")
    
    if args1.optimizer in ["shampoo", "sgd", "muon", "taia"]:
        parser.add_argument("--momentum", default=0.9, type=float,
                            help="First momentum")

    if args1.optimizer in ["soap"]:
        parser.add_argument("--shampoo_beta", default=-1, type=float,
                            help="momentum for SOAP. if -1, the equals to beta2")
    if args1.optimizer in ["shampoo", "soap", "diag-hvp"]:
        parser.add_argument("--update_freq", default=1, type=int,
                            help="Freqiensy to update Q for Shampoo and SOAP")
    if args1.optimizer in ["muon", "taia"]:
        parser.add_argument("--ns_steps", default=10, type=int,
                            help="Number of the NS steps algo")
    if args1.optimizer in ["taia"]:
        parser.add_argument("--lmo", default="frobenious", type=str,
                            choices=["frobenious", "spectral"],
                            help="LMO type for TAIA optimizer")
        parser.add_argument("--precondition_type", default="norm", type=str,
                            choices=["norm", "fisher"],
                            help="Preconditioning type for TAIA optimizer")
    
    args, unparced_args = parser.parse_known_args()
    if len(unparced_args) > 0:
        print(colored("~~~~~~~~~~~ WARNING: UNPARCED ARGS ~~~~~~~~~~~", "red"))
        line = f"You pass unrecognized arguments:"
        print(colored(line, "red"), end="")
        for arg in unparced_args:
            if "--" in arg:
                print(colored(f"\n{arg}", "red"), end=" ")
            else:
                print(colored(arg, "red"), end="")
        print()
    if not args.wandb and not args.verbose:
        print(colored("~~~~~~~~~~~ WARNING: NO VERBOSE ~~~~~~~~~~~", "yellow"))
        line = "wandb and verbose set to False, so we set verbose to True"
        print(colored(line, "yellow"))
        args.verbose = True
    return args, parser