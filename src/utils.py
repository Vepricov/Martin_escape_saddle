import random
import torch
import os
import numpy as np
import torch.nn as nn

def print_trainable_parameters(model, verbose=True):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    for param in model.buffers():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if verbose:
        print(
            f"trainable params: {trainable_params}"
        )

    return all_param, trainable_params

def set_global_seed(seed=18):
    def seed_worker(worker_seed):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return g, seed_worker

class TransformdDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self._dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, i):
        X, y = self._dataset[i]
        if self.transform is not None:
            X = self.transform(X)
        return X, y

def set_device(device_no: int): # выбирает GPU-шку и выводит название
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_no}")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(device_no))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    return device

def get_run_name(args, parser, tuning=False):
    key_args = ["dataset", "optimizer", "model"]
    ignore_args = [
        "verbose", "seed", "run_prefix", "wandb_project",
        "results_path", "tune_path", "data_path", "wandb",
        "tune", "use_old_tune_params", "eval_runs", "augment"
    ]
    ignore_args_tuning = [
        "n_epoches", "lr", "weight_decay", "scale", "rotate", "scale_bound",
        "weight_init", "ns_steps"
    ]
    # Get the default values
    defaults = vars(parser.parse_args([]))

    # Generate the prefix with key arguments
    if not tuning: print("~~~~~~~~~~~ KEY ARGUMENTS ~~~~~~~~~~~")
    prefix_parts = []
    for key in key_args:
        if hasattr(args, key):
            value = getattr(args, key)
            if not tuning: print(f"{key}: {value}")
            if value != defaults[key]: prefix_parts.append(f"{value}")

    prefix = "_".join(prefix_parts)

    # Generate the rest of the string with non-default arguments
    non_default_parts = []
    for key, value in vars(args).items():
        if key in ignore_args:
            continue
        if key in ignore_args_tuning and tuning:
            continue
        if key not in defaults:
            print(f"Warning: {key} not in defaults")
            continue
        if key not in key_args and value != defaults[key]:
            if type(value) == bool:
                non_default_parts.append(f"{key}")
            else:
                non_default_parts.append(f"{key}-{value}")
            if len(non_default_parts) == 1 and not tuning:
                print("~~~~~~~~~~~ NON-DEFAULT ARGUMENTS ~~~~~~~~~~~")
            if not tuning: print(f"{key}: {value}")

    non_default_string = "__".join(non_default_parts)

    if args.run_prefix is not None and not tuning:
        prefix = args.run_prefix + "__" + prefix

    # Combine prefix and non-default string
    if non_default_string:
        return f"{prefix}__{non_default_string}"
    else:
        return prefix


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim=3, output_dim=1, w_0 = None, dtype=None):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, dtype=dtype)
        if w_0 is None:
            nn.init.zeros_(self.linear.weight)
    def forward(self, x):
        y = self.linear(x)
        return y

def zero_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
def uniform_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=10, output_dim=2, 
                 weight_init="uniform", dtype=None, bias=True, A = None):
        super(LinearClassifier, self).__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim, dtype=dtype, bias=bias),
                # nn.Dropout(p=0.1),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim, dtype=dtype, bias=bias),
                torch.nn.Softmax(dim=1),
            )
        else:
            self.net = nn.Sequential(
                torch.nn.Linear(input_dim, output_dim, dtype=dtype, bias=bias),
                torch.nn.ReLU(),
                torch.nn.Softmax(dim=1),
            )

        if weight_init == "zeroes": 
            init_fn = zero_init_weights
        else:
            init_fn = uniform_init_weights
        self.net.apply(init_fn)
        if weight_init == "bad_scaled":
            with torch.no_grad():
                val_w, val_b = 1e2, 1e2
                for layer in self.net:
                    if hasattr(layer, "weight"):
                        layer.weight.data *= val_w
                        val_w = val_w**(-1)
                    if hasattr(layer, "bias"):
                        layer.weight.data *= val_b
                        val_b = val_b**(-1)
        if A is not None:
            A_inv = torch.tensor(np.linalg.inv(A), dtype=dtype)
            self.net[0].weight.data = self.net[0].weight.data @ A_inv.T
            self.A = torch.tensor(A)
        else:
            self.A = None

    def forward(self, x):
        out = self.net(x)
        return out