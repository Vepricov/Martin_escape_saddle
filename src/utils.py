import json
import random
import torch
import datasets
import os
import numpy as np
import warnings
import torch.nn as nn
from sklearn.datasets import (
    load_svmlight_file, 
    make_circles,
    make_classification,
)

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

class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim=3, output_dim=1, w_0 = None, dtype=None):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, dtype=dtype)
        if w_0 is None:
            nn.init.zeros_(self.linear.weight)
    def forward(self, x):
        y = self.linear(x)
        return y
    
class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim = 10, 
                 output_dim=2, w_0 = None, dtype=None):
        super(LinearClassifier, self).__init__()
        self.net = nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim, dtype=dtype),
            nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim, dtype=dtype),
            torch.nn.Softmax(dim=1),
        )
        if w_0 is None:
            for layer in self.net:
                if hasattr(layer, "weight"):
                    nn.init.zeros_(layer.weight)
    def forward(self, x):
        out = self.net(x)
        return out