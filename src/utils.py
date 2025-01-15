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
            # f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            f"trainable params: {trainable_params}"
        )

    return all_param, trainable_params

def set_seed(seed): # ставит сид
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
            # torch.nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim, dtype=dtype),
            # torch.nn.Linear(hidden_dim, hidden_dim // 2, dtype=dtype),
            # torch.nn.BatchNorm1d(2 * hidden_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden_dim // 2, output_dim, dtype=dtype),
            torch.nn.Softmax(dim=1),
        )
        if w_0 is None:
            for layer in self.net:
                if hasattr(layer, "weight"):
                    nn.init.zeros_(layer.weight)
    def forward(self, x):
        out = self.net(x)
        return out

def libsvm_prepocess(data_args, model_args, training_args, device):
    if data_args.task_name == "make_classification":
        X, y = make_classification(random_state=data_args.dataset_seed, 
                                   n_samples=data_args.n_samples,
                                   n_informative=data_args.n_informative,
                                   n_classes=data_args.n_classes)
    if data_args.task_name == "mushrooms":
        if not os.path.exists('./data/mushrooms'):
            os.system("cd ./data\n wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms \n cd ..")
        X, y = load_svmlight_file("./data/mushrooms")
        y = y - 1
        warnings.warn("X is a sparse matrix, but we convert is to np.array :(")
        X = X.toarray()
    if data_args.task_name == "binary":
        if not os.path.exists('./data/covtype.libsvm.binary.scale.bz2'):
            os.system("cd ./data\n wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2 \n cd ..")
        X, y = load_svmlight_file("./data/covtype.libsvm.binary.scale.bz2")
        y = y - 1
        warnings.warn("X is a sparse matrix, but we convert is to np.array :(")
        X = X.toarray()

    if data_args.use_scaling:
        for j in range(X.shape[1]):
            a = np.random.uniform(-10, 10)
            X[:, j] *= np.exp(a)    
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=X.dtype)

    dataset = torch.utils.data.TensorDataset(X, y)
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3])
    training_args.train_len = len(train_dataset)
    training_args.eval_len = len(eval_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,  
        batch_size=training_args.per_device_train_batch_size, 
        shuffle=True
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=training_args.per_device_eval_batch_size, 
        shuffle=True
    )

    model = LinearClassifier(input_dim=X.shape[1], hidden_dim=model_args.hidden_dim,
                             output_dim=len(np.unique(y)), dtype=X.dtype)

    return train_dataloader, eval_dataloader, model.to(device)