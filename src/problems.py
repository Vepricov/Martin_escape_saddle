import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file, make_classification

from torch_optimizer import Shampoo
from optimizers import new_soap, soap

import utils

def cifar_prepocess(config, d_out=10):
    g, seed_worker = utils.set_global_seed(config["seed"])
    transform_base = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if config["augment"]:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transform_test

    ds = torchvision.datasets.CIFAR10("./data", train=True, 
                                      transform=transform_base, download=True)
    train_idx, val_idx = train_test_split(np.arange(len(ds)), test_size=0.2, 
                                          stratify=ds.targets, random_state=config["seed"])
    ds_train = Subset(ds, train_idx)
    ds_val = Subset(ds, val_idx)
    ds_train = utils.TransformdDataset(ds_train, transform=transform_train)
    ds_val = utils.TransformdDataset(ds_val, transform=transform_test)
    ds_test = torchvision.datasets.CIFAR10("./data", train=False, transform=transform_base)
    ds_test = utils.TransformdDataset(ds_test, transform=transform_test)
    
    train_dataloader = DataLoader(
        ds_train, 
        batch_size=config["batch_size"],
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        ds_val, 
        batch_size=config["batch_size"],
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        ds_test, 
        batch_size=config["batch_size"],
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    if config["model"] == "resnet18":
        model = torchvision.models.resnet18().train()
        model.fc = nn.Linear(model.fc.in_features, d_out)
    else:
        raise ValueError(f"Wrong model name: {config["model"]}")

    return train_dataloader, val_dataloader, test_dataloader, loss_fn, model

def libsvm_prepocess(config):
    g, seed_worker = utils.set_global_seed(config["seed"])
    if config["task"] == "MakeClassification":
        X, y = make_classification(
            n_samples=config["n_samples"],
            n_informative=config["n_informative"],
            n_classes=config["n_classes"],
            random_state=config["seed"], 
        )
    if config["task"] == "Mushrooms":
        if not os.path.exists('./data/mushrooms'):
            os.system("cd ./data\n wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms \n cd ..")
        X, y = load_svmlight_file("./data/mushrooms")
        y = y - 1
        X = X.toarray()
    if config["task"] == "Binary":
        if not os.path.exists('./data/covtype.libsvm.binary.scale.bz2'):
            os.system("cd ./data\n wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2 \n cd ..")
        X, y = load_svmlight_file("./data/covtype.libsvm.binary.scale.bz2")
        y = y - 1
        X = X.toarray()

    if config["scale"]:
        X *= np.exp(np.random.uniform(-10, 10, X.shape[1]))
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=X.dtype)

    ds = TensorDataset(X, y)
    ds_train, ds_val, ds_test = torch.utils.data.random_split(ds, [0.7, 0.2, 0.1])
    train_dataloader = DataLoader(
        ds_train, 
        batch_size=config["batch_size"],
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        ds_val, 
        batch_size=config["batch_size"],
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        ds_test, 
        batch_size=config["batch_size"],
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    if config["model"] == "LinearClassifier":
        hidden_dim = config["hidden_dim"] if "hidden_dim" in config.keys() else 10
        model = utils.LinearClassifier(input_dim=X.shape[1], 
                                       hidden_dim=hidden_dim,
                                       output_dim=len(np.unique(y)), dtype=X.dtype)
    else:
        raise ValueError(f"Wrong model name: {config["model"]}")

    return train_dataloader, val_dataloader, test_dataloader, loss_fn, model

def get_problem(config):
    #device_id = 0 if "device_id" not in config.keys() else config["device_id"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if config["task"] == "Cifar10":
        (train_dataloader, val_dataloader, test_dataloader, 
         loss_fn, model) = cifar_prepocess(config)
        main_metric = "f1"
    elif config["task"] in ["MakeClassification", "Mushrooms", "Binary"]:
        (train_dataloader, val_dataloader, test_dataloader, 
         loss_fn, model) = libsvm_prepocess(config)
        main_metric = "accuracy"
    else:
        raise ValueError(f"Wrong task name: {config["task"]}")
    model.to(device)

    default_optim_params = {
        "lr"           : 1e-3,
        "betas"        : (0.9, 0.999),
        "momentum"     : 0,
        "eps"          : 1e-8,
        "weight_decay" : 0.,
        "shampoo_beta" : -1,
        "prec_freq"    : 1,

    }
    for param in default_optim_params.keys():
        if param not in config.keys():
            config[param] = default_optim_params[param]

    if config["optimizer"] == "adam":
        optimizer = optim.Adam(
            params       = model.parameters(), 
            lr           = config["lr"], 
            betas        = config["betas"],
            eps          = config["eps"],
            weight_decay = config["weight_decay"],
        )
    elif config["optimizer"] == "soap":
        optimizer = soap.SOAP(
            params                 = model.parameters(), 
            lr                     = config["lr"], 
            betas                  = config["betas"],
            shampoo_beta           = config["shampoo_beta"],
            eps                    = config["eps"],
            weight_decay           = config["weight_decay"],
            precondition_frequency = config["prec_freq"],
        )
    elif config["optimizer"] == "shampoo":
        optimizer = Shampoo(
            params       = model.parameters(), 
            lr           = config["lr"],
            momentum     = config["momentum"],
            epsilon      = config["eps"],
            weight_decay = config["weight_decay"],
            update_freq  = config["prec_freq"],
        )
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            params       = model.parameters(), 
            lr           = config["lr"],
            momentum     = config["momentum"],
            weight_decay = config["weight_decay"],
        )
    # if config["optimizer"] == "new_soap":
    #     optimizer = new_soap.NEW_SOAP(
    #         model.parameters(), 
    #         lr=config["lr"], 
    #         weight_decay=config["weight_decay"],
    #         momentum=config["momentum"], 
    #         update_freq=config["update_freq"],
    #     )
    else:
        raise NotImplementedError(f"Wrong optimizer name {config["optimizer"]}")
    return (
        model, 
        optimizer, 
        train_dataloader, 
        val_dataloader, 
        test_dataloader, 
        loss_fn, 
        main_metric,
        device
    )
