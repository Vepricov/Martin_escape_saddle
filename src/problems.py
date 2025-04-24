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
from optimizers import soap, adam_sania, muon, taia
from optimizers.diag_hvp import DiagonalPreconditionedOptimizer

import utils

def cifar_prepocess(args, d_out=10):
    g, seed_worker = utils.set_global_seed(args.seed)
    transform_base = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.not_augment:
        transform_train = transform_test
    else:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    ds = torchvision.datasets.CIFAR10(args.data_path, train=True, 
                                      transform=transform_base, download=True)
    train_idx, val_idx = train_test_split(np.arange(len(ds)), test_size=0.2, 
                                          stratify=ds.targets, random_state=args.seed)
    ds_train = Subset(ds, train_idx)
    ds_val = Subset(ds, val_idx)
    ds_train = utils.TransformdDataset(ds_train, transform=transform_train)
    ds_val = utils.TransformdDataset(ds_val, transform=transform_test)
    ds_test = torchvision.datasets.CIFAR10(args.data_path, train=False, 
                                           transform=transform_base)
    ds_test = utils.TransformdDataset(ds_test, transform=transform_test)
    
    train_dataloader = DataLoader(
        ds_train, 
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        ds_val, 
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        ds_test, 
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    if args.model == "resnet18":
        model = torchvision.models.resnet18().train()
        model.fc = nn.Linear(model.fc.in_features, d_out)
    else:
        raise ValueError(f"Wrong model name: {args.model}")

    return train_dataloader, val_dataloader, test_dataloader, loss_fn, model

def libsvm_prepocess(args):
    g, seed_worker = utils.set_global_seed(args.seed)
    if args.dataset == "mushrooms":
        if not os.path.exists(f"./{args.data_path}/mushrooms"):
            os.system(f"cd ./{args.data_path}\n wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms \n cd ..")
        X, y = load_svmlight_file(f"./{args.data_path}/mushrooms")
        y = y - 1
        X = X.toarray()
    if args.dataset == "binary":
        if not os.path.exists(f"./{args.data_path}/covtype.libsvm.binary.scale.bz2"):
            os.system(f"cd ./{args.data_path}\n wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2 \n cd ..")
        X, y = load_svmlight_file(f"./{args.data_path}/covtype.libsvm.binary.scale.bz2")
        y = y - 1
        X = X.toarray()

    A = None
    if args.scale:
        A = np.diag(np.exp(np.random.uniform(0, args.scale_bound, X.shape[1])))
        X = X @ A
    if args.rotate:
        B = np.random.random([X.shape[1], X.shape[1]])
        A, _ = np.linalg.qr(B.T @ B)
        X = X @ A
    
    # X = torch.tensor(X, dtype=torch.float32)
    X = torch.tensor(X)
    y = torch.tensor(y, dtype=X.dtype)

    ds = TensorDataset(X, y)
    ds_train, ds_val, ds_test = torch.utils.data.random_split(ds, [0.7, 0.2, 0.1],
                                                              generator=g)
    train_dataloader = DataLoader(
        ds_train, 
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        ds_val, 
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        ds_test, 
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    if args.model == "linear-classifier":
        model = utils.LinearClassifier(
            input_dim=X.shape[1], hidden_dim=args.hidden_dim, 
            output_dim=len(np.unique(y)), dtype=X.dtype, bias=not args.no_bias,
            weight_init=args.weight_init, A=A
        )
    else:
        raise ValueError(f"Wrong model name: {args.model} for dataset {args.dataset}")

    return train_dataloader, val_dataloader, test_dataloader, loss_fn, model

def get_problem(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.dataset == "cifar10":
        (train_dataloader, val_dataloader, test_dataloader, 
         loss_fn, model) = cifar_prepocess(args)
        main_metric = "f1"
    else:
        (train_dataloader, val_dataloader, test_dataloader, 
         loss_fn, model) = libsvm_prepocess(args)
        main_metric = "accuracy"
    model.to(device)

    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            params       = model.parameters(), 
            lr           = args.lr, 
            betas        = (args.beta1, args.beta2),
            eps          = args.eps,
            weight_decay = args.weight_decay,
        )
    elif args.optimizer == "soap":
        optimizer = soap.SOAP(
            params                 = model.parameters(), 
            lr                     = args.lr, 
            betas                  = (args.beta1, args.beta2),
            shampoo_beta           = args.shampoo_beta,
            eps                    = args.eps,
            weight_decay           = args.weight_decay,
            precondition_frequency = args.update_freq,
        )
    elif args.optimizer == "shampoo":
        optimizer = Shampoo(
            params       = model.parameters(), 
            lr           = args.lr, 
            momentum     = args.momentum,
            epsilon      = args.eps,
            weight_decay = args.weight_decay,
            update_freq  = args.update_freq,
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params       = model.parameters(), 
            lr           = args.lr, 
            momentum     = args.momentum,
            weight_decay = args.weight_decay,
        )
    elif args.optimizer == "adam-sania":
        optimizer = adam_sania.AdamSania(
            params       = model.parameters(), 
            lr           = args.lr, 
            betas        = (args.beta1, args.beta2),
            eps          = args.eps,
            weight_decay = args.weight_decay,
        )
    elif args.optimizer == "muon":
        optimizer = muon.Muon(
            muon_params  = list(model.parameters()), 
            lr           = args.lr, 
            adamw_betas  = (args.beta1, args.beta2),
            adamw_eps    = args.eps, 
            adamw_wd     = args.weight_decay,
            momentum     = args.momentum,
            ns_steps     = args.ns_steps,
        )
    elif args.optimizer == "taia":
        taia_params, adamw_params = [], []
        for i, param in enumerate(model.parameters()):
            if i == 0:
                taia_params.append(param)
            else:
                adamw_params.append(param)
        optimizer = taia.TAIA(
            taia_params       = taia_params, 
            adamw_params      = adamw_params,
            lr                = args.lr, 
            adamw_betas       = (args.beta1, args.beta2),
            adamw_eps         = args.eps, 
            adamw_wd          = args.weight_decay,
            momentum          = args.momentum,
            ns_steps          = args.ns_steps,
            lmo               = args.lmo,
            precondition_type = args.precondition_type,
            A = model.A,
        )
    elif args.optimizer == "diag-hvp":
        optimizer = DiagonalPreconditionedOptimizer(
            params = model.parameters(),
            lr = args.lr,
            eps = args.eps,
            update_freq = args.update_freq
        )
    else:
        raise NotImplementedError(f"Wrong optimizer name {args.optimizer}")
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
