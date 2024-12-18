import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import sklearn.model_selection
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from torch_optimizer import Shampoo

import numpy as np
import pandas as pd
import random
import sys
sys.path.append('../')
from src import new_soap, soap
import os


def set_global_seed(seed=18):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return g, seed_worker


def get_problem(config):
    g, seed_worker = set_global_seed(config['seed'])
    sampler = None
    if config['device_id'] is None:
        device = 'cpu'
    else:
        device = f'cuda:{config["device_id"]}' if torch.cuda.is_available() else 'cpu'
    transform_base = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if config['augment']:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transform_test
    ds = torchvision.datasets.CIFAR10('../data', train=True, transform=transform_base)
    train_idx, val_idx = train_test_split(np.arange(len(ds)), test_size=0.2, stratify=ds.targets, random_state=config['seed'])
    ds_train = Subset(ds, train_idx)
    ds_val = Subset(ds, val_idx)
    ds_test = torchvision.datasets.CIFAR10('../data', 
                                           train=False, transform=transform_base)
    d_out = 10
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    config['loss_scale'] = len(ds_train) / config['batch_size']
    train_dataloader = DataLoader(ds_train, 
                                  batch_size=config['batch_size'],
                                  worker_init_fn=seed_worker,
                              generator=g,
                              shuffle=sampler is None,
                              sampler=sampler,
                             )
    val_dataloader = DataLoader(ds_val, 
                              batch_size=config['batch_size'],
                              worker_init_fn=seed_worker,
                              generator=g,
                              shuffle=True,
                             )
    test_dataloader = DataLoader(ds_test, 
                              batch_size=config['batch_size'],
                              worker_init_fn=seed_worker,
                              generator=g,
                              shuffle=True,
                             )
    if config['model'] == 'resnet18':
        model = torchvision.models.resnet18().train()
        model.fc = nn.Linear(model.fc.in_features, d_out)
        
    model.to(device)

    if config['optimizer'] == 'new_soap':
        optimizer = new_soap.NEW_SOAP(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay'],
            momentum=config['momentum'], 
            update_freq=config['update_freq'],
        )
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == "soap":
        optimizer = soap.SOAP(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == "shampoo":
        optimizer = Shampoo(
            model.parameters(), 
            lr=config['lr']
        )
    elif config['optimizer'] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config['lr']
        )
    else:
        raise NotImplementedError(f"Unknown optimizer {config['optimizer']}")
    return model, optimizer, train_dataloader, val_dataloader, test_dataloader, loss_fn, device
