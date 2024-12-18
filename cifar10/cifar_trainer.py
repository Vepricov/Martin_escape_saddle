import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm
import wandb

from collections import defaultdict

def train_step(model, optimizer, dataloader, loss_fn, device, config, tuning=False):
    model.train()
    total_loss = 0
    for t, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        preds = model(X).to(device)
        losses = loss_fn(preds, y)
        loss = losses.mean().to(device)
        if config["report_to"] == "wandb" and not tuning:
            wandb.log({"train_loss": loss.item(), "train_step" : config["train_step"]})
        config["train_step"] += 1
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        #print(loss.item(), end=' ')
    #print("")

    return total_loss / t


@torch.no_grad
def eval_step(model, dataloader, loss_fn, device, config, validation=True, tuning=False):
    model.eval()
    total_loss = 0
    total_true = np.array([])
    total_pred = np.array([])
    for t, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        preds = model(X).to(device)
        losses = loss_fn(preds, y)
        y_pred = preds.argmax(dim=-1)
        total_true = np.append(total_true, y.cpu().detach().numpy())
        total_pred = np.append(total_pred, y_pred.cpu().detach().numpy())
        loss = losses.mean()
        total_loss += loss.item()
        if validation and config["report_to"] == "wandb" and not tuning:
            wandb.log({"val_loss": loss.item(), "val_step": config["val_step"]})
            config["val_step"] += 1
        elif config["report_to"] == "wandb" and not tuning:
            wandb.log({"test_loss": loss.item(), "test_step": config["test_step"]})
            config["test_step"] += 1

    average = 'weighted' if len(np.unique(total_true)) > 2 else 'binary'
    f1 = f1_score(total_true, total_pred, average=average)
    precision = precision_score(total_true, total_pred, zero_division=0.0, average=average)
    recall = recall_score(total_true, total_pred, average=average)
    results = {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    return total_loss / t, results


def train(model, optimizer, train_dataloader, val_dataloader, 
          test_dataloader, loss_fn, device, config, tuning=False):
    val_metrics = defaultdict(list)
    test_metrics = defaultdict(list)
    config["train_step"], config["val_step"], config["test_step"] = 0, 0, 0
    if "n_epoches_tune" not in config: config["n_epoches_tune"] = config["n_epoches"]
    e_list = range(config['n_epoches_tune']) if tuning else tqdm(range(config['n_epoches']))
    for e in e_list:
        train_loss = train_step(model, optimizer, train_dataloader, 
                                loss_fn, device, config, tuning=tuning)
        
        _, val_results = eval_step(model, val_dataloader, loss_fn, 
                                   device, config, tuning=tuning)

        for key in val_results:
            val_metrics[key].append(val_results[key])
        
        _, test_results = eval_step(model, test_dataloader, loss_fn, device, 
                                    config, validation=False, tuning=tuning)
        
        if config["report_to"] == "wandb" and not tuning:
            wandb.log({'val_f1'         : val_results["f1"], 
                       'val_precision'  : val_results["precision"], 
                       'val_recall'     : val_results["recall"],
                       'test_f1'        : test_results["f1"], 
                       'test_precision' : test_results["precision"], 
                       'test_recall'    : test_results["recall"],
                       'epoch'          : e + 1})
        for key in test_results:
            test_metrics[key].append(test_results[key])
    
    return model, val_metrics, test_metrics