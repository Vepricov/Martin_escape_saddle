import torch
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm.auto import tqdm
import wandb

from collections import defaultdict

def train_step(model, optimizer, dataloader, loss_fn, device, args, 
               tuning=False, epoch=0, verbose=False, train_config=None):
    model.train()
    total_loss = 0
    for t, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.type(torch.LongTensor).to(device)
        
        def closure():
            if hasattr(closure, 'has_been_called') and closure.has_been_called:
                # Don't zero_grad on subsequent calls to preserve computational graph
                preds = model(X).to(device)
                losses = loss_fn(preds, y)
                loss = losses.mean().to(device)
                loss.backward(retain_graph=True)
                return loss
            else:
                optimizer.zero_grad()
                preds = model(X).to(device)
                losses = loss_fn(preds, y)
                loss = losses.mean().to(device)
                loss.backward(retain_graph=True)
                closure.has_been_called = True
                return loss
        
        # Step with closure for optimizers that need it (like HVP)
        loss = optimizer.step(closure)
        
        if args.wandb and not tuning:
            wandb.log({"train_loss": loss.item(), 
                       "train_step" : train_config["train_step"]})
        if verbose and not tuning \
            and round((t+1)/len(dataloader), 1)-round(t/len(dataloader), 1)>0:
            line = f"[TRAIN {epoch+t/len(dataloader):.1f}/{args.n_epoches}] train_loss {loss.item():.4f}"
            print(f"{line}")
        train_config["train_step"] += 1
        total_loss += loss.item()

    return total_loss / t

@torch.no_grad
def eval_step(model, dataloader, loss_fn, device, args, validation=True): # + optimizer_name
    model.eval()
    total_loss = 0
    total_true = np.array([])
    total_pred = np.array([])
    for t, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.type(torch.LongTensor).to(device)
        preds = model(X).to(device)
        losses = loss_fn(preds, y)
        y_pred = preds.argmax(dim=-1)
        total_true = np.append(total_true, y.cpu().detach().numpy())
        total_pred = np.append(total_pred, y_pred.cpu().detach().numpy())
        loss = losses.mean()
        total_loss += loss.item()


    prefix = "val" if validation else "test"
    if args.dataset in ["cifar10"]:
        average = 'weighted' if len(np.unique(total_true)) > 2 else 'binary'
        f1 = f1_score(total_true, total_pred, average=average)
        precision = precision_score(total_true, total_pred, zero_division=0.0, average=average)
        recall = recall_score(total_true, total_pred, average=average)
        results = {
            f'{prefix}_f1': f1,
            f'{prefix}_precision': precision,
            f'{prefix}_recall': recall
        }
    else:
        accuracy = accuracy_score(total_true, total_pred)
        results = {
            f'{prefix}_accuracy': accuracy
        }
    results[f"{prefix}_loss"] = total_loss / t

    return results

def train(model, optimizer, train_dataloader, val_dataloader, 
          test_dataloader, loss_fn, device, args, tuning=False, verbose=False):
    train_config = {}
    train_config["train_step"] = 0
    n_epoches = args.n_epoches_tune if tuning else args.n_epoches
    e_list = range(n_epoches) if tuning or args.verbose else tqdm(range(n_epoches))
    val_metrics = defaultdict(list)
    test_metrics = defaultdict(list)
    for e in e_list:
        _ = train_step(
            model, optimizer, train_dataloader, 
            loss_fn, device, args, tuning=tuning, verbose=verbose, 
            train_config=train_config, epoch=e,
        )
        val_results = eval_step(
            model, val_dataloader, loss_fn, 
            device, args, validation=True
        )
                
        test_results = eval_step(
            model, test_dataloader, loss_fn, 
            device, args, validation=False
        )
        
        for key in val_results.keys():
            val_metrics[key].append(val_results[key])
        for key in test_results.keys():
            test_metrics[key].append(test_results[key])
        
        if args.wandb and not tuning:
            wandb_config = val_results | test_results
            wandb_config["epoch"] = e + 1
            wandb.log(wandb_config)
        
        if verbose and not tuning:
            line = f">>>[VAL  {e+1}/{args.n_epoches}] "
            for key in val_results.keys():
                line += f"{key.split("_")[1]} {val_results[key]:.4f} "
            print(f"{line}")
            line = f">>>[TEST {e+1}/{args.n_epoches}] "
            for key in test_results.keys():
                line += f"{key.split("_")[1]} {test_results[key]:.4f} "
            print(f"{line}")
    return model, val_metrics, test_metrics