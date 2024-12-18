import torch
from src import new_soap, soap
from transformers import get_scheduler
from torch_optimizer import Shampoo
import wandb
def get_optimizer(model, training_args):
    if training_args.optimizer_name == "new_soap":
        optimizer = new_soap.NEW_SOAP(
            model.parameters(), 
            lr=training_args.learning_rate, 
            beta=training_args.beta, 
            weight_decay=training_args.weight_decay, 
            precondition_frequency=training_args.precondition_frequency,
            normalize_grads=False,
            correct_bias=False
        )
    elif training_args.optimizer_name == "soap":
        optimizer = soap.SOAP(model.parameters(), lr=training_args.learning_rate)
    elif training_args.optimizer_name == "shampoo":
        optimizer = Shampoo(model.parameters(), lr=training_args.learning_rate)
    elif training_args.optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    elif training_args.optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=training_args.learning_rate)
    else:
        raise ValueError(f"Wrong optimizer name {training_args.optimizer_name}")
    return optimizer

def get_scheduler(optimizer, training_args):
    if training_args.scheduler_name is not None:
        raise NotImplementedError
    return None


def train_one_epoch(model, train_loader, loss_fn, optimizer, report_to,
                    device="cpu"):
    running_loss = 0.
    running_acc = 0.
    for i, data in enumerate(train_loader):
        inputs, labels = data
        labels = labels.type(torch.LongTensor).to(device)
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(labels == preds) / len(labels)
        running_acc += acc
        if "wandb" in report_to:
            wandb.log({"train_loss": loss, "train_accuracy" : acc})
        # if num_verbose is not None and i % num_verbose == num_verbose-1:
        #     last_loss = running_loss / num_verbose
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     running_loss = 0.

    return running_loss / len(train_loader), running_acc / len(train_loader)

class Trainer():
    def __init__(self, model, training_args, loss=None,
                 train_dataloader=None, eval_dataloader=None, device="cpu"):
        self.model = model
        self.training_args = training_args
        self.optimizer = get_optimizer(self.model, training_args)
        self.schduler = get_scheduler(self.optimizer, training_args)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        if loss is None:
            self.loss = torch.nn.CrossEntropyLoss()
        self.device = device

    def train(self, num_verbose = 10):
        for epoch in range(self.training_args.num_epochs):
            self.model.train(True)
            train_loss, train_acc = train_one_epoch(
                self.model, 
                self.train_dataloader, 
                self.loss, self.optimizer,
                self.training_args.report_to, self.device
            )
            
            running_vloss = 0.
            running_vacc = 0.
            self.model.eval()
            with torch.no_grad():
                for i, vdata in enumerate(self.eval_dataloader):
                    vinputs, vlabels = vdata
                    vinputs = vinputs.to(self.device)
                    vlabels = vlabels.type(torch.LongTensor).to(self.device)
                    vlabels = vlabels
                    voutputs = self.model(vinputs)
                    vloss = self.loss(voutputs, vlabels)
                    vpreds = torch.argmax(voutputs, dim=1)
                    running_vloss += vloss
                    running_vacc += torch.sum(vlabels == vpreds) / len(vlabels)

            eval_loss = running_vloss / (i + 1)
            eval_acc = running_vacc / (i + 1)
            if "wandb" in self.training_args.report_to:
                wandb.log({"val_loss": eval_loss, "val_accuracy" : eval_acc})
            if (epoch+1) % num_verbose == 0:
                if epoch < 9:
                    print('Epoch#0{}:'.format(epoch + 1), end=' ')
                else:
                    print('Epoch#{}:'.format(epoch + 1), end=' ')
                print(f'LOSS: train {train_loss:.4f} eval {eval_loss:.4f}', end=' ; ')
                print(f'ACCURACY: train {train_acc:.4f} eval {eval_acc:.4f}')