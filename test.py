import torch
import torch.utils
import torch.utils.data
from src import new_soap
import torch.nn as nn
import numpy as np
from sklearn.datasets import (
    load_svmlight_file, 
    make_circles,
    make_classification,
)

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

def train_one_epoch(model, train_loader, loss_fn, optimizer):
    running_loss = 0.
    running_acc = 0.
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.long()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        running_acc += torch.sum(labels == preds) / len(labels)
        # if num_verbose is not None and i % num_verbose == num_verbose-1:
        #     last_loss = running_loss / num_verbose
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     running_loss = 0.

    return running_loss / len(train_loader), running_acc / len(train_loader)

def main_classifiaction_sklearn():
    seed=18
    torch.manual_seed(seed)
    # dataset_name = "mushrooms.txt" 
    # data = load_svmlight_file(dataset_name)
    X, y = make_classification(random_state=seed, n_informative=4, n_classes=2)
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=X.dtype)
    dataset = torch.utils.data.TensorDataset(X, y)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

    model = LinearClassifier(input_dim=X.shape[1], hidden_dim=10,
                             output_dim=len(np.unique(y)), dtype=X.dtype)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = new_soap.NEW_SOAP(
        model.parameters(), 
        lr=5e-2, 
        beta=.05, 
        weight_decay=.01, 
        precondition_frequency=1,
        normalize_grads=True,
        correct_bias=True
    )
    # optimizer = soap.SOAP(model.parameters(), lr=5e-2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

    print(f"Using optimizer: {optimizer.__class__.__name__}")
    EPOCHS = 200
    NUM_VERBOSE = 20
    for epoch in range(EPOCHS):
        model.train(True)
        train_loss, train_acc = train_one_epoch(model, train_dataloader, 
                                                loss_fn, optimizer)
        
        running_vloss = 0.
        running_vacc = 0.
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(test_dataloader):
                vinputs, vlabels = vdata
                vlabels = vlabels.long()
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                vpreds = torch.argmax(voutputs, dim=1)
                running_vloss += vloss
                running_vacc += torch.sum(vlabels == vpreds) / len(vlabels)

        test_loss = running_vloss / (i + 1)
        test_acc = running_vacc / (i + 1)
        if (epoch+1) % NUM_VERBOSE == 0:
            print('#{}:'.format(epoch + 1), end=' ')
            print(f'LOSS: train {train_loss:.4f} test {test_loss:.4f}', end=' ; ')
            print(f'ACCURACY: train {train_acc:.4f} test {test_acc:.4f}')

def main_syntetic():
    seed=18
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    n_samples = 10
    n_features = 3
    n_target = 2
    X = torch.randn(n_samples, n_features)
    w = torch.randn(n_features, n_target)
    y = X @ w
        
    model = LinearClassifier(input_dim=X.shape[1], output_dim=y.shape[1],
                             dtype=X.dtype)
    criterion = torch.nn.MSELoss()
    optimizer = new_soap.NEW_SOAP(
        model.parameters(), 
        lr=5e-2, 
        beta=.05, 
        weight_decay=.01, 
        precondition_frequency=1,
        normalize_grads=False,
        correct_bias=False
    )

    num_epochs = 100
    num_verbose = 1
    for epoch in range(num_epochs):
        pred_y = model(X)
        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % num_verbose == 0:
            print('Epoch {}: loss {}'.format(epoch, loss.item()))
            #print("$"*60)

if __name__ == "__main__":
    main_classifiaction_sklearn()
    
    