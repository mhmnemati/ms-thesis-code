import numpy as np
import torch as pt
import torch.nn as nn

from torch.utils.data import DataLoader

from data import CHBMITDataset
from braindecode.models import Deep4Net, EEGInception


def transform(item):
    return (item["data"], item["label"])


BATCH = 8

train_set = CHBMITDataset(train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH)

test_set = CHBMITDataset(train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH)

model = EEGInception(n_outputs=2, n_chans=23, n_times=3000)

loss_fn = nn.CrossEntropyLoss()
optimizer = pt.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with pt.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(pt.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")

# print(f"Train set: {len(trainset)}")
# print(f"Test set: {len(testset)}")
