import os
import glob
import datetime
import numpy as np
import torch as pt
import torch.nn as nn
import torcheval.metrics as metrics

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
metric_fn = metrics.MulticlassAccuracy()
optimizer = pt.optim.SGD(model.parameters(), lr=1e-3)

writer = SummaryWriter("./log")


def train(model, epochs, train_loader, validation_loader, losses, metrics, optimizer):
    # Loss: per output
    # Metric: per output
    # Optimizer: single/multiple

    # Checkpoint => model + epoch + [batch]? + losses + metrics + optimizer
    pass


def test(model, test_loader, losses, metrics):
    pass


root = os.getcwd()
checkpoints = glob.glob(f"{root}/backup/experiment_name/*.pt")
checkpoint_epoch = 0
checkpoint_batch = 0
if len(checkpoints) > 0:
    checkpoint = pt.load(checkpoints[-1])
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    loss_fn = checkpoint["loss"]
    metric_fn = checkpoint["metric"]
    checkpoint_epoch = checkpoint["epoch"]
    checkpoint_batch = checkpoint["batch"]

for epoch in range(checkpoint_epoch, 5):
    print(f"Epoch {epoch+1}\n-------------------------------")

    # Load checkpoint
    model.train()
    metric_fn.reset()
    loss_fn.zero_grad()
    size = len(train_loader.dataset)
    for batch, (X, y) in enumerate(train_loader):
        if batch < checkpoint_batch:
            continue
        else:
            checkpoint_batch = 0

        # Compute prediction error
        pred = model(X)

        # Compute losses
        loss = loss_fn(pred, y)
        loss.backward()

        # Compute metrics
        metric_fn.update(pred, y)

        # Optimize (Backpropagation)
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            # Save checkpoint
            pt.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "batch": batch,
                "loss": loss_fn,
                "metric": metric_fn,
                "optimizer": optimizer.state_dict(),
            }, f"{root}/backup/experiment_name/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

        if batch % 10 == 0:
            # Log losses + metrics
            loss_val = loss.item()
            metric_val = metric_fn.compute()
            current = (batch + 1) * len(X)
            print(f"loss: {loss_val:>7f}\tmetric: {metric_val:>7f}\t[{current:>5d}/{size:>5d}]")

            writer.add_scalar(f"Train Loss", loss_val, (epoch * size) + batch)
            writer.add_scalar(f"Train Metric", metric_val, (epoch * size) + batch)

    model.eval()
    metric_fn.reset()
    loss_fn.zero_grad()
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss = 0
    with pt.no_grad():
        for X, y in test_loader:
            pred = model(X)
            metric_fn.update(pred, y)
            test_loss += loss_fn(pred, y).item()

    loss_val = test_loss / num_batches
    metric_val = metric_fn.compute()

    print(f"avg loss: {loss_val:>7f}\tmetric: {metric_val:>7f}")
    writer.add_text("Test Avg Loss", f"{loss_val:>7f}")
    writer.add_text("Test Avg Metric", f"{metric_val:>7f}")

writer.close()

# print(f"Train set: {len(trainset)}")
# print(f"Test set: {len(testset)}")
