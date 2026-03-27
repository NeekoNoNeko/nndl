# -*- coding: utf-8 -*-
"""
What is torch.nn really? - MNIST Tutorial
==========================================
This script demonstrates how to build and train a neural network in PyTorch,
starting from basic tensor operations and gradually introducing
torch.nn, torch.optim, Dataset, and DataLoader.

Final implementation: CNN with DataLoader, validation, and GPU support.
"""

import math
import gzip
import pickle
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Utility classes and functions (from the tutorial)
# ----------------------------------------------------------------------
class Lambda(nn.Module):
    """Create a layer from a custom function."""
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class WrappedDataLoader:
    """Apply a preprocessing function to each batch from a DataLoader."""
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield self.func(*b)


def loss_batch(model, loss_func, xb, yb, opt=None):
    """Compute loss for a batch; if optimizer is given, perform backprop."""
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    """Training loop with validation loss reporting."""
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)


def get_data(train_ds, valid_ds, bs):
    """Create DataLoaders for training and validation."""
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


# ----------------------------------------------------------------------
# Main execution block (required for Windows multiprocessing)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Download and load MNIST data
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"
    PATH.mkdir(parents=True, exist_ok=True)

    URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
    FILENAME = "mnist.pkl.gz"
    if not (PATH / FILENAME).exists():
        print("Downloading MNIST dataset...")
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    # Convert to torch tensors
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    n, c = x_train.shape
    print(f"Training data shape: {x_train.shape}")

    # 2. Create datasets and data loaders
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    bs = 64  # batch size
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

    # 3. Define preprocessing: reshape to (batch, 1, 28, 28) and move to device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    def preprocess(x, y):
        return x.view(-1, 1, 28, 28).to(device), y.to(device)

    # Wrap data loaders with preprocessing
    train_dl = WrappedDataLoader(train_dl, preprocess)
    valid_dl = WrappedDataLoader(valid_dl, preprocess)

    # 4. Define CNN model using nn.Sequential
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1), 
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        Lambda(lambda x: x.view(x.size(0), -1)),
    )
    model.to(device)

    # 5. Define loss function and optimizer
    loss_func = F.cross_entropy
    lr = 0.1
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 6. Train the model
    epochs = 2
    print("Starting training...")
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)

    # Optional: show one sample prediction
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(valid_dl))
        preds = model(xb)
        print("Sample predictions (first 5):", torch.argmax(preds[:5], dim=1))
        print("Actual labels:           ", yb[:5])