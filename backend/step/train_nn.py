import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import net
from utils.dataset import PwdJobsDataset

import pandas as pd
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_num_correct(pred, y):
    pred_ints = torch.round(torch.sigmoid(pred))
    return torch.sum(pred_ints == y).item()

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # X = X.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += get_num_correct(pred, y)

    test_loss / num_batches
    correct /= (size * (30))
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n')

def run(args):
    
    dataset = PwdJobsDataset(args.X_file, args.Y_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=False)

    model = net.NeuralNetwork().to(device)
    if (Path(args.model_weights).is_file()):
        model.load_state_dict(torch.load(args.model_weights))

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    for t in range(args.epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_loop(dataloader, model, loss_fn, optimizer)
        test_loop(dataloader, model, loss_fn)

    print('Done!')
    torch.save(model.state_dict(), args.model_weights)
