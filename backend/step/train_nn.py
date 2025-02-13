import os
import sys
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import PwdJobsDataset

import pandas as pd
from pathlib import Path
from models.net import NeuralNetwork

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
    print("Training neural network...")
    
    # Load training data - skip header row (skiprows=1)
    X = torch.tensor(np.loadtxt(args.X_file, delimiter=',', skiprows=1), dtype=torch.float32)
    Y = torch.tensor(np.loadtxt(args.Y_file, delimiter=',', skiprows=1), dtype=torch.float32)
    
    # Split into train/test sets
    indices = torch.randperm(len(X))
    train_size = int(0.8 * len(X))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]
    
    # Initialize model
    model = NeuralNetwork()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, Y_test)
            predictions = (torch.sigmoid(test_outputs) > 0.5).float()
            accuracy = (predictions == Y_test).float().mean()
            
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}]')
            print(f'Train Loss: {loss.item():.4f}')
            print(f'Test Loss: {test_loss.item():.4f}')
            print(f'Test Accuracy: {accuracy.item()*100:.2f}%\n')
    
    # Save model weights
    torch.save(model.state_dict(), args.model_weights)
    print("Training completed")
