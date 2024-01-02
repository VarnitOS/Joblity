import os
import pandas as pd
import torch
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PwdJobsDataset(Dataset):
    def __init__(self, X_file, Y_file, transform=None, target_transform=None):
        self.X = pd.read_csv(X_file, header=None, skiprows=1)
        self.Y = pd.read_csv(Y_file, header=None, skiprows=1)
        self.transform=transform
        self.target_transform=target_transform

    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X.iloc[idx]).float().to(device)
        y = torch.tensor(self.Y.iloc[idx]).float().to(device)
        return x, y
