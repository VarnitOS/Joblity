import random
import numpy as np
import pandas as pd
import torch
from utils import net
from utils.pwd_utils import get_job_vector, print_difficulties, print_negated_jobs

def run(args):

    # Set up NN model
    model = net.NeuralNetwork()
    model.load_state_dict(torch.load(args.model_weights))
    model.eval()

    # Read in data
    X_file = pd.read_csv(args.X_file, header=None, skiprows=1)

    for i in random.sample(range(200,500), 3):
        print(f'\n-- Predicting for line {i} --')

        # Get data for example
        x = torch.tensor(X_file.iloc[i]).float()

        # Get model prediction --> sigmoid to convert to "probabilities" --> round to nearest integer, 0 or 1
        y_pred = model(x)
        y_ints = torch.round(torch.sigmoid(y_pred))

        # Validate to see it makes sense
        print('x:', x)
        print_difficulties(x)
        print('y:', y_ints)
        print_negated_jobs(y_ints)
