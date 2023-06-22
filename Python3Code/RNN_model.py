import sys
import copy
from util import util
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import pickle as pkl
from sklearn.model_selection import train_test_split
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning
import fire
from util.VisualizeDataset import VisualizeDataset
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME1 = 'chapter5_resultIvo.csv'
DATASET_FNAME2 = 'chapter5_resultJoost.csv'
DATASET_FNAME3 = 'chapter5_resultFlo.csv'
RESULT_FNAME = 'chapter7_classification_result.csv'
EXPORT_TREE_PATH = Path('./figures/crowdsignals_ch7_classification/')

# Next, we declare the parameters we'll use in the algorithms.
N_FORWARD_SELECTION = 50

try:
    dataset1 = pd.read_csv(DATA_PATH / DATASET_FNAME1, index_col=0)
    dataset2 = pd.read_csv(DATA_PATH / DATASET_FNAME2, index_col=0)
    dataset3 = pd.read_csv(DATA_PATH / DATASET_FNAME3, index_col=0)
    # dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x[0].size(0), self.hidden_size).to(x[0].device)
        packed_output, hn = self.rnn(x, h0)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        return output

class VariableLengthDataset(Dataset):
    def __init__(self, sequences, lengths, targets):
        self.sequences = sequences
        self.lengths = lengths
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.lengths[index], self.targets[index]

def create_dataset(df, window_size):
    prepare = PrepareDatasetForLearning()

    train_X, test_X, train_y, test_y = prepare.split_multiple_datasets_classification([dataset1, dataset2, dataset3], ['label'], 'like', 0.7, filter=True, temporal=False, unknown_users=True)

    return train_X, train_y, test_X, test_y

def variable_length_collate(batch):
    sequences, lengths, targets = zip(*batch)
    sequences = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor(lengths)
    targets = pad_sequence(targets, batch_first=True)
    packed_sequences = pack_padded_sequence(sequences, lengths, batch_first=True, enforce_sorted=False)
    return packed_sequences, targets

def main(
        fe=True, lr=0.01, num_epochs=100, batch_size=8,
        hidden_size=16, num_layers=1, output_size=1, window_size = 8):
    
    if fe:
        with open("./models/pca.pkl", "rb") as file:
            pca = pkl.load(file)
        df = pd.read_csv("./datasets/feature_engineered.csv")
    else:
        df = pd.read_csv("./datasets/properly_parsed_and_filtered.csv")
    df["time"] = pd.to_datetime(df["time"])

    sequences, targets, lengths = create_dataset(df, window_size)
    dataset = VariableLengthDataset(sequences, lengths, targets)
    train_dataset = Subset(dataset, range(0, int(len(dataset)*0.8)))

    test_dataset = Subset(dataset, range(int(len(dataset)*0.8), len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=variable_length_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=variable_length_collate)

    # Create the RNN model
    model = RNN(df.drop(["mood", "time", "id"], axis=1).shape[1], hidden_size, num_layers, output_size, batch_size)
    # Define the loss function and optimizer
    loss_fn = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_curve = []
    for epoch in range(num_epochs):
        running_loss = [] 
        for i, (batch_x, batch_y) in enumerate(train_loader, 0):
            # Forward pass
            model.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs.squeeze(), batch_y)

            # Backward pass and update parameters
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item())
            if i % 10 == 9: # print every 10 mini-batches
                print('[%d, %5d] loss: %f' % (epoch + 1, i + 1, np.mean(running_loss)), end="\r")
        loss_curve.append(np.mean(running_loss))
        running_loss = 0.0
    torch.save(model.state_dict(), "./models/trained.ckpt")

    plt.plot(loss_curve)
    plt.show()

    loss_curve = []
    model.eval()
    for i, (batch_x, batch_y) in enumerate(test_loader, 0):
        # Forward pass
        model.zero_grad()
        outputs = model(batch_x)
        loss = loss_fn(outputs.squeeze(), batch_y)

        # Backward pass and update parameters
        loss.backward()
        optimizer.step()
        
        if i % 10 == 9: # print every 10 mini-batches
            print('[%d, %5d] loss: %f' % (epoch + 1, i + 1, loss.item()), end="\r")
        loss_curve.append(loss.item())
        running_loss = 0.0
    
    plt.plot(loss_curve)
    plt.show()
if __name__ == "__main__":
    fire.Fire(main)