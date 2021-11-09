"""
This implementation of DeepLog is based on the open-source code at 
https://github.com/wuyifan18/DeepLog 
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters 
num_classes = 28 # Fixed for this demo
window_size = 10 # Fixed for this demo 
num_layers = 2 
hidden_size = 64 
num_epochs = 300
batch_size = 2048
num_candidates = 9 

class LSTM_onehot(nn.Module):
    def __init__(self, hidden_size, num_layers, num_keys):
        super(LSTM_onehot, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_keys, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_deeplog(input_seq, output_label):
    seq_dataset = TensorDataset(torch.tensor(input_seq, dtype=torch.long), torch.tensor(output_label))
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    model = LSTM_onehot( hidden_size, num_layers, num_classes).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    start_time = time.time()
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            seq = seq.clone().detach().view(-1, window_size).to(device)
            seq = F.one_hot(seq,num_classes=num_classes).float()
            output = model(seq)
            
            loss = criterion(output, label.to(device))
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
        
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))

    return model


def test_deeplog(model, test_normal_loader, test_abnormal_loader):
    model.eval()
    TP = 0
    FP = 0
    # Test the model
    start_time = time.time()
    with torch.no_grad():
        for line in test_normal_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.long).view(-1, window_size).to(device)
                seq = F.one_hot(seq,num_classes=num_classes).float()
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    FP += 1
                    break

    with torch.no_grad():
        for line in test_abnormal_loader:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                if label == -1:
                    TP += 1
                    break
                seq = torch.tensor(seq, dtype=torch.long).view(-1, window_size).to(device)
                seq = F.one_hot(seq,num_classes=num_classes).float()
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    break

    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
