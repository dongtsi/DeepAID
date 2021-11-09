import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import more_itertools

def se2rmse(a):
    return torch.sqrt(sum(a.t())/a.shape[1])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# hyper params
feature_size = 100
hidden_len = 50
batch_size = 256
lr = 1e-3
weight_decay = 1e-5
epoches = 20
seq_len = 5

class LSTM_multivariate(nn.Module):
    def __init__(self):
        super(LSTM_multivariate, self).__init__()

        self.rnn = nn.LSTM(         
            input_size=feature_size,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, feature_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out
    
criterion = nn.MSELoss()
model = LSTM_multivariate().to(device)
optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
getMSEvec = nn.MSELoss(reduction='none')


def train(train_data):
    model.train()

    X_train = more_itertools.windowed(train_data,n=seq_len,step=1)
    X_train = np.asarray(list(X_train))
    y_train = np.asarray(train_data[seq_len-1:])
    print("SHAPE",X_train.shape,y_train.shape)
    X_train = torch.from_numpy(X_train).type(torch.float).to(device)
    y_train = torch.from_numpy(y_train).type(torch.float).to(device)

    
    torch_dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(loader):
            output = model(batch_x)
            loss = criterion(output, batch_y)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
            if step % 10 == 0 :
                print('epoch:{}/{}'.format(epoch,step), '|Loss:', loss.item())
    
    model.eval()
    output = model(X_train)
    mse_vec = getMSEvec(output,y_train)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()

    print("max AD score",max(rmse_vec))
    thres = max(rmse_vec)
    rmse_vec.sort()
    pctg = 0.99999
    thres = rmse_vec[int(len(rmse_vec)*pctg)]
    print("thres:",thres)
    return model, thres
    

@torch.no_grad()
def test(model, thres, test_data):
    model.eval()
    X_test = more_itertools.windowed(test_data,n=seq_len,step=1)
    X_test = np.asarray(list(X_test))
    y_test = np.asarray(test_data[seq_len-1:])
    X_test = torch.from_numpy(X_test).type(torch.float).to(device)
    y_test = torch.from_numpy(y_test).type(torch.float).to(device)

    output = model(X_test)
    mse_vec = getMSEvec(output,y_test)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
    rmse_vec = np.concatenate((np.asarray([0.]*(seq_len-1)),rmse_vec))
    idx_mal = np.where(rmse_vec>thres)
    idx_ben = np.where(rmse_vec<=thres)
    # print(len(rmse_vec[idx_ben]),len(rmse_vec[idx_mal]))
    return rmse_vec


@torch.no_grad()
def test_from_iter(model, thres, X_test):
    model.eval()
    y_test = X_test[:,-1,:]
    # print("X_test",X_test.shape,"y_test",y_test.shape)
    X_test = torch.from_numpy(X_test).type(torch.float).to(device)
    y_test = torch.from_numpy(y_test).type(torch.float).to(device)

    output = model(X_test)
    # print("output",output.size(),"y_test",y_test.size())
    mse_vec = getMSEvec(output,y_test)
    rmse_vec = se2rmse(mse_vec).cpu().data.numpy()
    rmse_vec = np.concatenate((np.asarray([0.]*(seq_len-1)),rmse_vec))
    idx_mal = np.where(rmse_vec>thres)
    idx_ben = np.where(rmse_vec<=thres)
    # print(len(rmse_vec[idx_ben]),len(rmse_vec[idx_mal]))
    return rmse_vec


