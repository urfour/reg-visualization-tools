import os
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import manual_seed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

manual_seed(0)
random.seed(0)
np.random.seed(0)

import utils.losses.pytorch as losses
plt.ioff()

def concat_results(results):
    """ Concatenate results from multiple runs of the same model"""
    res = defaultdict(list)
    for d in results:
        for k, v in d.items():
            res[k].append(v)
    print(*res.keys())
    for key in res.keys():
        res[key] = [n for n in res[key] if n != 0.]
        print(f'{np.mean(res[key]):.4f} ± {np.std(res[key]):.4f}')

class LSTMModel(nn.Module):
    """ LSTM model for CMAPSS dataset

    Parameters:
        input_size (int): Number of features in the input
        hidden_size (int): Number of hidden units in the LSTM layer
        num_layers (int): Number of LSTM layers
    """
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 50)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

        out, _ = self.lstm(x, (h0, c0))
        out = self.linear1(out[:, -1, :])
        out = self.linear2(out)
        out = self.relu(out)
        return out
    
class VanillaLSTMModel(nn.Module):
    """ Vanilla LSTM model for CMAPSS dataset

    Parameters:
        input_size (int): Number of features in the input
        hidden_size (int): Number of hidden units in the LSTM layer
        num_layers (int): Number of LSTM layers
    """
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(VanillaLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        out = self.relu(out)
        return out
    
class CNNModel(nn.Module):
    """ CNN model for CMAPSS dataset

    Parameters:
        input_size (int): Number of features in the input
    """
    def __init__(self, input_size):
        super(CNNModel, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(input_size, 60, 2)
        self.conv2 = nn.Conv1d(60, 120, 2)
        self.dropout = nn.Dropout(0.2)
        self.maxpool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1680, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x

class CMAPSSTraining():
    """ Class for training models on CMAPSS dataset

    Parameters:
        dataset (str): Dataset to use (FD001, FD002, FD003, FD004)
        model_type (str): Type of model to use (cnn, lstm, vanilla)
        window_size (int): Size of the window for the LSTM model
        criterion (object): Loss function to use
    """
    def __init__(self, dataset : str = 'FD001', model_type='cnn', window_size : int = 15, criterion : object = nn.MSELoss()):
        self.dataset = dataset
        self.window_size = window_size
        self.model_type = model_type
        self.criterion = criterion
        self.import_data()

    def series_to_supervised(self, set : str, n_in=1, dropnan=True):
        """ Convert series to supervised learning problem

        Parameters:
            set (str): Dataset to use (train, test)
            n_in (int): Number of lag observations as input
            dropnan (bool): Drop rows with NaN values
        """
        if set == 'train':
            data = self.df_train.copy()
        else:
            data = self.df_test.copy()
        all_df = []
        for engine in data['engine'].unique():
            df = data[data['engine'] == engine]
            cols = list()
            for i in range(n_in, 0, -1):
                shifted_col = df.shift(i)
                shifted_col.columns = [f'{name} (t-{str(i)})' for name in shifted_col.columns]
                cols.append(shifted_col)
            cols.append(df)
            agg = pd.concat(cols, axis=1)
            if dropnan:
                agg.dropna(inplace=True)
            all_df.append(agg)
        return pd.concat(all_df)
        
    def create_lstm_data(self, data, k):
        """ Create LSTM data from time series data

        Parameters:
            data (numpy array): Time series data
            k (int): Size of the window
        """
        X_data = np.zeros([data.shape[0]-k, k, data.shape[1]-1])
        y_data = []

        for i in range(k, data.shape[0]):
            cur_sequence = data[i-k: i, :-1]
            cur_target = data[i-1, -1]

            X_data[i-k,:, :] = cur_sequence.reshape(1, k, X_data.shape[2])
            y_data.append(cur_target)
        
        return X_data, np.asarray(y_data)
    
    def create_dataset(self, data, cuda = True):
        """ Create dataset for training

        Parameters:
            data (pandas DataFrame): Data to use
            cuda (bool): Use GPU for training
        """
        X, y = self.create_lstm_data(data.to_numpy(), self.window_size)
        if cuda:
            return torch.FloatTensor(X).cuda(), torch.FloatTensor(y).cuda()
        else:
            return torch.FloatTensor(X), torch.FloatTensor(y)

    def import_data(self):
        """ Import and preprocess data """
        df = pd.read_csv('data/train_'+self.dataset+'.txt', sep=' ', header=None)
        df.drop([26, 27], axis=1, inplace=True)

        settings = ['sensor'+str(i) for i in range(1, 22)]
        df.columns = ['engine', 'cycle', 'op1', 'op2', 'op3', *settings]    
        self.features = [
            'sensor2', 'sensor3', 'sensor4', 'sensor6', 'sensor7', 'sensor8', 'sensor9', 'sensor11',
            'sensor12', 'sensor13', 'sensor14', 'sensor15', 'sensor17', 'sensor20', 'sensor21'
        ]

        for engine in df['engine'].unique():
            df.loc[df['engine'] == engine, 'RUL'] = max(
                df[df['engine'] == engine]['cycle']) - df[df['engine'] == engine]['cycle']

        train_indices, test_indices = train_test_split(df['engine'].unique(), test_size=0.2, random_state=42)
        self.df_train = df[df['engine'].isin(train_indices)].copy()
        self.df_test = df[df['engine'].isin(test_indices)].copy()

        scaler = MinMaxScaler()
        self.df_train[self.features] = scaler.fit_transform(self.df_train[self.features])
        self.df_test[self.features] = scaler.fit_transform(self.df_test[self.features])

        self.df_train = self.series_to_supervised('train')
        self.df_test = self.series_to_supervised('test')

        self.features_windows = [col for col in self.df_train.columns 
                                 if any(feature in col for feature in self.features)]

        X_train = self.df_train.loc[self.df_train['RUL'] < 125, self.features_windows]
        X_train['RUL'] = self.df_train.loc[self.df_train['RUL'] < 125, 'RUL']
        X_test = self.df_test.loc[self.df_test['RUL'] < 125, self.features_windows]
        X_test['RUL'] = self.df_test.loc[self.df_test['RUL'] < 125, 'RUL']

        self.X_train, self.y_train = self.create_dataset(X_train)
        self.X_test, self.y_test = self.create_dataset(X_test)

    def train(self, epochs=40, hidden_size=32, save_model=False, show_grad=False, verbose=True, lr=1e-3):
        """ Train the model

        Parameters:
            epochs (int): Number of epochs to train
            hidden_size (int): Number of hidden units in the LSTM layer
            save_model (bool): Save the model after training
            show_grad (bool): Show gradients during training
            verbose (bool): Show progress bar during training
            lr (float): Learning rate
        """
        self.last_loss = 0
        current_epoch = 0
        training_failed = True
        if verbose:
            print('Training model for %d epochs' % (epochs))
        while training_failed:
            all_losses = []
            if self.model_type == 'cnn':
                self.model = CNNModel(self.X_train.shape[1]).cuda()
            elif self.model_type == 'vanilla':
                self.model = VanillaLSTMModel(self.X_train.shape[1], hidden_size).cuda()
            else:
                self.model = LSTMModel(self.X_train.shape[2], hidden_size).cuda()
            optimizer = optim.Adam(self.model.parameters(), lr)
            loader = data.DataLoader(data.TensorDataset(self.X_train, self.y_train), shuffle=False, batch_size=32)
            while current_epoch < epochs:
                self.model.train()
                if verbose:
                    with tqdm(loader, unit='batch') as tepoch:
                        for X_batch, y_batch in tepoch:
                            tepoch.set_description(f"Epoch {current_epoch+1}")
                            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                            y_pred = self.model(X_batch).squeeze(1)
                            optimizer.zero_grad()
                            loss = self.criterion(y_pred, y_batch)
                            loss.backward()
                            if show_grad:
                                print(self.model.linear.weight.grad)
                            optimizer.step()
                            tepoch.set_postfix(loss=loss.item())
                else:
                    for X_batch, y_batch in loader:
                        X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                        y_pred = self.model(X_batch).squeeze(1)
                        optimizer.zero_grad()
                        loss = self.criterion(y_pred, y_batch)
                        loss.backward()
                        if show_grad:
                            print(self.model.linear.weight.grad)
                        optimizer.step()
                self.model.eval()
                with torch.no_grad():
                    y_pred = self.model(self.X_train).squeeze(1)
                    train_loss = self.criterion(y_pred, self.y_train)
                    if train_loss == 0 or self.last_loss == train_loss:
                        training_failed = True
                        break
                    else:
                        all_losses.append(train_loss)
                        train_rmse = torch.sqrt(nn.MSELoss()(y_pred, self.y_train))
                        if verbose:
                            tepoch.set_postfix(loss=train_loss, RMSE=train_rmse)
                        training_failed = False
                        self.last_loss = train_loss
                        current_epoch += 1
        if save_model:
            self.save_model()

    def load_model(self, path):
        """ Load a model from a file

        Parameters:
            path (str): Path to the model file
        """
        if self.model_type == 'cnn':
            self.model = CNNModel(self.X_train.shape[1])
        else:
            self.model = LSTMModel(self.X_train.shape[2])
        try:
            self.model.load_state_dict(torch.load(path))
        except:
            raise

    def save_model(self, name):
        """ Save the model to a file

        Parameters:
            name (str): Name of the model file
        """
        os.makedirs(f'models/{self.dataset}', exist_ok='True')
        torch.save(self.model.state_dict(), os.path.join('models', self.dataset, name))

    def calculate_all_losses(self) -> dict:
        """ Calculate all losses for the model """
        all_losses = {}
        y_pred = self.model(self.X_test).squeeze(1)
        all_losses['rmse'] = losses.rmse(self.y_test, y_pred).detach().cpu().numpy()
        all_losses['mae'] = losses.mae(self.y_test, y_pred).detach().cpu().numpy()
        all_losses['lin_lin_01_1'] = losses.lin_lin(0.1, 1.0)(self.y_test, y_pred).detach().cpu().numpy()
        all_losses['quad_quad_02'] = losses.quad_quad(0.2)(self.y_test, y_pred).detach().cpu().numpy()
        all_losses['quad_quad_08'] = losses.quad_quad(0.8)(self.y_test, y_pred).detach().cpu().numpy()
        all_losses['quad_quad_09'] = losses.quad_quad(0.9)(self.y_test, y_pred).detach().cpu().numpy()
        all_losses['rul_score'] = losses.rul_score(self.y_test, y_pred).detach().cpu().numpy()
        return all_losses

    def calc_errors(self, loss_name):
        """ Calculate errors for the model

        Parameters:
            loss_name (str): Name of the loss function
        """
        indices = self.df_test.groupby('engine')['cycle'].count().sort_values(ascending=True).index.to_numpy()
        all_y_pred = []
        for i, engine in enumerate(indices):
            to_predict = self.df_test.loc[(self.df_test['RUL'] < 125) & (self.df_test['engine'] == engine)][self.features_windows]
            to_predict['RUL'] = self.df_test.loc[(self.df_test['RUL'] < 125) & (self.df_test['engine'] == engine)]['RUL']
            to_predict, y = self.create_dataset(to_predict)
            y_pred = self.model(to_predict.cuda()).squeeze(1).cpu().detach().numpy()
            for j in range(len(y_pred)):
                all_y_pred.append((y[j].item(), y_pred[j].item(), (y_pred[j] - y[j]).item()))
        df = pd.DataFrame(all_y_pred, columns=['RUL', 'RUL_'+loss_name, 'error_'+loss_name])
        return df