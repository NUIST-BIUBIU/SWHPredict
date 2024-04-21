from model.tcn import TCN
from model.lstm import LSTM
from model.tcn_attention import TCNAttention
from data_provider.data_loader import data_provider
from torch import optim
import torch
import torch.nn as nn
import time
import numpy as np
import random
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,r2_score


class TCNForecasting:
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TCN': TCN,
            'LSTM': LSTM,
            'TCNAttention': TCNAttention
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        return device

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        torch.manual_seed(3407)
        np.random.seed(3407)
        random.seed(3407)
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for batch_x, batch_y in train_loader:
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            rmse,mae,r2 = self.test(epoch, test_loader)
            print("Epoch: {}, cost time: {:.2f}s, Train Loss: {:.3f}, Test rmse: {:.3f}, Test mae: {:.3f}, Test r2: {:.3f}".format(
                epoch + 1, time.time() - epoch_time, train_loss, rmse, mae, r2))

        return self.model

    def test(self, epoch, test_loader):
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                rmse = root_mean_squared_error(pred, true)
                mae = mean_absolute_error(pred,true)
                r2 = r2_score(pred,true)

        self.model.train()
        return rmse,mae,r2
