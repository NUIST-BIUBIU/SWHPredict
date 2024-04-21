from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        input_size = 1
        hidden_size = 8
        num_layers = 1
        output_size = 1
        num_directions = 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = num_directions
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                            bidirectional=(self.num_directions == 2))
        self.linear = nn.Linear(self.hidden_size * self.num_directions, self.output_size)

    def forward(self, x):
        h_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        output, _ = self.lstm(x, (h_0, c_0))
        pred = self.linear(output[:, -1, :])
        return pred
