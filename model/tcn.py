import torch.nn as nn
from layers.Temporal import TCNBlock


class TCN(nn.Module):
    def __init__(self, configs):
        num_inputs = 1
        num_channels = [8,1]
        kernel_size = 2
        dropout = 0.2
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                padding=(kernel_size-1) * dilation_size, activation=configs.activation, dropout=dropout)]
        self.out = nn.Linear(configs.seq_len, 1)
        self.network = nn.Sequential(*layers, self.out)

    def forward(self, x):
        pred = self.network(x)
        pred = pred[:,:,0]
        return pred

