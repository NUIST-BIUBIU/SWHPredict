import torch.nn as nn
from layers.Temporal import TCNBlock
from layers.Attention_Family import FullAttention, AttentionLayer


class TCNAttention(nn.Module):
    def __init__(self, configs):
        num_inputs = 1
        num_channels = [4,1]
        kernel_size = 2
        dropout = 0.2
        super(TCNAttention, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                padding=(kernel_size-1) * dilation_size, activation=configs.activation, dropout=dropout)]
        self.tcnNetwork = nn.Sequential(*layers)
        self.attention = AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=False), configs.seq_len, configs.n_heads)
        self.outNetwork = nn.Linear(configs.seq_len, 1)

    def forward(self, x):
        out1 = self.tcnNetwork(x)
        att_out, _ = self.attention(out1, out1, out1, attn_mask=None)
        att_out = att_out[:,0,:]
        pred = self.outNetwork(att_out)
        return pred

