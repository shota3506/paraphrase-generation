import math

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor `(S, N, E)`
        Output:
            Tensor `(S, N, E)`

            where S is the sequence length, N is the batch size, E is the feature number
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
