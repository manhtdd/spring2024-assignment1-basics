from torch import nn
from typing import Literal
from .gelu import gelu

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0, activation: Literal['gelu']='gelu', device=None):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False, device=device)
        if activation == 'gelu':
            self.activation = gelu
        # elif activation == 'silu':
        #     self.activation = nn.SiLU()
        else:
            raise ValueError
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False, device=device)
    
    def set_weights_from_dict(self, d):
        self.linear1.weight.data = d["w1.weight"]
        self.linear2.weight.data = d["w2.weight"]
    
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))