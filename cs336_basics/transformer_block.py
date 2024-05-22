from torch import nn
from typing import Literal
from .multi_head_self_attn import MHSelfAttention
from .rmsnorm import RMSNorm
from .positionwise_feedforward import PositionwiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 attn_pdrop: float | None = None,
                 activation: Literal['gelu'] = 'gelu',
                 residual_pdrop: float | None = None,
                 parallel_layers: bool = False,
                 post_norm: bool = False,
                 device=None):
        super().__init__()
        self.attn = MHSelfAttention(d_model=d_model, num_heads=num_heads, attn_pdrop=attn_pdrop, device=device)
        self.ln1 = RMSNorm(d_model, device=device)
        self.ffn = PositionwiseFeedForward(
            d_model, d_ff, activation=activation, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.dropout = nn.Dropout(residual_pdrop or 0.0)
        self.parallel_layers = parallel_layers
        self.post_norm = post_norm

    def set_weights_from_dict(self, d):
        def dict_subset(d, module):
            out_d = {}
            for k, v in d.items():
                if k.startswith(f'{module}.'):
                    out_d[k[len(module)+1:]] = v
            return out_d

        self.attn.set_weights_from_dict(dict_subset(d, "attn"))
        self.ln1.set_weights_from_dict(dict_subset(d, "ln1"))
        self.ln2.set_weights_from_dict(dict_subset(d, "ln2"))
        self.ffn.set_weights_from_dict(dict_subset(d, "ffn"))

    def forward(self, x):
        if self.parallel_layers:
            x = x + self.dropout(self.attn(self.ln1(x))) + self.dropout(self.ffn(self.ln2(x)))
        elif self.post_norm:
            x = self.ln1(x + self.dropout(self.attn(x)))
            x = self.ln2(x + self.dropout(self.ffn(x)))
        else:
            x = x + self.dropout(self.attn(self.ln1(x)))
            x = x + self.dropout(self.ffn(self.ln2(x)))
        return x
