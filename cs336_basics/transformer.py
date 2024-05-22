import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from typing import Literal
from .transformer_utils.transformer_block import TransformerBlock
from .transformer_utils.rmsnorm import RMSNorm


class Transformer(nn.Module):
    def __init__(self,
                 *,
                 vocab_size: int,
                 context_length: int, 
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 activation: Literal['gelu'] = 'gelu',
                 tie_embeddings: bool = False,
                 attn_pdrop: float | None = None,
                 residual_pdrop: float | None = None,
                 parallel_layers: bool = False,
                 post_norm: bool = False,
                 device=None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, device=device)
        if tie_embeddings:
            nn.init.kaiming_uniform_(self.token_embedding.weight, a=sqrt(5))
        self.position_embedding = nn.Parameter(torch.zeros(context_length, d_model, device=device))
        self.blocks = nn.Sequential(*[TransformerBlock(d_model=d_model,
                                                       num_heads=num_heads,
                                                       activation=activation,
                                                       d_ff=d_ff,
                                                       attn_pdrop=attn_pdrop,
                                                       residual_pdrop=residual_pdrop,
                                                       parallel_layers=parallel_layers,
                                                       post_norm=post_norm,
                                                       device=device) for _ in range(num_layers)])
        self.tie_embeddings = tie_embeddings
        self.dropout = nn.Dropout(residual_pdrop or 0.0)
        self.ln_final = RMSNorm(d_model, device=device)
        if not self.tie_embeddings:
            self.lm_head = nn.Linear(
                d_model, vocab_size, bias=False, device=device)
        print(f"{self=}")

    def set_weights_from_dict(self, d):
        def dict_subset(d, module):
            out_d = {}
            for k, v in d.items():
                if k.startswith(f'{module}.'):
                    out_d[k[len(module)+1:]] = v
            return out_d

        self.token_embedding.weight.data[:] = d["token_embeddings.weight"]
        self.position_embedding.data[:] = d["position_embeddings.weight"]
        for i, block in enumerate(self.blocks.children()):
            block.set_weights_from_dict(dict_subset(d, f"layers.{i}"))
        assert f'layers.{i + 1}' not in d, "Extra weights in state dict"
        self.ln_final.set_weights_from_dict(dict_subset(d, "ln_final"))
        self.lm_head.weight.data[:] = d["lm_head.weight"]

    def forward(self, x):
        if hasattr(self, 'position_embedding'):
            x = self.dropout(self.token_embedding(x) + self.position_embedding[None, :x.shape[-1], :])
        else:
            x = self.dropout(self.token_embedding(x))
        x = self.blocks(x)
        x = self.ln_final(x)
        if self.tie_embeddings:
            x = F.linear(x, self.token_embedding.weight)
        else:
            x = self.lm_head(x)
        # x = softmax(x, dim=-1)
        return x
