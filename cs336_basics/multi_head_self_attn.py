import torch
from torch import nn
from einops import einsum, rearrange
from .sdpa import sdpa
from math import sqrt


class MHSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float | None = None, device=None):
        super().__init__()
        self.attn_pdrop = attn_pdrop
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        d_k = d_model // num_heads
        d_v = d_k  # Not necessarily the case
        self.d_k = d_k
        self.d_v = d_v
        self.W_qkv = nn.Parameter(torch.empty(
            3, num_heads, d_k, d_model, device=device))
        self.W_o = nn.Linear(num_heads * d_v, d_model,
                             bias=False, device=device)
        self.reset_parameters()

    def reset_parameters(self):
        for qkv in range(self.W_qkv.shape[0]):
            for head in range(self.W_qkv.shape[1]):
                nn.init.kaiming_uniform_(self.W_qkv[qkv, head], a=sqrt(5))

    def set_weights_from_dict(self, d):
        if 'q_heads.0.weight' in d:
            for qkvi, qkvn in enumerate('qkv'):
                for head in range(self.num_heads):
                    self.W_qkv.data[qkvi, head] = d[f"{
                        qkvn}_heads.{head}.weight"]
        else:
            for qkvi, qkvn in enumerate('qkv'):
                weight = d[f"{qkvn}_proj.weight"]
                weight = rearrange(
                    weight, "(heads d) dm -> heads d dm", heads=self.num_heads)
                self.W_qkv.data[qkvi, ...] = weight

        self.W_o.weight.data[:] = d["output_proj.weight"]

    def forward(self, x):
        seq_len = x.shape[-2]

        qkv_heads = einsum(
            x, self.W_qkv, "... s m, qkv h d m -> ... qkv h s d")
        Q, K, V = qkv_heads[..., 0, :, :, :], qkv_heads[..., 1, :, :, :], qkv_heads[..., 2, :, :, :]
        mask = torch.triu(torch.ones(seq_len, seq_len,
                          dtype=torch.bool, device=x.device), diagonal=1)
        attn_output = sdpa(Q, K, V, mask=mask, pdrop=self.attn_pdrop)
        # attn_output: (..., heads, seq_len, dv)
        concatenated = rearrange(
            attn_output, "... head token d -> ... token (head d)")

        out = einsum(concatenated, self.W_o.weight,
                     "... token head, d head -> ... token d")

        return out
