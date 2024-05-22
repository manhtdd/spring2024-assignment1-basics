import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None):
        # Identical to T5LayerNorm
        super().__init__()
        self.device = device
        self.weight = nn.Parameter(torch.ones(hidden_size, device=self.device))
        self.eps = eps
    
    def set_weights_from_dict(self, d):
        self.weight.data[:] = d["weight"]
    
    def forward(self, x):
        mean_squared = x.pow(2).mean(dim=-1, keepdim=True)
        x = x / torch.sqrt(mean_squared + self.eps)
        x = x.to(self.device)
        return self.weight * x
    