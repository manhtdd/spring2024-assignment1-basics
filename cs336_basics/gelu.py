import torch

def gelu(x):
    device = x.device
    dtype = x.dtype
    numbers = [
        torch.tensor(0.5, dtype=dtype, device=device),
        torch.tensor(1.0, dtype=dtype, device=device),
        torch.tensor(2.0, dtype=dtype, device=device)
    ]
    return x * numbers[0] * (numbers[1] + torch.erf(x / torch.sqrt(numbers[2])))