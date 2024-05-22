import torch
from torch import einsum, nn
from math import sqrt
from .softmax import softmax

def sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None, pdrop: float = None) -> torch.Tensor:
    """
    Scaled Dot-Product Attention (SDPA) function.

    Args:
    - Q (torch.Tensor): Query tensor of shape (..., q, d)
    - K (torch.Tensor): Key tensor of shape (..., k, d)
    - V (torch.Tensor): Value tensor of shape (..., k, d)
    - mask (torch.Tensor, optional): Mask tensor of shape (..., q, k) to apply on the attention weights.
    - pdrop (float, optional): Dropout probability to apply on the attention weights.

    Returns:
    - torch.Tensor: The output tensor of shape (..., q, d)
    """
    
    # Compute dot product between Q and K
    qk_prod = einsum('...qd,...kd -> ...qk', Q, K)
    
    # Apply mask if provided
    if mask is not None:
        qk_prod = qk_prod.masked_fill(mask, -float("inf"))
    
    # Scale the dot product by the square root of the last dimension of Q
    attn_weights = softmax(qk_prod / sqrt(Q.shape[-1]), dim=-1)
    
    # Apply dropout if pdrop is provided
    if pdrop is not None:
        attn_weights = nn.functional.dropout(attn_weights, pdrop)
    
    # Compute the final output by multiplying the attention weights with the values
    return einsum('...qk,...kd -> ...qd', attn_weights, V)
