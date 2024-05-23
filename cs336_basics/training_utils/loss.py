import torch
from torch import Tensor

def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Computes the cross-entropy loss between logits and targets.

    Args:
    - logits (Tensor): The logits with shape (batch_size, num_classes).
    - targets (Tensor): The target indices with shape (batch_size, ).

    Returns:
    - Tensor: The computed cross-entropy loss.
    """
    # Ensure logits are numerically stable by subtracting the max value in each logit vector
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    loss = -torch.mean(torch.gather(log_probs, 1, targets.unsqueeze(-1)))

    return loss

# Example usage
if __name__ == "__main__":
    batch_size = 1
    num_classes = 5
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    print(logits)
    targets = torch.tensor([1, 4, 0])
    loss = cross_entropy_loss(logits, targets)
    print(f"Cross-entropy loss: {loss.item()}")
