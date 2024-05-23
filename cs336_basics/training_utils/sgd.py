import torch
from torch import optim
from torch import Tensor
from typing import Callable, Optional
import math

class SGD(optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[Tensor]:
        """
        Performs a single optimization step.
        
        Args:
        - closure (Callable, optional): A closure that re-evaluates the model and returns the loss.
        
        Returns:
        - Optional[Tensor]: The loss if the closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "t" not in state:
                    state["t"] = 0

                t = state["t"]
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss

# Example usage
if __name__ == "__main__":
    model = torch.nn.Linear(2, 1)
    optimizer = SGD(model.parameters(), lr=1e-2)

    # Dummy input and target
    input = torch.randn(3, 2)
    target = torch.randn(3, 1)

    criterion = torch.nn.MSELoss()

    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        return loss

    for _ in range(100):
        loss = optimizer.step(closure)
        print(f"Loss: {loss.item()}")
