import torch
from torch import optim
from typing import Callable, Optional, Tuple

class AdamW(optim.Optimizer):
    def __init__(
        self, 
        params, 
        lr: float = 1e-3, 
        betas: Tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-6, 
        weight_decay: float = 0.0
    ):
        assert lr >= 0, f"Invalid learning rate: {lr}, must be >= 0"
        assert 0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0, f"Invalid beta values: {betas}, must be in [0, 1)"
        assert eps >= 0, f"Invalid epsilon value: {eps}, must be >= 0"
        assert weight_decay >= 0, f"Invalid weight_decay value: {weight_decay}, must be >= 0"
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
        - closure (Callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
        - Optional[float]: The loss if the closure is provided, otherwise None.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad

                if 't' not in state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]

                m, v = state["m"], state["v"]
                b1, b2 = group["betas"]

                # Update biased first moment estimate
                m.mul_(b1).add_(grad, alpha=(1.0 - b1))
                # Update biased second raw moment estimate
                v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
                # Compute bias-corrected first moment estimate
                m_hat = m / (1.0 - b1 ** t)
                # Compute bias-corrected second raw moment estimate
                v_hat = v / (1.0 - b2 ** t)

                denom = v_hat.sqrt().add_(group["eps"])

                alpha = group["lr"]

                p.addcdiv_(m_hat, denom, value=-alpha)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-alpha * group["weight_decay"]))

        return loss

# Example usage
if __name__ == "__main__":
    model = torch.nn.Linear(2, 1)
    optimizer = AdamW(model.parameters(), lr=1e-3)

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
