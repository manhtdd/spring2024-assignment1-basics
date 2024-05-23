def gradient_clipping(parameters, max_norm: float):
    acc = 0
    for p in parameters:
        if p.grad is not None:
            acc += p.grad.data.square().sum()
    total_norm = acc.sqrt()
    if total_norm > max_norm:  # TODO make this unconditional
        total_norm += 1e-6
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= max_norm / total_norm