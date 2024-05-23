import math

def cosine_with_warmup_lr_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
    return min_learning_rate