import torch


def generate_batch(batch_size: int, device: torch.device = torch.device("cpu")):
    x = torch.randn(batch_size, 10, device=device)
    y = x.sum(dim=1, keepdim=True) ** 2
    y = y + torch.randn_like(y) * 10  # gaussian noise: mean=y, std=10 (variance=100)
    return x, y
