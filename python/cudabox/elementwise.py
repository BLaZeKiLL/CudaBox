import torch


def softmax(tensor: torch.tensor) -> torch.tensor:
    return torch.ops.cudabox.softmax(tensor)


def rmsnorm(tensor: torch.tensor, gamma: torch.tensor, eps=1e-5) -> torch.tensor:
    return torch.ops.cudabox.rmsnorm(tensor, gamma, eps)
