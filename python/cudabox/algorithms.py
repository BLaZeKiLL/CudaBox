import torch


def histogram(tensor: torch.tensor, num_bins: int) -> torch.tensor:
    return torch.ops.cudabox.histogram(tensor, num_bins)
