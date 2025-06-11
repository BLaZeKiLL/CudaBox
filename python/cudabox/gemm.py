import torch


def sgemm(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    return torch.ops.cudabox.sgemm(mat_a, mat_b)


def tgemm(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    return torch.ops.cudabox.tgemm(mat_a, mat_b)
