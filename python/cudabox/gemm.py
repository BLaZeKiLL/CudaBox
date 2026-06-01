import torch


def simple_gemm(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    return torch.ops.cudabox.simple_gemm(mat_a, mat_b)


def tiled_gemm(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    return torch.ops.cudabox.tiled_gemm(mat_a, mat_b)


def sm90_pipelined_tma_mma_gemm(
    mat_a: torch.Tensor, mat_b: torch.Tensor
) -> torch.Tensor:
    return torch.ops.cudabox.sm90_pipelined_tma_mma_gemm(mat_a, mat_b)
