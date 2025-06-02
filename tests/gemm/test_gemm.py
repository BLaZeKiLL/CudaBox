import pytest
import torch
import cudabox


def test_sgemm():
    A = torch.zeros((16, 32), device="cuda")
    B = torch.zeros((32, 16), device="cuda")

    C_ref = torch.matmul(A, B)
    C_out = cudabox.gemm.sgemm(A, B)

    torch.testing.assert_close(C_out, C_ref)


if __name__ == "__main__":
    pytest.main([__file__])
