import cudabox
import pytest
import torch


@pytest.mark.parametrize("M", [1, 19, 99, 989])
@pytest.mark.parametrize("N", [1, 19, 99, 989])
@pytest.mark.parametrize("K", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
def test_simple_gemm(M, N, K):
    A = torch.rand((M, K), device="cuda")
    B = torch.rand((K, N), device="cuda")

    C_ref = torch.matmul(A, B)
    C_out = cudabox.gemm.simple_gemm(A, B)

    torch.testing.assert_close(C_out, C_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("M", [1, 19, 99, 989])
@pytest.mark.parametrize("N", [1, 19, 99, 989])
@pytest.mark.parametrize("K", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
def test_tgemm(M, N, K):
    A = torch.rand((M, K), device="cuda")
    B = torch.rand((K, N), device="cuda")

    C_ref = torch.matmul(A, B)
    C_out = cudabox.gemm.tiled_gemm(A, B)

    torch.testing.assert_close(C_out, C_ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
