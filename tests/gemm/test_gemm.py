import pytest
import torch
import cudabox

@pytest.mark.parametrize("M", [1, 19, 99, 989])
@pytest.mark.parametrize("N", [1, 19, 99, 989])
@pytest.mark.parametrize("K", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
def test_sgemm(M, N, K):
    A = torch.rand((M, K), device="cuda")
    B = torch.rand((K, N), device="cuda")

    C_ref = torch.matmul(A, B)
    C_out = cudabox.gemm.sgemm(A, B)

    torch.testing.assert_close(C_out, C_ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
