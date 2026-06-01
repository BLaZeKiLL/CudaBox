import math

import cudabox
import pytest
import torch
from utils import dtype_str_id


@pytest.mark.parametrize("M", [1, 19, 99, 989])
@pytest.mark.parametrize("N", [1, 19, 99, 989])
@pytest.mark.parametrize("K", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=dtype_str_id)
def test_simple_gemm(M, N, K, dtype):
    A = torch.rand((M, K), device="cuda", dtype=dtype)
    B = torch.rand((K, N), device="cuda", dtype=dtype)

    C_ref = torch.matmul(A, B)
    C_out = cudabox.gemm.simple_gemm(A, B)

    torch.testing.assert_close(C_out, C_ref)


@pytest.mark.parametrize("M", [1, 19, 99, 989])
@pytest.mark.parametrize("N", [1, 19, 99, 989])
@pytest.mark.parametrize("K", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=dtype_str_id)
def test_tgemm(M, N, K, dtype):
    A = torch.rand((M, K), device="cuda", dtype=dtype)
    B = torch.rand((K, N), device="cuda", dtype=dtype)

    C_ref = torch.matmul(A, B)
    C_out = cudabox.gemm.tiled_gemm(A, B)

    torch.testing.assert_close(C_out, C_ref)


@pytest.mark.parametrize("M", [1, 8, 128, 1024, 2048, 4096])
@pytest.mark.parametrize("N", [8, 16, 128, 1024, 2048, 4096])
@pytest.mark.parametrize("K", [8, 16, 128, 1024, 2048, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=dtype_str_id)
def test_sm90_pipelined_tma_mma_gemm(M, N, K, dtype):
    A = torch.rand((M, K), device="cuda", dtype=dtype).contiguous()
    B = torch.rand((N, K), device="cuda", dtype=dtype).contiguous()

    C_ref = torch.matmul(A, B.T)
    C_out = cudabox.gemm.sm90_pipelined_tma_mma_gemm(A, B)

    # Different precision floors per dtype.
    eps = {
        torch.float16: 2**-10,  # 9.77e-4
        torch.bfloat16: 2**-7,  # 7.81e-3
    }[dtype]
    rtol = max(1e-3, 3 * math.sqrt(K) * eps)
    atol = max(1e-5, 3 * math.sqrt(K) * eps * 0.25 * K)
    torch.testing.assert_close(C_out, C_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__])
