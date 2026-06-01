import torch
from cudabox.gemm import sm90_pipelined_tma_mma_gemm

"""
Profile the SM90 pipelined TMA+WGMMA GEMM at a single (M, N, K) shape.

ncu invocations (run from repo root):

# Full profile, NCU report file
ncu --set full \
    --nvtx --nvtx-include "gemm_profile/" \
    --import-source yes \
    --target-processes all \
    -o profiler/gemm_profile -f \
    python -m profiler.profile_gemm

# Quick summary, terminal only
ncu --set full --nvtx --nvtx-include "gemm_profile/" \
    --target-processes all \
    python -m profiler.profile_gemm

# Without nvtx — filter by kernel name + skip warmup launches.
# The mangled name starts with `_ZN7cudabox4gemm...gemm_device`.
ncu --set full \
    --kernel-name regex:".*gemm_device.*" \
    --launch-skip 3 --launch-count 1 \
    --target-processes all \
    --import-source yes \
    -o profiler/gemm_profile -f \
    python -m profiler.profile_gemm
"""


def main():
    # Square problem at the typical "training-shape sweet-spot" size used by
    # the bench. M, N must satisfy the kernel's TMA alignment (N is mul of 8
    # for fp16; M has no alignment constraint); K must be a mul of 8.
    M, N, K = 4096, 4096, 4096
    dtype = torch.float16

    torch.manual_seed(0)
    # A is (M, K) row-major; B is (N, K) row-major (kernel computes A @ B^T).
    A = torch.rand((M, K), dtype=dtype, device="cuda").contiguous()
    B = torch.rand((N, K), dtype=dtype, device="cuda").contiguous()

    # Warm up: kernel JIT-of-PTX (if any), TMA descriptor cache, cuBLAS init
    # (Torch eager calls into cuBLAS for other ops at process start). NCU
    # should be told to --launch-skip these, or use the NVTX range below to
    # scope only the launch we care about.
    for _ in range(3):
        _ = sm90_pipelined_tma_mma_gemm(A, B)
    torch.cuda.synchronize()

    # The single launch NCU should profile.
    torch.cuda.nvtx.range_push("gemm_profile")
    C = sm90_pipelined_tma_mma_gemm(A, B)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # Sanity-check the output isn't DCE'd. A trivial accumulator suffices.
    print(f"C shape={tuple(C.shape)} dtype={C.dtype} sum={C.float().sum().item():.3e}")


if __name__ == "__main__":
    main()
