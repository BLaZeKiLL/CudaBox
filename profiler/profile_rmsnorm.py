import argparse

import torch
from cudabox.elementwise import rmsnorm

"""
-- set full/basic/roofline

# Full profile
ncu --set full \
    --nvtx --nvtx-include "rmsnorm_profile/" \
    --import-source yes \
    --target-processes all \
    -o profiler/rmsnorm_profile -f \
    python -m profiler.profile_rmsnorm

# Quick summary terminal only
ncu --set full --nvtx --nvtx-include "rmsnorm_profile/" \
    --target-processes all \
    python -m profiler.profile_rmsnorm

# Without nvtx
ncu --set full \
    --kernel-name rmsnorm_kernel \
    --launch-skip 3 --launch-count 1 \
    --target-processes all \
    --import-source yes \
    -o profiler/rmsnorm_profile -f \
    python -m profiler.profile_rmsnorm

# Override the shape (forwarded after the module name)
python -m profiler.profile_rmsnorm --rows 1024 --cols 4096
"""


def main():
    parser = argparse.ArgumentParser(description="Profile the rmsnorm kernel.")
    parser.add_argument("--rows", type=int, default=256, help="batch dim (B)")
    parser.add_argument("--cols", type=int, default=8192, help="normalized dim (N)")
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn((args.rows, args.cols), dtype=torch.float32, device="cuda")
    gamma = torch.randn((args.cols,), dtype=torch.float32, device="cuda")

    # Warm up: JIT/loaders, allocator, etc. NCU profiles the kernels AFTER this
    # point (use --target-processes + --launch-skip, or the NVTX range below).
    for _ in range(3):
        _ = rmsnorm(x, gamma, args.eps)
    torch.cuda.synchronize()

    # The launch we actually care about.
    torch.cuda.nvtx.range_push("rmsnorm_profile")
    y = rmsnorm(x, gamma, args.eps)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print(y.sum().item())  # ensure result isn't DCE'd


if __name__ == "__main__":
    main()
