import argparse

import torch
from cudabox.elementwise import online_softmax

"""
-- set full/basic/roofline

# Full profile
ncu --set full \
    --nvtx --nvtx-include "online_softmax_profile/" \
    --import-source yes \
    --target-processes all \
    -o profiler/online_softmax_profile -f \
    python -m profiler.profile_online_softmax

# Quick summary terminal only
ncu --set full --nvtx --nvtx-include "online_softmax_profile/" \
    --target-processes all \
    python -m profiler.profile_online_softmax

# Without nvtx
ncu --set full \
    --kernel-name online_softmax_kernel \
    --launch-skip 3 --launch-count 1 \
    --target-processes all \
    --import-source yes \
    -o profiler/online_softmax_profile -f \
    python -m profiler.profile_online_softmax

# Override the shape (forwarded after the module name)
python -m profiler.profile_online_softmax --rows 1024 --cols 4096
"""


def main():
    parser = argparse.ArgumentParser(description="Profile the online_softmax kernel.")
    parser.add_argument("--rows", type=int, default=256, help="batch dim (B)")
    parser.add_argument("--cols", type=int, default=8192, help="softmax dim (N)")
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn((args.rows, args.cols), dtype=torch.float32, device="cuda")

    # Warm up: JIT/loaders, allocator, etc. NCU profiles the kernels AFTER this
    # point (use --target-processes + --launch-skip, or the NVTX range below).
    for _ in range(3):
        _ = online_softmax(x)
    torch.cuda.synchronize()

    # The launch we actually care about.
    torch.cuda.nvtx.range_push("online_softmax_profile")
    y = online_softmax(x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print(y.sum().item())  # ensure result isn't DCE'd


if __name__ == "__main__":
    main()
