import argparse
import os

import torch
import triton
import triton.testing
from cudabox.gemm import simple_gemm as cudabox_simple_gemm
from cudabox.gemm import sm90_pipelined_tma_mma_gemm as cudabox_sm90_gemm
from cudabox.gemm import tiled_gemm as cudabox_tiled_gemm

from .utils import DEFAULT_DEVICE, run_benchmark

# M=N square sizes to sweep. For each value we emit one plot per dtype with
# K on the x-axis. Yields len(MN_VALUES) * len(dtypes) = 8 plots total in
# the default config.
MN_VALUES = [1024, 2048, 4096, 8192]
K_VALUES = [128, 1024, 2048, 4096, 8192, 16384]


# Catalogue of providers. Each entry maps a provider key to
# (display_name, plot_style). Anchored at the catalogue so order /
# styling stays consistent across plots regardless of subset chosen.
ALL_PROVIDERS = {
    "torch_matmul": ("torch.matmul", ("black", "-")),
    "simple_gemm": ("Cudabox simple_gemm", ("blue", "--")),
    "tiled_gemm": ("Cudabox tiled_gemm", ("green", "-.")),
    "sm90_pipelined_tma_mma_gemm": (
        "Cudabox sm90_pipelined_tma_mma_gemm",
        ("red", "-"),
    ),
}


def _run(M: int, N: int, K: int, dtype: torch.dtype, provider: str):
    """Single benchmark sample. Shared body for every provider."""
    # A is always (M, K) row-major. For simple/tiled/torch the right operand
    # is (K, N) row-major; for the SM90 kernel the right operand is (N, K)
    # row-major (the kernel computes A @ B^T internally).
    A = torch.rand((M, K), device=DEFAULT_DEVICE, dtype=dtype).contiguous()
    B_kn = torch.rand((K, N), device=DEFAULT_DEVICE, dtype=dtype).contiguous()
    B_nk = B_kn.t().contiguous()  # (N, K) view, then own storage

    if provider == "torch_matmul":
        fn = lambda: torch.matmul(A, B_kn)
    elif provider == "simple_gemm":
        fn = lambda: cudabox_simple_gemm(A, B_kn)
    elif provider == "tiled_gemm":
        fn = lambda: cudabox_tiled_gemm(A, B_kn)
    elif provider == "sm90_pipelined_tma_mma_gemm":
        fn = lambda: cudabox_sm90_gemm(A, B_nk)
    else:
        raise ValueError(f"unknown provider: {provider}")

    return run_benchmark(fn)


def _make_benchmark(
    dtype: torch.dtype, mn: int, providers: list[str], suffix: str = ""
) -> triton.testing.Benchmark:
    """Build a single Benchmark with K on the x-axis at fixed M=N=mn.
    `dtype`, `M`, `N` are bound via `args={}` so they don't appear on the
    plot's x-axis. Optional `suffix` appended to the plot_name to distinguish
    runs (e.g. "-Native" when naive kernels are included)."""
    line_names = [ALL_PROVIDERS[p][0] for p in providers]
    styles = [ALL_PROVIDERS[p][1] for p in providers]
    return triton.testing.Benchmark(
        x_names=["K"],
        x_vals=K_VALUES,
        line_arg="provider",
        line_vals=providers,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name=f"M{mn}-N{mn}{suffix}",
        args={"dtype": dtype, "M": mn, "N": mn},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cudabox GEMM kernels against torch.matmul "
        "across fp16 and bf16. K is swept on the x-axis; one plot per "
        "(M=N, dtype) pair."
    )
    parser.add_argument(
        "--include-naive",
        action="store_true",
        help="Include simple_gemm and tiled_gemm in the comparison. These "
        "are ~2-3 orders of magnitude slower than the SM90 / torch.matmul "
        "kernels and dominate the y-axis, so they're off by default.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "results", "gemm"),
        help="Where to write plot and CSV outputs. Per-dtype subdirs are "
        "created underneath (e.g. results/gemm/float16/, results/gemm/bfloat16/).",
    )
    args = parser.parse_args()

    providers = ["torch_matmul", "sm90_pipelined_tma_mma_gemm"]
    suffix = ""
    if args.include_naive:
        providers = [
            "torch_matmul",
            "simple_gemm",
            "tiled_gemm",
            "sm90_pipelined_tma_mma_gemm",
        ]
        # Distinguish output files from the default (sm90-only) run so they
        # don't clobber each other when both are present in the same dir.
        suffix = "-Native"

    # Build + run a single Benchmark per dtype, each writing into its own
    # subdirectory. Close the matplotlib figure between runs so memory and
    # the open-figure warning never become an issue.
    import matplotlib.pyplot as plt

    for dtype in (torch.float16, torch.bfloat16):
        dtype_str = str(dtype).removeprefix("torch.")
        dtype_out = os.path.join(args.out_dir, dtype_str)
        os.makedirs(dtype_out, exist_ok=True)

        # One Benchmark per M=N value, each producing its own plot file
        # (M{mn}-N{mn}.png/.csv). Run them sequentially and close figures
        # between runs so matplotlib never holds more than a single plot in
        # memory.
        for mn in MN_VALUES:
            benchmark = triton.testing.perf_report(
                [_make_benchmark(dtype, mn, providers, suffix)]
            )(_run)
            benchmark.run(print_data=True, save_path=dtype_out)
            plt.close("all")


if __name__ == "__main__":
    main()
