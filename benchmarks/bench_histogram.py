import os

import torch
import triton
import triton.testing
from cudabox.algorithms import histogram as cudabox_histogram
from utils import DEFAULT_DEVICE, DEFAULT_DTYPE, run_benchmark


def torch_histogram(x: torch.Tensor, num_bins: int) -> torch.Tensor:
    step = x.max() / num_bins
    idx = (x / step).to(torch.int64).clamp_(0, num_bins - 1)
    out = torch.zeros(num_bins, dtype=torch.int32, device=x.device)
    out.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
    return out


# Data volume swept on the x-axis. Mirrors the size sweep used by bench_softmax.
SIZES = sorted(set([4096, 8192] + [1 << i for i in range(13, 21)]))

# One plot is emitted per bin count. Bin count drives the shared-memory
# footprint and the per-thread binning cost, so it's the natural second axis.
NUM_BINS_VALUES = [16, 64, 256]


# Catalogue of providers: provider key -> (display_name, plot_style).
ALL_PROVIDERS = {
    "torch_histogram": ("Torch histogram", ("black", "-")),
    "cudabox_histogram": ("Cudabox histogram", ("blue", "--")),
}


def _run(size: int, num_bins: int, provider: str):
    """Single benchmark sample. Shared body for every provider."""
    # rand() is non-negative and float32, satisfying the histogram kernel's
    # contract (non-negative, contiguous, 1D float32).
    x = torch.rand(size, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)

    if provider == "cudabox_histogram":
        fn = lambda: cudabox_histogram(x, num_bins)
    elif provider == "torch_histogram":
        fn = lambda: torch_histogram(x, num_bins)
    else:
        raise ValueError(f"unknown provider: {provider}")

    return run_benchmark(fn)


def _make_benchmark(num_bins: int) -> triton.testing.Benchmark:
    """Build a single Benchmark with size on the x-axis at a fixed num_bins.
    `num_bins` is bound via `args={}` so it doesn't appear on the x-axis."""
    providers = list(ALL_PROVIDERS.keys())
    return triton.testing.Benchmark(
        x_names=["size"],
        x_vals=SIZES,
        line_arg="provider",
        line_vals=providers,
        line_names=[ALL_PROVIDERS[p][0] for p in providers],
        styles=[ALL_PROVIDERS[p][1] for p in providers],
        ylabel="us",
        plot_name=f"histogram-bins{num_bins}",
        args={"num_bins": num_bins},
    )


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "results", "histogram")
    os.makedirs(out_dir, exist_ok=True)

    # One Benchmark per bin count, each producing its own plot file
    # (histogram-bins{N}.png/.csv). Close figures between runs so matplotlib
    # never holds more than a single plot in memory.
    import matplotlib.pyplot as plt

    for num_bins in NUM_BINS_VALUES:
        benchmark = triton.testing.perf_report([_make_benchmark(num_bins)])(_run)
        benchmark.run(print_data=True, save_path=out_dir)
        plt.close("all")
