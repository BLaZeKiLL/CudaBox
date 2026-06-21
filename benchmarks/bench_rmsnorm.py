import os

import torch
import torch.nn.functional as F
import triton
import triton.testing
from cudabox.elementwise import rmsnorm as cudabox_rmsnorm
from utils import DEFAULT_DEVICE, DEFAULT_DTYPE, run_benchmark

EPS = 1e-5

# Rows (batch) are swept across separate plots; columns are swept on the
# x-axis within each plot.
ROWS = [1, 16, 64, 256, 1024, 4096]
COLS = sorted(set([1536, 3072, 4096, 5120] + [1 << i for i in range(10, 17)]))


def torch_rmsnorm(x, gamma):
    return F.rms_norm(x, (x.size(1),), weight=gamma, eps=EPS)


LINE_VALS = [
    "cudabox_rmsnorm",
    "torch_rmsnorm",
]
LINE_NAMES = [
    "Cudabox RMSNorm",
    "Torch RMSNorm",
]
STYLES = [
    ("blue", "--"),
    ("purple", "-."),
]


def _run(rows: int, cols: int, provider: str):
    input = torch.randn((rows, cols), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    gamma = torch.randn((cols,), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    FN_MAP = {
        "cudabox_rmsnorm": lambda: cudabox_rmsnorm(input, gamma, EPS),
        "torch_rmsnorm": lambda: torch_rmsnorm(input, gamma),
    }
    fn = FN_MAP[provider]
    return run_benchmark(fn)


def _make_benchmark(rows: int) -> triton.testing.Benchmark:
    """Cols on the x-axis at a fixed `rows`. `rows` is bound via `args` so it
    doesn't appear on the plot's x-axis."""
    return triton.testing.Benchmark(
        x_names=["cols"],
        x_vals=COLS,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name=f"rmsnorm-rows{rows}",
        args={"rows": rows},
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    out_dir = os.path.join(os.path.dirname(__file__), "results", "rmsnorm")
    os.makedirs(out_dir, exist_ok=True)

    for rows in ROWS:
        benchmark = triton.testing.perf_report([_make_benchmark(rows)])(_run)
        benchmark.run(print_data=True, save_path=out_dir)
        plt.close("all")
