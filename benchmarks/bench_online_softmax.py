import os

import torch
import triton
import triton.testing
from cudabox.elementwise import online_softmax as cudabox_online_softmax
from cudabox.elementwise import sm90_online_softmax as cudabox_sm90_online_softmax
from utils import DEFAULT_DEVICE, DEFAULT_DTYPE, run_benchmark

# Rows (batch) are swept across separate plots; columns are swept on the
# x-axis within each plot.
ROWS = [1, 16, 64, 256, 1024, 4096]
COLS = sorted(set([1536, 3072, 4096, 5120] + [1 << i for i in range(10, 17)]))


def torch_softmax(x):
    return torch.softmax(x, dim=1)


LINE_VALS = [
    "cudabox_sm90_online_softmax",
    "cudabox_online_softmax",
    "torch_softmax",
]
LINE_NAMES = [
    "Cudabox SM90 Online Softmax",
    "Cudabox Online Softmax",
    "Torch Softmax",
]
STYLES = [
    ("black", "-"),
    ("blue", "--"),
    ("purple", "-."),
]


def _run(rows: int, cols: int, provider: str):
    input = torch.randn((rows, cols), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    FN_MAP = {
        "cudabox_sm90_online_softmax": lambda: cudabox_sm90_online_softmax(input),
        "cudabox_online_softmax": lambda: cudabox_online_softmax(input),
        "torch_softmax": lambda: torch_softmax(input),
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
        plot_name=f"online-softmax-rows{rows}",
        args={"rows": rows},
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    out_dir = os.path.join(os.path.dirname(__file__), "results", "online_softmax")
    os.makedirs(out_dir, exist_ok=True)

    for rows in ROWS:
        benchmark = triton.testing.perf_report([_make_benchmark(rows)])(_run)
        benchmark.run(print_data=True, save_path=out_dir)
        plt.close("all")
