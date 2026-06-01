import os
from typing import Callable, Sequence, Tuple

import torch
import triton.testing

DEFAULT_DTYPE = torch.float32
DEFAULT_DEVICE = "cuda"
DEFAULT_QUANTILES = [0.5, 0.2, 0.8]


def run_benchmark(
    fn: Callable,
    quantiles: Sequence[float] = (),
    scale: float = 1.0,
) -> Tuple[float, float, float]:
    """Execute benchmark using CUDA graph and return times in microseconds.

    Args:
        fn: Function to benchmark
        quantiles: Quantiles for timing measurements [median, min, max]
        scale: Scale the result down (usually num_layers).

    Returns:
        Tuple of (median_us, max_us, min_us)
    """
    quantiles = list(quantiles or DEFAULT_QUANTILES)
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms / scale, 1000 * max_ms / scale, 1000 * min_ms / scale


# Monkey-patch triton.testing.Mark.run to delete its `results.html` index
# page after every invocation. The HTML is a thin gallery of `<img>` tags
# pointing at the PNGs we already keep, it gets overwritten on each .run()
# call (so the content is meaningless past the first invocation), and it
# clutters git status. Triton doesn't expose a flag to disable it, so we
# patch it once at import time. Every bench module that imports from
# `benchmarks.utils` automatically gets the cleanup behavior.
_orig_mark_run = triton.testing.Mark.run


def _mark_run_no_html(self, *args, **kwargs):
    result = _orig_mark_run(self, *args, **kwargs)
    save_path = kwargs.get("save_path", "")
    if save_path:
        html_path = os.path.join(save_path, "results.html")
        if os.path.exists(html_path):
            os.remove(html_path)
    return result


triton.testing.Mark.run = _mark_run_no_html
