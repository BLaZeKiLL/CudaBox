import cudabox
import pytest
import torch


def histogram_ref(x: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Reference mirroring the kernel's binning math.

    The kernel bins a non-negative input into ``num_bins`` equal-width buckets
    over ``[0, max]`` via ``bin = trunc(v / (max / num_bins))``. We replicate the
    exact float32 arithmetic so interior bin assignments match bit-for-bit, and
    fold the maximum element into the last bin (right-edge inclusive convention).
    """
    x = x.to(torch.float32)
    max_val = x.max()
    step = max_val / num_bins
    idx = (x / step).to(torch.int)
    idx = idx.clamp_(0, num_bins - 1)
    return torch.bincount(idx.to(torch.int64), minlength=num_bins)[:num_bins].to(
        torch.int32
    )


@pytest.mark.parametrize("N", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("num_bins", [4, 8, 16, 32, 64, 128, 256])
def test_histogram(N, num_bins):
    A = torch.rand((N,), device="cuda")

    out_ref = histogram_ref(A, num_bins)
    out = cudabox.algorithms.histogram(A, num_bins)

    assert out.shape == (num_bins,)
    # every element must land in exactly one bin
    assert int(out.sum().item()) == N
    torch.testing.assert_close(out, out_ref)


if __name__ == "__main__":
    pytest.main([__file__])
