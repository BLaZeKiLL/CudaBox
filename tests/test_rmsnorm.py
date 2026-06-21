import cudabox
import pytest
import torch
import torch.nn.functional as F


def rmsnorm_ref(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference using PyTorch's built-in RMSNorm.

    The kernel normalizes each row over ``cols`` and scales by the per-column
    weight ``gamma`` of shape ``(cols,)``, which is exactly ``F.rms_norm`` with
    ``normalized_shape=(cols,)`` and ``weight=gamma``.
    """
    return F.rms_norm(x, (x.size(1),), weight=gamma, eps=eps)


@pytest.mark.parametrize("rows", [1, 8, 99, 512])
@pytest.mark.parametrize("cols", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_rmsnorm(rows, cols, eps):
    x = torch.rand((rows, cols), device="cuda", dtype=torch.float32)
    gamma = torch.rand((cols,), device="cuda", dtype=torch.float32)

    out_ref = rmsnorm_ref(x, gamma, eps)
    out = cudabox.elementwise.rmsnorm(x, gamma, eps)

    assert out.shape == (rows, cols)
    torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
