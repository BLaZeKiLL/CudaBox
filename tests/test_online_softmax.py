import cudabox
import pytest
import torch


@pytest.mark.parametrize("rows", [1, 8, 99, 512])
@pytest.mark.parametrize("cols", [128, 500, 1024, 3072, 3584, 4096, 8192, 16384])
def test_online_softmax(rows, cols):
    # softmax is over the last dim (each row independently). `cols` are kept
    # multiples of 4 because the kernel vectorizes each row with float4 loads,
    # which require the row base (input + row*cols) to stay 16-byte aligned.
    x = torch.rand((rows, cols), device="cuda", dtype=torch.float32)

    out_ref = torch.softmax(x, dim=1)
    out = cudabox.elementwise.online_softmax(x)

    assert out.shape == (rows, cols)
    torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
