#include "cudabox_ops.hpp"
#include "torch_utils.hpp"

namespace cudabox::gemm
{
    torch::Tensor sgemm(const torch::Tensor &mat_a, const torch::Tensor &mat_b)
    {
        TORCH_TENSOR_CHECK(mat_a);
        TORCH_TENSOR_CHECK(mat_b);

        torch::Tensor out = torch::empty({mat_a.size(0), mat_b.size(1)}, torch::dtype(mat_a.dtype()).device(torch::kCUDA));

        return out;
    }
} // namespace cudabox::gemm
