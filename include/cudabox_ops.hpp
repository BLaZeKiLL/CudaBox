#pragma once

#include <torch/library.h>
#include <torch/torch.h>

namespace cudabox
{
    namespace gemm
    {
        torch::Tensor sgemm(const torch::Tensor &mat_a, const torch::Tensor &mat_b);
    } // namespace gemm

} // namespace cudabox
