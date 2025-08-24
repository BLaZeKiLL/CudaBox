#pragma once

#include <torch/torch.h>

namespace cudabox {
namespace gemm {
torch::Tensor simple_gemm(const torch::Tensor &mat_a,
                          const torch::Tensor &mat_b);

torch::Tensor tiled_gemm(const torch::Tensor &mat_a,
                         const torch::Tensor &mat_b);
} // namespace gemm

} // namespace cudabox
