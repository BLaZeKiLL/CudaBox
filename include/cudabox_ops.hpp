#pragma once

#include <torch/torch.h>

namespace cudabox {

namespace algorithms {

torch::Tensor histogram(const torch::Tensor &tensor, int64_t num_bins);

} // namespace algorithms

namespace elementwise {

torch::Tensor softmax(const torch::Tensor &tensor);

}

namespace gemm {

torch::Tensor simple_gemm(const torch::Tensor &mat_a,
                          const torch::Tensor &mat_b);

torch::Tensor tiled_gemm(const torch::Tensor &mat_a,
                         const torch::Tensor &mat_b);

namespace sm90_pipelined_tma_mma {

torch::Tensor gemm(const torch::Tensor &mat_a, const torch::Tensor &mat_b);

} // namespace sm90_pipelined_tma_mma

} // namespace gemm

} // namespace cudabox
