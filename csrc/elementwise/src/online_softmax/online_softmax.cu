#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "cuda_utils.cuh"
#include "cudabox_ops.hpp"
#include "logger.hpp"
#include "torch_utils.hpp"

namespace cg = cooperative_groups;

namespace cudabox::elementwise {

struct MaxSum {
  float max;
  float exp_sum;
};

static constexpr MaxSum MaxSumIdentity = {-INFINITY, 0.0f};

struct MaxSumOp {
  __device__ MaxSum operator()(const MaxSum &a, const MaxSum &b) {
    MaxSum result;
    result.max = fmax(a.max, b.max);
    // expf performs the normalization adjustment and the operand order is
    // important for normalization correction effect.
    //
    // Guard the both-empty case: when both operands are the identity
    // ({-INF, 0}) result.max is -INF, and the scale expf(-INF - (-INF)) is
    // expf(NaN) = NaN, which would poison the reduction. Empty warps occur
    // whenever block_size > cols, so this matters for small rows.
    result.exp_sum = result.max == -INFINITY
                         ? 0.0f
                         : a.exp_sum * expf(a.max - result.max) +
                               b.exp_sum * expf(b.max - result.max);
    return result;
  }
};

namespace block_reduce {

__global__ void online_softmax_kernel(const float *input, float *output,
                                      const int64_t rows, const int64_t cols) {
  const auto block = cg::this_thread_block();
  const auto warp = cg::tiled_partition<utils::THREADS_PER_WARP>(block);

  const auto bid = blockIdx.x;
  const auto tid = threadIdx.x;

  const auto block_size = blockDim.x;

  const auto *row_input = input + bid * cols;
  auto *row_output = output + bid * cols;

  MaxSumOp max_sum_op{};

  __shared__ MaxSum smem_max_sums[utils::NUM_WARPS];
  __shared__ MaxSum block_max_sum;

  // Thread reduction
  MaxSum thread_max_sum = MaxSumIdentity;
  for (auto i = tid; i < cols; i += block_size) {
    thread_max_sum = max_sum_op(thread_max_sum, {row_input[i], 1.0f});
  }

  // Warp reduction
  const auto warp_id = warp.meta_group_rank();
  MaxSum warp_max_sum = cg::reduce(warp, thread_max_sum, max_sum_op);
  cg::invoke_one(warp, [&] { smem_max_sums[warp_id] = warp_max_sum; });
  block.sync();

  // Block reduction
  if (warp_id == 0) {
    const auto value =
        tid < utils::NUM_WARPS ? smem_max_sums[tid] : MaxSumIdentity;
    const auto result = cg::reduce(warp, value, max_sum_op);
    cg::invoke_one(warp, [&] { block_max_sum = result; });
  }
  block.sync();

  // Compute output
  const float row_max = block_max_sum.max;
  const float row_sum_inv = 1.0f / block_max_sum.exp_sum;

  for (auto i = tid; i < cols; i += block_size) {
    row_output[i] = expf(row_input[i] - row_max) * row_sum_inv;
  }
}

cudaError_t online_softmax_launch(const float *input, float *output,
                                  const int64_t rows, const int64_t cols,
                                  cudaStream_t stream = 0) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(rows);
  config.blockDim = dim3(utils::THREADS_PER_BLOCK);
  config.stream = stream;

  CUDABOX_LOG_DEBUG("Dispatching online_softmax, rows={}, cols={}", rows, cols);
  CUDABOX_CUDA_CALL(cudaLaunchKernelEx(&config, online_softmax_kernel, input,
                                       output, rows, cols));

  return cudaSuccess;
}

} // namespace block_reduce

namespace cluster_reduce {

__global__ void online_softmax_kernel(const float *input, float *output,
                                      const int64_t rows, const int64_t cols) {
  const auto cluster = cg::this_cluster();
  const auto block = cg::this_thread_block();
  const auto warp = cg::tiled_partition<utils::THREADS_PER_WARP>(block);

  const auto bid = blockIdx.x;
  const auto tid = threadIdx.x;
  const auto row_id = (bid / utils::BLOCKS_PER_CLUSTER);
  const auto rtid = (bid % utils::BLOCKS_PER_CLUSTER) * blockDim.x + tid;

  const auto cluster_size = cluster.num_threads();

  const auto *row_input = input + row_id * cols;
  auto *row_output = output + row_id * cols;

  MaxSumOp max_sum_op{};

  __shared__ MaxSum smem_max_sums[utils::NUM_WARPS];
  __shared__ MaxSum block_max_sum;
  __shared__ MaxSum cluster_max_sum;

  // Thread reduction
  MaxSum thread_max_sum = MaxSumIdentity;
  for (auto i = rtid; i < cols; i += cluster_size) {
    thread_max_sum = max_sum_op(thread_max_sum, {row_input[i], 1.0f});
  }

  // Warp reduction
  const auto warp_id = warp.meta_group_rank();
  MaxSum warp_max_sum = cg::reduce(warp, thread_max_sum, max_sum_op);
  cg::invoke_one(warp, [&] { smem_max_sums[warp_id] = warp_max_sum; });
  block.sync();

  // Block reduction
  if (warp_id == 0) {
    const auto value =
        tid < utils::NUM_WARPS ? smem_max_sums[tid] : MaxSumIdentity;
    const auto result = cg::reduce(warp, value, max_sum_op);
    cg::invoke_one(warp, [&] { block_max_sum = result; });
  }
  cluster.sync();

  // Cluster reduction
  if (warp_id == 0) {
    const auto value = tid < utils::NUM_WARPS
                           ? tid == cluster.block_rank()
                                 ? block_max_sum
                                 : *cluster.map_shared_rank(&block_max_sum, tid)
                           : MaxSumIdentity;
    const auto result = cg::reduce(warp, value, max_sum_op);
    cg::invoke_one(warp, [&] { cluster_max_sum = result; });
  }
  cluster.sync();

  // Compute output
  const float row_max = cluster_max_sum.max;
  const float row_sum_inv = 1.0f / cluster_max_sum.exp_sum;

  for (auto i = rtid; i < cols; i += cluster_size) {
    row_output[i] = expf(row_input[i] - row_max) * row_sum_inv;
  }
}

cudaError_t online_softmax_launch(const float *input, float *output,
                                  const int64_t rows, const int64_t cols,
                                  cudaStream_t stream = 0) {
  const auto blocks = rows * utils::BLOCKS_PER_CLUSTER;

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(blocks);
  config.blockDim = dim3(utils::THREADS_PER_BLOCK);
  config.stream = stream;

  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = utils::BLOCKS_PER_CLUSTER;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;

  config.attrs = attr;
  config.numAttrs = 1;

  CUDABOX_LOG_DEBUG("Dispatching online_softmax, rows={}, cols={}", rows, cols);
  CUDABOX_CUDA_CALL(cudaLaunchKernelEx(&config, online_softmax_kernel, input,
                                       output, rows, cols));

  return cudaSuccess;
}

} // namespace cluster_reduce

torch::Tensor online_softmax(const torch::Tensor &tensor) {
  TORCH_TENSOR_CHECK(tensor);

  TORCH_CHECK(tensor.dim() == 2, "online softmax only supports 2D tensors");
  TORCH_CHECK(tensor.is_contiguous(),
              "online softmax only supports contiguous tensors");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              "online softmax only supports float32 tensors");

  auto device = tensor.device();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const int64_t rows = tensor.size(0);
  const int64_t cols = tensor.size(1);

  auto out = torch::empty_like(tensor);

  if (cols >= 2048) {
    cudaError_t status = cluster_reduce::online_softmax_launch(
        tensor.data_ptr<float>(), out.data_ptr<float>(), rows, cols, stream);

    TORCH_CHECK(status == cudaSuccess, "cluster reduce online softmax failed");
  } else {
    cudaError_t status = block_reduce::online_softmax_launch(
        tensor.data_ptr<float>(), out.data_ptr<float>(), rows, cols, stream);

    TORCH_CHECK(status == cudaSuccess, "block reduce online softmax failed");
  }

  return out;
}

} // namespace cudabox::elementwise
