#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "cuda_utils.cuh"
#include "cudabox_ops.hpp"
#include "logger.hpp"
#include "torch_utils.hpp"

namespace cg = cooperative_groups;

namespace cudabox::elementwise {

// rmsnorm(xi) = xi / sqrt(sum(xi^2) / N + eps)
// input = (B, N), normalize over N
__global__ __launch_bounds__(utils::THREADS_PER_BLOCK) void rmsnorm_kernel(
    const float *input, const float *gamma, float *output, const double eps,
    const unsigned int rows, const unsigned int cols) {
  auto cluster = cg::this_cluster();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const unsigned int bid = blockIdx.x;
  const unsigned int rowid = bid / utils::BLOCKS_PER_CLUSTER;

  const unsigned int tid = threadIdx.x;
  const unsigned int rtid =
      (bid % utils::BLOCKS_PER_CLUSTER) * blockDim.x + tid;

  // This kernel uses one cluster (BLOCKS_PER_CLUSTER blocks) per row and reads
  // each row with plain scalar loads, NOT vectorized float4 loads.
  //
  // float4 vectorization can't be applied for an arbitrary `cols`: row `r`
  // starts at element `r * cols`, so in float4 units its base is `r * cols /
  // 4`. That is only an integer -- and only 16-byte aligned, as a 128-bit load
  // requires -- when `cols % 4 == 0`. For other `cols` the per-row float4
  // offset `r * (cols / 4)` points at the wrong element for every row past the
  // first (e.g. cols=111: row 1 starts at element 111, but 1 * (111/4) * 4 =
  // 108), so the reads/writes silently corrupt data. Vectorizing for any `cols`
  // would require zero-padding each row up to a multiple of 4 so every row base
  // is aligned; we keep the scalar path here for generality.
  const unsigned int rowoffset = rowid * cols;

  __shared__ float warp_sums[utils::NUM_WARPS];
  __shared__ float block_sum;
  __shared__ float rms_factor;

  // thread reduction: cluster-strided over the row
  float thread_sum = 0.0f;
  for (auto i = rtid; i < cols; i += cluster.num_threads()) {
    const float in = input[rowoffset + i];
    thread_sum += in * in;
  }

  // warp reduction
  float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>());
  auto warp_id = warp.meta_group_rank();
  cg::invoke_one(warp, [&] { warp_sums[warp_id] = warp_sum; });
  block.sync();

  // block reduction
  if (tid < utils::NUM_WARPS) {
    auto sub_warps = cg::tiled_partition<utils::NUM_WARPS>(block);
    float value = warp_sums[tid];
    float block_sum_partial = cg::reduce(sub_warps, value, cg::plus<float>());
    cg::invoke_one(sub_warps, [&] { block_sum = block_sum_partial; });
  }
  cluster.sync();

  // cluster reduction
  if (tid < utils::BLOCKS_PER_CLUSTER) {
    auto sub_warps = cg::tiled_partition<utils::BLOCKS_PER_CLUSTER>(block);
    float value = tid == cluster.block_rank()
                      ? block_sum
                      : *cluster.map_shared_rank(&block_sum, tid);
    float sum_squared = cg::reduce(sub_warps, value, cg::plus<float>());
    rms_factor = 1.0f / sqrtf(sum_squared / cols + eps);
  }

  // this still needs to be a cluster sync so that all blocks
  // have finished reading each others block_sum in smem
  cluster.sync();

  // write output: cluster-strided over the row
  for (auto i = rtid; i < cols; i += cluster.num_threads()) {
    output[rowoffset + i] = input[rowoffset + i] * gamma[i] * rms_factor;
  }
}

cudaError_t rmsnorm_launch(const float *input, const float *gamma,
                           float *output, const double eps,
                           const unsigned int rows, const unsigned int cols,
                           cudaStream_t stream = 0) {
  cudaLaunchConfig_t config{};
  config.blockDim = utils::THREADS_PER_BLOCK;
  config.gridDim = rows * utils::BLOCKS_PER_CLUSTER;
  config.stream = stream;

  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = utils::BLOCKS_PER_CLUSTER;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;

  config.attrs = attr;
  config.numAttrs = 1;

  CUDABOX_LOG_DEBUG("Dispatching rmsnorm, rows={}, cols={}", rows, cols);
  CUDABOX_CUDA_CALL(cudaLaunchKernelEx(&config, rmsnorm_kernel, input, gamma,
                                       output, eps, rows, cols));

  return cudaSuccess;
}

torch::Tensor rmsnorm(const torch::Tensor &tensor, const torch::Tensor &gamma,
                      const double eps) {
  TORCH_TENSOR_CHECK(tensor);
  TORCH_TENSOR_CHECK(gamma);

  TORCH_CHECK(tensor.dim() == 2, "rmsnorm only supports 2D tensors");
  TORCH_CHECK(gamma.dim() == 1, "gamma only supports 1D tensors");
  TORCH_CHECK(tensor.is_contiguous(),
              "rmsnorm only supports contiguous tensors");
  TORCH_CHECK(gamma.is_contiguous(), "gamma only supports contiguous tensors");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              "rmsnorm only supports float32 tensors");
  TORCH_CHECK(gamma.scalar_type() == torch::kFloat32,
              "gamma only supports float32 tensors");

  auto device = tensor.device();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const int64_t rows = tensor.size(0);
  const int64_t cols = tensor.size(1);

  auto out = torch::empty_like(tensor);

  cudaError_t status =
      rmsnorm_launch(tensor.data_ptr<float>(), gamma.data_ptr<float>(),
                     out.data_ptr<float>(), eps, rows, cols, stream);

  TORCH_CHECK(status == cudaSuccess, "rmsnorm failed");

  return out;
}
} // namespace cudabox::elementwise
