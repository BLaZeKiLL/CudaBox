#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "cuda_utils.cuh"
#include "cudabox_ops.hpp"
#include "logger.hpp"
#include "torch_utils.hpp"

namespace cg = cooperative_groups;

namespace cudabox::algorithms {

struct SharedStorage {
  float max_storage[utils::NUM_WARPS];
  float block_max;
  int bin_storage[];
};

__global__ __launch_bounds__(utils::THREADS_PER_BLOCK) void histogram_kernel(
    const float *input, int *output, int64_t size, int64_t num_bins,
    float *workspace) {
  auto grid = cg::this_grid();
  auto cluster = cg::this_cluster();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const unsigned int tid = threadIdx.x;
  const unsigned int bid = blockIdx.x;
  const unsigned int gtid = bid * blockDim.x + tid;
  const unsigned int gstride = gridDim.x * blockDim.x;

  extern __shared__ char shared_memory[];
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  const unsigned int n4 = size >> 2;       // number of full float4s
  const unsigned int tail_start = n4 << 2; // index of first scalar tail element

  // pass 1: thread reduction
  float thread_max = -INFINITY;
  for (int64_t i = gtid; i < n4; i += gstride) {
    auto v = input4[i];
    thread_max = fmaxf(thread_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
  }
  // scalar tail (at most 3 elements)
  if (tail_start + gtid < size) {
    thread_max = fmaxf(thread_max, input[tail_start + gtid]);
  }

  // pass 2: warp reduction
  const unsigned int warp_id =
      warp.meta_group_rank(); // threadIdx.x / utils::THREADS_PER_WARP;

  float warp_max = cg::reduce(warp, thread_max, cg::greater<float>());

  cg::invoke_one(warp, [&] { smem.max_storage[warp_id] = warp_max; });

  // pass 3: block reduction
  block.sync();

  if (tid < utils::NUM_WARPS) {
    const unsigned int lane_id =
        warp.thread_rank(); // threadIdx.x % utils::THREADS_PER_WARP;
    auto sub_group = cg::tiled_partition<utils::NUM_WARPS>(block);
    float value = smem.max_storage[lane_id];
    float block_max = cg::reduce(sub_group, value, cg::greater<float>());
    cg::invoke_one(sub_group, [&] { smem.block_max = block_max; });
  }

  // pass 4: cluster reduction
  cluster.sync();

  const unsigned int cluster_block_rank = cluster.block_rank();
  const unsigned int cluster_num_blocks = cluster.num_blocks();

  if (cluster_block_rank == 0 && tid < cluster_num_blocks) {
    auto sub_group = cg::tiled_partition<utils::BLOCKS_PER_CLUSTER>(block);
    float value = tid == 0 ? smem.block_max
                           : *cluster.map_shared_rank(&smem.block_max, tid);
    float cluster_max = cg::reduce(sub_group, value, cg::greater<float>());
    cg::invoke_one(sub_group,
                   [&] { utils::atomic_max_float(workspace, cluster_max); });
  }

  grid.sync();

  // pass 5: histogram binning
  const float max = workspace[0];
  const float step = max / num_bins;

  int acc = 0;
  int cur_bin = 0;

  for (int i = tid; i < num_bins; i += blockDim.x) {
    smem.bin_storage[i] = 0;
  }

  block.sync();

  // pass 6: smem accumulated binning
  for (int64_t i = gtid; i < size; i += gstride) {
    // step = max / num_bins, so the maximum element yields bin == num_bins.
    // Clamp it into the last bin (right-edge inclusive) to keep it in range and
    // avoid an out-of-bounds write into bin_storage.
    int bin = static_cast<int>(input[i] / step);
    if (bin >= num_bins) {
      bin = static_cast<int>(num_bins) - 1;
    }

    if (bin == cur_bin) {
      acc++;
    } else {
      atomicAdd(&smem.bin_storage[cur_bin], acc);
      cur_bin = bin;
      acc = 1;
    }
  }
  // last bin update
  if (acc > 0) {
    atomicAdd(&smem.bin_storage[cur_bin], acc);
  }

  block.sync();

  // pass 7: write output
  for (int i = tid; i < num_bins; i += blockDim.x) {
    auto bin_val = smem.bin_storage[i];
    if (bin_val > 0) {
      atomicAdd(&output[i], bin_val);
    }
  }
}

cudaError_t histogram_launch(const float *input, int *output, int64_t size,
                             int64_t num_bins, float *workspace,
                             cudaStream_t stream = 0) {
  size_t smem_bytes = sizeof(SharedStorage) + num_bins * sizeof(int);

  // Build the launch config up front so the occupancy query sees the exact
  // block size, dynamic shared-memory footprint, and cluster dimension we
  // actually launch with.
  cudaLaunchConfig_t config{};
  config.blockDim = dim3(utils::THREADS_PER_BLOCK);
  config.stream = stream;
  config.dynamicSmemBytes = smem_bytes;

  cudaLaunchAttribute attributes[2];
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim.x = utils::BLOCKS_PER_CLUSTER;
  attributes[0].val.clusterDim.y = 1;
  attributes[0].val.clusterDim.z = 1;
  attributes[1].id = cudaLaunchAttributeCooperative;
  attributes[1].val.cooperative = 1;
  config.attrs = attributes;
  config.numAttrs = 2;

  int requested_blocks =
      (size + utils::THREADS_PER_BLOCK - 1) / utils::THREADS_PER_BLOCK;
  int requested_clusters = (requested_blocks + utils::BLOCKS_PER_CLUSTER - 1) /
                           utils::BLOCKS_PER_CLUSTER;

  // cudaOccupancyMaxActiveClusters validates the full launch config, so gridDim
  // must already be a valid multiple of the cluster dimension before the query
  // (a zero gridDim is rejected as a cluster misconfiguration).
  config.gridDim = dim3(requested_clusters * utils::BLOCKS_PER_CLUSTER);

  // A cooperative cluster launch requires every block to be co-resident.
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor ignores the cluster
  // constraint and overestimates, so the launch is rejected with "too many
  // blocks in cooperative launch" at large sizes. Query the cluster-aware
  // occupancy to get the true number of simultaneously resident clusters.
  int max_active_clusters = 0;
  CUDABOX_CUDA_CALL(cudaOccupancyMaxActiveClusters(&max_active_clusters,
                                                   histogram_kernel, &config));

  int clusters = std::max(1, std::min(requested_clusters, max_active_clusters));
  int blocks = clusters * utils::BLOCKS_PER_CLUSTER;
  config.gridDim = dim3(blocks);

  CUDABOX_LOG_DEBUG("Dispatching histogram, size={}, blocks={}, clusters={}, "
                    "max_active_clusters={}, smem_bytes={}",
                    size, blocks, clusters, max_active_clusters, smem_bytes);
  CUDABOX_CUDA_CALL(cudaLaunchKernelEx(&config, histogram_kernel, input, output,
                                       size, num_bins, workspace));

  return cudaSuccess;
}

torch::Tensor histogram(const torch::Tensor &tensor, int64_t num_bins) {
  TORCH_TENSOR_CHECK(tensor);

  TORCH_CHECK(tensor.dim() == 1, "histogram only supports 1D tensors");
  TORCH_CHECK(tensor.is_contiguous(),
              "histogram only supports contiguous tensors");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              "histogram only supports float32 tensors");

  auto device = tensor.device();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const int64_t size = tensor.size(0);

  auto out = torch::zeros({num_bins}, tensor.options().dtype(torch::kInt32));
  auto workspace = torch::empty({1}, tensor.options().dtype(torch::kFloat32));

  auto status =
      histogram_launch(tensor.data_ptr<float>(), out.data_ptr<int>(), size,
                       num_bins, workspace.data_ptr<float>(), stream);

  TORCH_CHECK(status == cudaSuccess, "histogram failed");
  return out;
}

} // namespace cudabox::algorithms
