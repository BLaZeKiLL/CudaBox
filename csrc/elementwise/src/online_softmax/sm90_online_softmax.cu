#include "cuda_utils.cuh"
#include "cudabox_ops.hpp"
#include "logger.hpp"
#include "torch_utils.hpp"

#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

// NOTE: cute/tensor.hpp is the umbrella header and MUST be included before any
// standalone cute/atom/* or cute/arch/* headers. Including copy_traits_sm90_tma
// (or other atom headers) first pulls in cute/algorithm/copy.hpp before
// Copy_Atom is defined, which triggers a bogus "copy_if already declared /
// template parameter pack not at end" cascade in copy.hpp.
#include <cute/tensor.hpp>

#include <cute/atom/copy_traits_sm90_tma.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace cudabox::elementwise::sm90 {

// =====================================================================
// Configuration
// =====================================================================
static constexpr int STAGES = 4;
static constexpr int TILE_SIZE = 256; // tile size along row
static constexpr int PRODUCER_WARPS = 1;
static constexpr int CONSUMER_WARPS = 3;
static constexpr int CONSUMER_THREADS = CONSUMER_WARPS * 32;
static constexpr int BLOCK_THREADS = (PRODUCER_WARPS + CONSUMER_WARPS) * 32;
static constexpr int CONSUMER_BARRIER_ID = 0;

using Pipeline = cutlass::PipelineTmaAsync<STAGES>;

// =====================================================================
// MaxSum reduction
// =====================================================================
struct MaxSum {
  float max_val;
  float sum_exp;
};

constexpr MaxSum MaxSumIdentity = {-INFINITY, 0.0f};

struct MaxSumOp {
  __device__ MaxSum operator()(const MaxSum &a, const MaxSum &b) const {
    MaxSum result;
    result.max_val = fmax(a.max_val, b.max_val);
    // expf performs the normalization adjustment and the operand order is
    // important for normalization correction effect.
    //
    // Guard the both-empty case: when both operands are the identity
    // ({-INF, 0}) result.max is -INF, and the scale expf(-INF - (-INF)) is
    // expf(NaN) = NaN, which would poison the reduction. Empty warps occur
    // whenever block_size > cols, so this matters for small rows.
    result.sum_exp = result.max_val == -INFINITY
                         ? 0.0f
                         : a.sum_exp * expf(a.max_val - result.max_val) +
                               b.sum_exp * expf(b.max_val - result.max_val);
    return result;
  }
};

// =====================================================================
// Shared memory layout
// =====================================================================
struct SharedStorage {
  // Multi-buffered smem tiles for the TMA load pipeline.
  alignas(128) float data[STAGES][TILE_SIZE];

  // CUTLASS TMA pipeline mbarriers.
  typename Pipeline::SharedStorage pipeline_storage;

  // Broadcast reduction results for the current row.
  float row_max;
  float row_inv_sum;

  // Consumer cross-warp reduction scratch.
  MaxSum reduce[CONSUMER_WARPS];
};

// =====================================================================
// Kernel
// =====================================================================
template <class TmaLoad, class TmaStore>
__global__ __launch_bounds__(BLOCK_THREADS) void softmax_persistent_cute_kernel(
    CUTE_GRID_CONSTANT TmaLoad const tma_load,
    CUTE_GRID_CONSTANT TmaStore const tma_store, int num_rows, int num_cols) {
  using namespace cute;

  const auto block = cg::this_thread_block();
  const auto warp = cg::tiled_partition<cutlass::NumThreadsPerWarp>(block);

  extern __shared__ char smem_raw[];
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(smem_raw);

  int tid = threadIdx.x;
  int warp_id = warp.meta_group_rank();
  int lane_id = warp.thread_rank();
  bool is_producer = (warp_id < PRODUCER_WARPS);
  bool is_leader = is_producer && (lane_id == 0);
  int consumer_warp_id = warp_id - PRODUCER_WARPS;
  int consumer_tid = consumer_warp_id * cutlass::NumThreadsPerWarp + lane_id;
  int num_tiles = (num_cols + TILE_SIZE - 1) / TILE_SIZE;

  // -----------------------------------------------------------------
  // TMA tensors and tiling (gmem is modeled column-major: (cols, rows))
  // -----------------------------------------------------------------
  Tensor mIn = tma_load.get_tma_tensor(make_shape(num_cols, num_rows));
  Tensor mOut = tma_store.get_tma_tensor(make_shape(num_cols, num_rows));
  auto cta_tiler = make_shape(Int<TILE_SIZE>{}, Int<1>{});
  auto smem_layout = make_layout(make_shape(Int<TILE_SIZE>{}, Int<1>{}));
  auto cta_load = tma_load.get_slice(0);
  auto cta_store = tma_store.get_slice(0);

  // -----------------------------------------------------------------
  // Pipeline setup
  // -----------------------------------------------------------------
  typename Pipeline::Params params;
  params.transaction_bytes = TILE_SIZE * sizeof(float);
  params.role = is_producer ? Pipeline::ThreadCategory::Producer
                            : Pipeline::ThreadCategory::Consumer;
  params.is_leader = is_leader ? 1u : 0u;
  params.num_consumers = CONSUMER_THREADS;
  params.num_producers = 1;

  Pipeline pipeline(smem.pipeline_storage, params, Shape<_1, _1, _1>{});

  // Producer/consumer pipeline states advance monotonically across rows and
  // passes; never reset them.
  auto wr_state = cutlass::make_producer_start_state<Pipeline>();
  typename Pipeline::PipelineState rd_state;

  // Ensure barriers are initialized before any thread uses them.
  __syncthreads();

  MaxSumOp op;

  // -----------------------------------------------------------------
  // Persistent loop over rows
  // -----------------------------------------------------------------
  for (int row = blockIdx.x; row < num_rows; row += gridDim.x) {

    // =============================================================
    // PASS 1: Online reduction -> (row_max, row_inv_sum)
    // =============================================================
    if (is_producer) {
      if (is_leader) {
        for (int tile = 0; tile < num_tiles; ++tile) {
          pipeline.producer_acquire(wr_state);
          int stage = wr_state.index();

          Tensor gIn = local_tile(mIn, cta_tiler, make_coord(tile, row));
          Tensor sIn =
              make_tensor(make_smem_ptr(&smem.data[stage][0]), smem_layout);
          Tensor tgIn = cta_load.partition_S(gIn);
          Tensor tsIn = cta_load.partition_D(sIn);

          copy(tma_load.with(*pipeline.producer_get_barrier(wr_state)), tgIn,
               tsIn);
          ++wr_state;
        }
      }
    } else {
      MaxSum local_ms = MaxSumIdentity;

      for (int tile = 0; tile < num_tiles; ++tile) {
        pipeline.consumer_wait(rd_state);
        int stage = rd_state.index();

        int tile_start = tile * TILE_SIZE;
        int tile_len = min(TILE_SIZE, num_cols - tile_start);

        for (int i = consumer_tid; i < tile_len; i += CONSUMER_THREADS) {
          local_ms = op(local_ms, {smem.data[stage][i], 1.0f});
        }

        pipeline.consumer_release(rd_state);
        ++rd_state;
      }

      // Warp-level reduce, then cross-warp reduce via shared memory.
      MaxSum warp_ms = cg::reduce(warp, local_ms, op);

      if (lane_id == 0) {
        smem.reduce[consumer_warp_id] = warp_ms;
      }
      cutlass::arch::NamedBarrier::sync(CONSUMER_THREADS, CONSUMER_BARRIER_ID);

      if (consumer_warp_id == 0) {
        const MaxSum val =
            (lane_id < CONSUMER_WARPS) ? smem.reduce[lane_id] : MaxSumIdentity;
        MaxSum final_ms = cg::reduce(warp, val, op);
        if (lane_id == 0) {
          smem.row_max = final_ms.max_val;
          smem.row_inv_sum = 1.0f / final_ms.sum_exp;
        }
      }
    }

    // All threads converge here; makes row_max/row_inv_sum visible.
    __syncthreads();

    float g_max = smem.row_max;
    float g_inv_sum = smem.row_inv_sum;

    // =============================================================
    // PASS 2: Reload, normalize, TMA store
    // =============================================================
    if (is_producer) {
      if (is_leader) {
        for (int tile = 0; tile < num_tiles; ++tile) {
          pipeline.producer_acquire(wr_state);
          int stage = wr_state.index();

          Tensor gIn = local_tile(mIn, cta_tiler, make_coord(tile, row));
          Tensor sIn =
              make_tensor(make_smem_ptr(&smem.data[stage][0]), smem_layout);
          Tensor tgIn = cta_load.partition_S(gIn);
          Tensor tsIn = cta_load.partition_D(sIn);

          copy(tma_load.with(*pipeline.producer_get_barrier(wr_state)), tgIn,
               tsIn);
          ++wr_state;
        }
      }
    } else {
      for (int tile = 0; tile < num_tiles; ++tile) {
        pipeline.consumer_wait(rd_state);
        int stage = rd_state.index();

        int tile_start = tile * TILE_SIZE;
        int tile_len = min(TILE_SIZE, num_cols - tile_start);

        // Normalize in-place in smem.
        for (int i = consumer_tid; i < tile_len; i += CONSUMER_THREADS) {
          smem.data[stage][i] = expf(smem.data[stage][i] - g_max) * g_inv_sum;
        }

        // Fence generic smem writes before the async-proxy TMA store reads
        // them.
        cute::tma_store_fence();

        // Descriptor setup is pure address/layout math (independent of the
        // synchronized smem data), so compute it before the barrier; only the
        // copy below actually reads smem and must wait for all writers.
        Tensor gOut = local_tile(mOut, cta_tiler, make_coord(tile, row));
        Tensor sOut =
            make_tensor(make_smem_ptr(&smem.data[stage][0]), smem_layout);
        Tensor tsOut = cta_store.partition_S(sOut);
        Tensor tgOut = cta_store.partition_D(gOut);

        // Make sure all consumers finished writing before the store reads smem.
        cutlass::arch::NamedBarrier::sync(CONSUMER_THREADS,
                                          CONSUMER_BARRIER_ID);

        // One thread issues the TMA store for the whole tile.
        if (consumer_tid == 0) {
          copy(tma_store, tsOut, tgOut);
          tma_store_arrive();
          tma_store_wait<0>();
        }

        // Store must complete before the stage is released for reuse.
        cutlass::arch::NamedBarrier::sync(CONSUMER_THREADS,
                                          CONSUMER_BARRIER_ID);
        pipeline.consumer_release(rd_state);
        ++rd_state;
      }
    }

    __syncthreads();
  }
}

// =====================================================================
// Host: build TMA copies with CuTe + launch
// =====================================================================
static cudaError_t softmax_persistent_launch(float *d_input, float *d_output,
                                             int num_rows, int num_cols,
                                             cudaStream_t stream) {
  using namespace cute;

  // gmem modeled column-major so each (tile, row) tile stays within one row:
  // element (col, row) lives at col + row * num_cols.
  Tensor gIn =
      make_tensor(make_gmem_ptr(d_input), make_shape(num_cols, num_rows),
                  make_stride(_1{}, num_cols));
  Tensor gOut =
      make_tensor(make_gmem_ptr(d_output), make_shape(num_cols, num_rows),
                  make_stride(_1{}, num_cols));

  auto smem_layout = make_layout(make_shape(Int<TILE_SIZE>{}, Int<1>{}));

  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gIn, smem_layout);
  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, gOut, smem_layout);

  // Persistent grid: one block per SM.
  int sm_count;
  CUDABOX_CUDA_CALL(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

  size_t smem_size = sizeof(SharedStorage);
  auto kernel =
      &softmax_persistent_cute_kernel<decltype(tma_load), decltype(tma_store)>;
  CUDABOX_CUDA_CALL(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(sm_count);
  config.blockDim = dim3(BLOCK_THREADS);
  config.stream = stream;
  config.dynamicSmemBytes = smem_size;

  CUDABOX_LOG_DEBUG("Dispatching sm90 online_softmax, rows={}, cols={}",
                    num_rows, num_cols);
  CUDABOX_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, tma_load, tma_store,
                                       num_rows, num_cols));

  return cudaSuccess;
}

torch::Tensor online_softmax(const torch::Tensor &tensor) {
  TORCH_TENSOR_CHECK(tensor);

  TORCH_CHECK(tensor.dim() == 2,
              "sm90 online softmax only supports 2D tensors");
  TORCH_CHECK(tensor.is_contiguous(),
              "sm90 online softmax only supports contiguous tensors");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              "sm90 online softmax only supports float32 tensors");

  const c10::cuda::OptionalCUDAGuard device_guard(tensor.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const int num_rows = static_cast<int>(tensor.size(0));
  const int num_cols = static_cast<int>(tensor.size(1));

  auto out = torch::empty_like(tensor);

  softmax_persistent_launch(tensor.data_ptr<float>(), out.data_ptr<float>(),
                            num_rows, num_cols, stream);

  return out;
}

} // namespace cudabox::elementwise::sm90
