#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"

#include "cuda_utils.cuh"
#include "gemm/sm90_pipelined_tma_mma_gemm/traits.cuh"

namespace cudabox::gemm::sm90_pipelined_tma_mma {

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto
convert_type(cute::Tensor<Engine, Layout> const &tensor) {
  using namespace cute;

  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(
          tensor.data()));
  return make_tensor(cute::make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <class ElementA, class ElementB, class ElementC, class SmemLayoutA,
          class SmemLayoutB, class SmemLayoutC>
struct SharedStorage {
  cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;

  // Only one of smem_B or smem_C can be used at a time
  // Mainloop uses smem_B, Epilogue uses smem_C
  // Barrier is used to synchronize between mainloop and epilogue
  union {
    cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_B;
    cute::array_aligned<ElementC, cute::cosize_v<SmemLayoutC>> smem_C;
  };

  // Last mode of the smem layout corresponds to pipeline size
  uint64_t tma_barrier[cute::size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[cute::size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler, class ClusterShape, class TA,
          class SmemLayoutA, class TmaA, class TB, class SmemLayoutB,
          class TmaB, class TC, class SmemLayoutC, class TmaC, class TiledMma,
          class SmemCopyAtomC>
__global__ static __launch_bounds__(
    decltype(size(TiledMma{}))::
        value) void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                                ClusterShape cluster_shape, TA const *A,
                                CUTLASS_GRID_CONSTANT TmaA const tma_a,
                                TB const *B,
                                CUTLASS_GRID_CONSTANT TmaB const tma_b, TC *C,
                                CUTLASS_GRID_CONSTANT TmaC const tma_store,
                                TiledMma mma, SmemCopyAtomC) {
  using namespace cute;

  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K)); // (M,K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K)); // (N,K) TMA Tensor

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m,n,k)
  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
  auto cluster_layout = make_layout(cluster_shape);

  // CTA-in-cluster coord & multicast mask for A.
  // The vmnk layout prepends a value mode V to the cluster shape.
  auto cluster_layout_vmnk = tiled_divide(
      make_layout(ClusterShape{}), make_tile(typename TiledMma::AtomThrID{}));
  auto cta_in_cluster_coord_vmnk =
      cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
  // A is multicast along the N mode (mode 2 of vmnk).
  uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(
      cluster_layout_vmnk, cta_in_cluster_coord_vmnk);

  // Tile mA by cta_tiler based on cta_coord with stepping on bM and bK
  // resulting tile = (BLK_M,BLK_K,k), where k = number of k-tiles
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  // Tile mB by cta_tiler based on cta_coord with stepping on bN and bK
  // resulting tile = (BLK_N,BLK_K,k), where k = number of k-tiles
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});

  // Raw smem allocation and shared storage binding
  extern __shared__ char shared_memory[];
  using SharedStorage =
      SharedStorage<TA, TB, TC, SmemLayoutA, SmemLayoutB, SmemLayoutC>;
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  // SMEM Tile A = (BLK_M,BLK_K,PIPE)
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutA{});
  // SMEM Tile B = (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.data()), SmemLayoutB{});

  // Partition the copying of A and B tiles
  // These are TMA partitionings, which have a dedicated custom partitioner.
  // A: The cluster rand and cluster layout are used for TMA multicast.
  // B: The Int<0>, Layout<_1> indicates that the TMAs are not multicasted.
  // The group_modes<0,2> transforms the (X,Y,Z)-shaped tensors into
  // ((X,Y),Z)-shaped tensors, group_modes<I,J> groups [I,J) modes of the tensor
  auto [tAgA, tAsA] =
      tma_partition(tma_a, block_rank_in_cluster, cluster_layout,
                    group_modes<0, 2>(sA), group_modes<0, 2>(gA));
  auto [tBgB, tBsB] =
      tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                    group_modes<0, 2>(gB));

  // TMA bytes for mem barrier
  constexpr int kTmaTransactionBytes =
      CUTE_STATIC_V(size<0>(tAsA)) * sizeof(TA) +
      CUTE_STATIC_V(size<0>(tBsB)) * sizeof(TB);

  // Max pipeline depth usage
  auto K_PIPE_MAX = size<1>(tAsA);
  // Total count of tiles
  int k_tile_count = size<1>(tAgA);
  // Current tile index in gmem to read from
  int k_tile = 0;

  // Initialize Barriers
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t *producer_mbar = smem.tma_barrier;
  uint64_t *consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier; // TMA
  using ConsumerBarType = cutlass::arch::ClusterBarrier;            // MMA
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[pipe], 1);
      ConsumerBarType::init(&consumer_mbar[pipe], size(mma));
    }
  }

  // Ensure barrier init is complete on all CTAs
  cluster_sync();

  // Issue the first TMA
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if (k_tile_count > 0 && (warp_idx == 0) && lane_predicate) {
      // Set expected Tx Bytes after each reset / init
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe],
                                            kTmaTransactionBytes);
      copy(tma_a.with(producer_mbar[pipe], tma_mcast_mask_a), tAgA(_, k_tile),
           tAsA(_, pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
    }
    --k_tile_count;
    ++k_tile;
  }

  // Initialize MMA Tensor Descriptors
  // The tCrA and tCrB are actually Tensors of MMA Descriptors constructed as
  // views of SMEM. The MMA Descriptor generation is automatic via inspection
  // and validation of the SMEM Layouts. Because the MMA reads directly from
  // SMEM and the fragments are descriptors rather than registers, there is no
  // need for copy(tCsA, tCrA) in the mainloop.
  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);

  // Accumulator
  Tensor tCrC = partition_fragment_C(mma, select<0, 1>(cta_tiler));
  clear(tCrC);

  // Allocate descriptor iterators
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  // A PipelineState is a circular pipe index [.index()] and a pipe phase
  // [.phase()] that flips each cycle through K_PIPE_MAX.
  // More advanced pipeline and warp-specialization strategies are available in
  // CUTLASS mainloops.
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>(); // TMA writes
  auto read_state = cutlass::PipelineState<K_PIPE_MAX>();  // MMA  reads

  // Mainloop
  CUTE_NO_UNROLL
  while (k_tile_count > -K_PIPE_MAX) {
    // Wait for Producer to complete
    int read_pipe = read_state.index();
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    // MMAs to cover 1 K_TILE
    warpgroup_arrive();
    cute::gemm(mma, tCrA(_, _, _, read_pipe), tCrB(_, _, _, read_pipe), tCrC);
    warpgroup_commit_batch();

    // Wait for all MMAs in a K_TILE to complete
    warpgroup_wait<0>();

    // Notify that consumption is done
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    ++read_state;

    if (k_tile_count > 0 && (warp_idx == 0) && lane_predicate) {
      int pipe = write_state.index();
      // Wait for Consumer to complete consumption
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
      // Set expected Tx Bytes after each reset / init
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe],
                                            kTmaTransactionBytes);
      copy(tma_a.with(producer_mbar[pipe], tma_mcast_mask_a), tAgA(_, k_tile),
           tAsA(_, pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
      ++write_state;
    }
    --k_tile_count;
    ++k_tile;
  }

  // Epilogue
  // Make sure all warpgroups have finished mma
  cutlass::arch::NamedBarrier::sync(size(mma), 0);

  // SmemCopyAtomC is supplied as a template parameter (per-dtype trait).
  Tensor sC = make_tensor(make_smem_ptr(smem.smem_C.data()), SmemLayoutC{});

  auto smem_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomC{}, mma);
  auto smem_thr_copy_C = smem_tiled_copy_C.get_thread_slice(threadIdx.x);

  // include this convert utility
  Tensor tCrC_out = convert_type<TC>(tCrC);
  Tensor taccCrC = smem_thr_copy_C.retile_S(tCrC_out);
  Tensor taccCsC = smem_thr_copy_C.partition_D(sC);
  cute::copy(smem_tiled_copy_C, taccCrC, taccCsC);
  cute::tma_store_fence(); // ensure smem writes are visible to

  // NumThreads == size(mma)
  cutlass::arch::NamedBarrier::arrive(
      size(mma) + cutlass::NumThreadsPerWarp,
      cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

  // Prepare TMA store
  Tensor mC = tma_store.get_tma_tensor(make_shape(M, N));
  Tensor gC = local_tile(mC, select<0, 1>(cta_tiler),
                         make_coord(blockIdx.x, blockIdx.y));
  auto block_tma_store = tma_store.get_slice(_0{}); // CTA slice
  Tensor tCgC = block_tma_store.partition_D(gC);    // (TMA, TMA_M, TMA_K)
  Tensor tCsC = block_tma_store.partition_S(sC);    // (TMA, TMA_M, TMA_K)

  // TMA STORE: SMEM -> GMEM
  if (warp_idx == 0) {
    // Ensure RMEM -> SMEM copy completes before issuing TMA store
    cutlass::arch::NamedBarrier::sync(
        size(mma) + cutlass::NumThreadsPerWarp,
        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  }

  if (warp_idx == 0 && lane_predicate) {
    cute::copy(tma_store, tCsC, tCgC);
  }
}

template <class TA, class TB, class TC, class Traits = gemm_traits<TA>>
cudaError_t gemm_host(int m, int n, int k, TA const *A, int ldA, TB const *B,
                      int ldB, TC *C, int ldC, cudaStream_t stream = 0) {
  using namespace cute;

  static_assert(std::is_same_v<TA, TB>,
                "gemm_host: A and B must share the same element type");
  static_assert(std::is_same_v<TA, TC>,
                "gemm_host: C element type must match A/B in this kernel");

  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  // Row-major equivalent to LayoutRight
  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(ldC, Int<1>{});

  // Tile + cluster + pipeline pulled from per-dtype traits.
  auto bM = typename Traits::BlockM{};
  auto bN = typename Traits::BlockN{};
  auto bK = typename Traits::BlockK{};
  auto bP = typename Traits::PipelineStages{};

  // BlockDim: (BLK_M, BLK_N, BLK_K)
  auto cta_tiler = make_shape(bM, bN, bK);

  // Cluster shape
  using ClusterShape = typename Traits::ClusterShape;
  auto cluster_shape = ClusterShape{};

  // Operand Tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

  // SMEM layouts (swizzled per-dtype atoms come from traits).
  auto sA =
      tile_to_shape(typename Traits::SmemAtomAB{}, make_shape(bM, bK, bP));
  auto sB =
      tile_to_shape(typename Traits::SmemAtomAB{}, make_shape(bN, bK, bP));
  auto sC = tile_to_shape(typename Traits::SmemAtomC{}, make_shape(bM, bN));

  // TMA Atoms
  auto tmaA = make_tma_atom(SM90_TMA_LOAD_MULTICAST{}, mA, sA(_, _, 0),
                            make_shape(bM, bK), size<1>(cluster_shape));
  auto tmaB =
      make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(bN, bK));
  auto tma_store =
      make_tma_copy(SM90_TMA_STORE{}, mC, sC, make_shape(bM, bN), Int<1>{});

  // MMA Atom + tiled MMA from traits.
  using MmaAtom = typename Traits::MmaAtom;
  using AtomLayoutMNK = typename Traits::AtomLayoutMNK;
  TiledMMA tiled_mma = make_tiled_mma(MmaAtom{}, AtomLayoutMNK{});

  // SMEM copy atom for the epilogue (RMEM accumulator -> SMEM C tile).
  using SmemCopyAtomC = typename Traits::SmemCopyAtomC;
  auto smem_copy_atom = SmemCopyAtomC{};

  int smem_size = int(sizeof(
      SharedStorage<TA, TB, TC, decltype(sA), decltype(sB), decltype(sC)>));

  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape),
                  size<2>(cluster_shape));
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster,
                                         smem_size, stream};

  void const *kernel_ptr = reinterpret_cast<void const *>(
      &gemm_device<decltype(prob_shape), decltype(cta_tiler), ClusterShape, TA,
                   decltype(sA), decltype(tmaA), TB, decltype(sB),
                   decltype(tmaB), TC, decltype(sC), decltype(tma_store),
                   decltype(tiled_mma), SmemCopyAtomC>);

  CUDABOX_CUDA_CALL(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, kernel_ptr, prob_shape, cta_tiler, cluster_shape, A, tmaA, B,
      tmaB, C, tma_store, tiled_mma, smem_copy_atom);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorLaunchFailure;
  }

  return cudaSuccess;
}

} // namespace cudabox::gemm::sm90_pipelined_tma_mma
