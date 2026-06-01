#include <cstddef>

#include <gtest/gtest.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

// kernel.cuh + traits.cuh are heavy (drag in all of CUTLASS), but `gemm_host`
// is a header-only function template with a default Traits parameter
// (`Traits = gemm_traits<TA>`). We need the full definition visible to
// instantiate it from this TU. Compile time of this file is dominated by
// CUTLASS template expansion; not worth optimizing.
#include "gemm/sm90_pipelined_tma_mma_gemm/kernel.cuh"

// SM90 WGMMA only supports fp16 and bf16 in this kernel (TF32/fp32 would need
// a different MMA atom and SmemCopyAtomC — see traits.cuh).
using Sm90GemmElements = nvbench::type_list<cute::half_t, cute::bfloat16_t>;

template <typename Element>
void sm90_pipelined_tma_mma_gemm_bench(nvbench::state &state,
                                       nvbench::type_list<Element>) {
  unsigned int M = static_cast<unsigned int>(state.get_int64("M"));
  unsigned int N = static_cast<unsigned int>(state.get_int64("N"));
  unsigned int K = static_cast<unsigned int>(state.get_int64("K"));

  std::size_t MK = std::size_t{M} * K;
  std::size_t NK = std::size_t{N} * K;
  std::size_t MN = std::size_t{M} * N;
  std::size_t MNK = std::size_t{M} * N * K;

  // Inputs/outputs. A = (M, K), B = (N, K), C = (M, N) — kernel computes
  // C = A * B^T, so B is laid out as (N, K) row-major.
  thrust::device_vector<Element> A(MK, Element{2});
  thrust::device_vector<Element> B(NK, Element{5});
  thrust::device_vector<Element> C(MN, Element{0});

  // Throughput info: 2 FMA flops per inner-loop element + memory volume.
  state.add_element_count(MNK * 2, "FMA-flops");
  state.add_global_memory_reads<Element>(MK + NK, "input-elems");
  state.add_global_memory_writes<Element>(MN, "output-elems");

  // Row-major strides.
  unsigned int ldA = K;
  unsigned int ldB = K;
  unsigned int ldC = N;

  state.exec([&](nvbench::launch &launch) {
    cudaError_t status =
        cudabox::gemm::sm90_pipelined_tma_mma::gemm_host<Element, Element,
                                                         Element>(
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
            thrust::raw_pointer_cast(A.data()), ldA,
            thrust::raw_pointer_cast(B.data()), ldB,
            thrust::raw_pointer_cast(C.data()), ldC, launch.get_stream());
    EXPECT_EQ(status, cudaSuccess);
  });
}

// Tile size of the kernel is bM=256, bN=192, bK=128 (from gemm_traits). Skip
// shapes that wouldn't fully utilize a single CTA tile so the benchmark is
// meaningful. Start at 2^8 = 256 along M and N (one CTA tile) and 2^7 = 128
// along K (one K-tile). Extend up to 4096 M/N and 16384 K.
NVBENCH_BENCH_TYPES(sm90_pipelined_tma_mma_gemm_bench,
                    NVBENCH_TYPE_AXES(Sm90GemmElements))
    .set_type_axes_names({"dtype"})
    .add_int64_power_of_two_axis("M", nvbench::range(8, 12))  // 256..4096
    .add_int64_power_of_two_axis("N", nvbench::range(8, 12))  // 256..4096
    .add_int64_power_of_two_axis("K", nvbench::range(7, 14)); // 128..16384
