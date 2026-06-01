#include <cstddef>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <gtest/gtest.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

namespace cudabox::gemm {
template <typename T>
cudaError_t tiled_gemm_launch(T const *A, T const *B, T *C, unsigned int M,
                              unsigned int N, unsigned int K,
                              cudaStream_t stream = 0);
}

using TiledGemmElements = nvbench::type_list<float, c10::Half, c10::BFloat16>;

template <typename Element>
void tiled_gemm_bench(nvbench::state &state, nvbench::type_list<Element>) {
  unsigned int M = static_cast<unsigned int>(state.get_int64("M"));
  unsigned int N = static_cast<unsigned int>(state.get_int64("N"));
  unsigned int K = static_cast<unsigned int>(state.get_int64("K"));

  std::size_t MK = std::size_t{M} * K;
  std::size_t KN = std::size_t{K} * N;
  std::size_t MN = std::size_t{M} * N;
  std::size_t MNK = std::size_t{M} * N * K;

  thrust::device_vector<Element> A(MK, Element{2.0f});
  thrust::device_vector<Element> B(KN, Element{5.0f});
  thrust::device_vector<Element> C(MN, Element{0.0f});

  state.add_element_count(MNK * 2, "FMA-flops");
  state.add_global_memory_reads<Element>(MK + KN, "input-elems");
  state.add_global_memory_writes<Element>(MN, "output-elems");

  state.exec([&](nvbench::launch &launch) {
    cudaError_t status = cudabox::gemm::tiled_gemm_launch<Element>(
        thrust::raw_pointer_cast(A.data()), thrust::raw_pointer_cast(B.data()),
        thrust::raw_pointer_cast(C.data()), M, N, K, launch.get_stream());

    EXPECT_EQ(status, cudaSuccess);
  });
}

NVBENCH_BENCH_TYPES(tiled_gemm_bench, NVBENCH_TYPE_AXES(TiledGemmElements))
    .set_type_axes_names({"dtype"})
    .add_int64_power_of_two_axis("M", nvbench::range(6, 10))
    .add_int64_power_of_two_axis("N", nvbench::range(6, 10))
    .add_int64_power_of_two_axis("K", nvbench::range(10, 14));
