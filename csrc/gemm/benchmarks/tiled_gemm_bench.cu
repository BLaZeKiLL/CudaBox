#include <cstddef>

#include <gtest/gtest.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

namespace cudabox::gemm {
template <typename T>
cudaError_t tiled_gemm_launch(T *A, T *B, T *C, unsigned int M, unsigned int N,
                              unsigned int K, cudaStream_t stream = 0);
}

void tiled_gemm_bench(nvbench::state &state) {
  unsigned int M = static_cast<unsigned int>(state.get_int64("M"));
  unsigned int N = static_cast<unsigned int>(state.get_int64("N"));
  unsigned int K = static_cast<unsigned int>(state.get_int64("K"));

  std::size_t MK = M * K;
  std::size_t KN = K * N;
  std::size_t MN = M * N;

  // each thread in M * N output, process 2 * K elements
  std::size_t MNK = M * N * K;

  // Allocate input data:
  thrust::device_vector<float> A(MK, 2);
  thrust::device_vector<float> B(KN, 5);
  thrust::device_vector<float> C(MN, 0);

  // Provide throughput information:
  state.add_element_count(MNK, "elements");
  state.add_global_memory_reads<float>(MK + KN, "reads");
  state.add_global_memory_writes<float>(MN, "writes");

  state.exec([&](nvbench::launch &launch) {
    cudaError_t status = cudabox::gemm::tiled_gemm_launch(
        thrust::raw_pointer_cast(A.data()), thrust::raw_pointer_cast(B.data()),
        thrust::raw_pointer_cast(C.data()), M, N, K, launch.get_stream());

    EXPECT_EQ(status, cudaSuccess);
  });
}

NVBENCH_BENCH(tiled_gemm_bench)
    .add_int64_power_of_two_axis("M", nvbench::range(6, 10))
    .add_int64_power_of_two_axis("N", nvbench::range(6, 10))
    .add_int64_power_of_two_axis("K", nvbench::range(10, 14));
