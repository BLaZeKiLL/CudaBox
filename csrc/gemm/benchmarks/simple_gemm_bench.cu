#include <cstddef>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <gtest/gtest.h>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>

// Forward-declare the templated launcher. The .cu file (simple_gemm.cu)
// instantiates float / c10::Half / c10::BFloat16 via AT_DISPATCH; the linker
// resolves these symbols when the bench is linked against gemm_ops_static.
namespace cudabox::gemm {
template <typename T>
cudaError_t simple_gemm_launch(T const *A, T const *B, T *C, unsigned int M,
                               unsigned int N, unsigned int K,
                               cudaStream_t stream = 0);
}

// Element types we benchmark across. Order here is the order rows appear in
// nvbench output. c10::Half and c10::BFloat16 match what AT_DISPATCH_*_AND2
// instantiates on the kernel side.
using SimpleGemmElements = nvbench::type_list<float, c10::Half, c10::BFloat16>;

template <typename Element>
void simple_gemm_bench(nvbench::state &state, nvbench::type_list<Element>) {
  unsigned int M = static_cast<unsigned int>(state.get_int64("M"));
  unsigned int N = static_cast<unsigned int>(state.get_int64("N"));
  unsigned int K = static_cast<unsigned int>(state.get_int64("K"));

  std::size_t MK = std::size_t{M} * K;
  std::size_t KN = std::size_t{K} * N;
  std::size_t MN = std::size_t{M} * N;

  // each thread in M * N output, processes K elements
  std::size_t MNK = std::size_t{M} * N * K;

  // Allocate input data with non-zero values so kernels don't degenerate.
  thrust::device_vector<Element> A(MK, Element{2.0f});
  thrust::device_vector<Element> B(KN, Element{5.0f});
  thrust::device_vector<Element> C(MN, Element{0.0f});

  // Provide throughput information:
  state.add_element_count(MNK * 2, "FMA-flops");
  state.add_global_memory_reads<Element>(MK + KN, "input-elems");
  state.add_global_memory_writes<Element>(MN, "output-elems");

  state.exec([&](nvbench::launch &launch) {
    cudaError_t status = cudabox::gemm::simple_gemm_launch<Element>(
        thrust::raw_pointer_cast(A.data()), thrust::raw_pointer_cast(B.data()),
        thrust::raw_pointer_cast(C.data()), M, N, K, launch.get_stream());

    EXPECT_EQ(status, cudaSuccess);
  });
}

NVBENCH_BENCH_TYPES(simple_gemm_bench, NVBENCH_TYPE_AXES(SimpleGemmElements))
    .set_type_axes_names({"dtype"})
    .add_int64_power_of_two_axis("M", nvbench::range(6, 10))
    .add_int64_power_of_two_axis("N", nvbench::range(6, 10))
    .add_int64_power_of_two_axis("K", nvbench::range(10, 14));
