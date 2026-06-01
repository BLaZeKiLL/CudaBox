#include "cuda_utils.cuh"
#include "cudabox_ops.hpp"
#include "logger.hpp"
#include "torch_utils.hpp"

#include <ATen/Dispatch.h>

namespace cudabox::gemm {

template <typename T>
__global__ void simple_gemm_kernel(T const *__restrict__ A,
                                   T const *__restrict__ B, T *__restrict__ C,
                                   unsigned int M, unsigned int N,
                                   unsigned int K) {
  // Index calculation for the grid
  unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

  // Boundary check
  if ((row >= M) || (col >= N))
    return;

  // Accumulate in fp32 regardless of input dtype. For fp16/bf16 inputs this
  // avoids catastrophic precision loss from in-place accumulation at K elems.
  float accumulator = 0.0f;

  // All linear indexes follow the following pattern
  // linear_index = row_no * no_of_columns [i.e row width] + col_no
  for (int i = 0; i < K; i++) {
    accumulator +=
        static_cast<float>(A[row * K + i]) * static_cast<float>(B[i * N + col]);
  }

  C[row * N + col] = static_cast<T>(accumulator);
}

template <typename T>
cudaError_t simple_gemm_launch(T const *A, T const *B, T *C, unsigned int M,
                               unsigned int N, unsigned int K,
                               cudaStream_t stream = 0) {
  constexpr unsigned int tile_size = 32;

  dim3 nblks(cudabox::utils::ceil_div(N, tile_size),
             cudabox::utils::ceil_div(M, tile_size), 1);
  dim3 nthrs(tile_size, tile_size, 1);

  cudaLaunchConfig_t config{};
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.stream = stream;

  auto kernel = simple_gemm_kernel<T>;

  CUDABOX_LOG_DEBUG("Dispatching simple gemm M={}, N={}, K={}", M, N, K);
  CUDABOX_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, A, B, C, M, N, K));

  return cudaSuccess;
}

torch::Tensor simple_gemm(const torch::Tensor &mat_a,
                          const torch::Tensor &mat_b) {
  TORCH_TENSOR_CHECK(mat_a);
  TORCH_TENSOR_CHECK(mat_b);

  TORCH_CHECK(mat_a.size(1) == mat_b.size(0),
              "Tensors dimensions are not compatible for matmul");
  TORCH_CHECK(mat_a.scalar_type() == mat_b.scalar_type(),
              "A and B must share the same dtype. Got ", mat_a.scalar_type(),
              " vs ", mat_b.scalar_type());

  unsigned int M = mat_a.size(0);
  unsigned int N = mat_b.size(1);
  unsigned int K = mat_a.size(1);

  torch::Tensor mat_c =
      torch::empty({M, N}, torch::dtype(mat_a.dtype()).device(torch::kCUDA));

  auto device = mat_a.device();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // Dispatch on dtype. AT_DISPATCH_FLOATING_TYPES_AND2 covers
  // float, double, and the two named extras (Half, BFloat16). The lambda gets
  // a typedef `scalar_t` for the per-instantiation element type — for fp16/bf16
  // that's `at::Half` / `at::BFloat16`, which provide device-side operator
  // overloads needed by the kernel.
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, mat_a.scalar_type(),
      "simple_gemm", [&] {
        cudaError_t status = simple_gemm_launch<scalar_t>(
            mat_a.data_ptr<scalar_t>(), mat_b.data_ptr<scalar_t>(),
            mat_c.data_ptr<scalar_t>(), M, N, K, stream);
        TORCH_CHECK(status == cudaSuccess, "simple_gemm failed");
      });

  return mat_c;
}

} // namespace cudabox::gemm
