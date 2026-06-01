#include "cuda_utils.cuh"
#include "cudabox_ops.hpp"
#include "logger.hpp"
#include "torch_utils.hpp"

#include <ATen/Dispatch.h>

namespace cudabox::gemm {

// Block Dim == Tile Size
template <typename T, int TILE_SIZE>
__global__ void tiled_gemm_kernel(T const *__restrict__ A,
                                  T const *__restrict__ B, T *__restrict__ C,
                                  unsigned int M, unsigned int N,
                                  unsigned int K) {
  // SMEM tiles stored in element type T to minimize SMEM usage.
  __shared__ T M_tile[TILE_SIZE][TILE_SIZE];
  __shared__ T N_tile[TILE_SIZE][TILE_SIZE];

  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;

  // Each thread will compute C[row][col]
  unsigned int row = by * TILE_SIZE + ty;
  unsigned int col = bx * TILE_SIZE + tx;

  // Accumulate in fp32 regardless of input dtype — see simple_gemm.cu for
  // the precision rationale.
  float accumulator = 0.0f;

  // 0 -> ceil(K / TILE_SIZE) tiles would be iterated upon to cover
  for (unsigned int tile_idx = 0;
       tile_idx < cudabox::utils::ceil_div(K, TILE_SIZE); ++tile_idx) {
    // load phase

    // load tile of M
    if ((row < M) && (tile_idx * TILE_SIZE + tx < K)) {
      M_tile[ty][tx] = A[row * K + tile_idx * TILE_SIZE + tx];
    } else {
      M_tile[ty][tx] = T{0};
    }

    // Tile of N
    if ((tile_idx * TILE_SIZE + ty < K) && (col < N)) {
      N_tile[ty][tx] = B[(tile_idx * TILE_SIZE + ty) * N + col];
    } else {
      N_tile[ty][tx] = T{0};
    }

    __syncthreads();
    // compute phase

    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
      accumulator +=
          static_cast<float>(M_tile[ty][k]) * static_cast<float>(N_tile[k][tx]);
    }

    __syncthreads();
  }

  if ((row < M) && (col < N)) {
    C[row * N + col] = static_cast<T>(accumulator);
  }
}

template <typename T>
cudaError_t tiled_gemm_launch(T const *A, T const *B, T *C, unsigned int M,
                              unsigned int N, unsigned int K,
                              cudaStream_t stream = 0) {
  constexpr unsigned int tile_size = 16;

  dim3 nblks(cudabox::utils::ceil_div(N, tile_size),
             cudabox::utils::ceil_div(M, tile_size), 1);
  dim3 nthrs(tile_size, tile_size, 1);

  constexpr size_t smem_size = 2 * tile_size * tile_size * sizeof(T);

  cudaLaunchConfig_t config{};
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.stream = stream;
  config.dynamicSmemBytes = smem_size;

  auto kernel = tiled_gemm_kernel<T, tile_size>;

  CUDABOX_LOG_DEBUG("Dispatching tiled gemm M={}, N={}, K={}, tile_size={}", M,
                    N, K, tile_size);
  CUDABOX_CUDA_CALL(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  CUDABOX_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, A, B, C, M, N, K));

  return cudaSuccess;
}

// Block Dim == Tile Size
torch::Tensor tiled_gemm(const torch::Tensor &mat_a,
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

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, mat_a.scalar_type(),
      "tiled_gemm", [&] {
        cudaError_t status = tiled_gemm_launch<scalar_t>(
            mat_a.data_ptr<scalar_t>(), mat_b.data_ptr<scalar_t>(),
            mat_c.data_ptr<scalar_t>(), M, N, K, stream);
        TORCH_CHECK(status == cudaSuccess, "tiled_gemm failed");
      });

  return mat_c;
}

} // namespace cudabox::gemm
