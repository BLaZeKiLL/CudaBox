#include "cuda_utils.cuh"
#include "cudabox_ops.hpp"
#include "logger.hpp"
#include "torch_utils.hpp"

#include "gemm/sm90_pipelined_tma_mma_gemm/kernel.cuh"

namespace cudabox::gemm::sm90_pipelined_tma_mma {

namespace {

// Per-dtype launch wrapper. Reinterpret-casts torch's element-typed pointer
// (at::Half / at::BFloat16, both bit-identical to the cute:: counterpart) and
// hands off to the templated host launcher.
template <class CuteT, class TorchT>
cudaError_t launch_typed(unsigned int M, unsigned int N, unsigned int K,
                         const torch::Tensor &a, unsigned int ldA,
                         const torch::Tensor &b, unsigned int ldB,
                         torch::Tensor &c, unsigned int ldC,
                         cudaStream_t stream) {
  return gemm_host<CuteT, CuteT, CuteT>(
      M, N, K, reinterpret_cast<const CuteT *>(a.data_ptr<TorchT>()), ldA,
      reinterpret_cast<const CuteT *>(b.data_ptr<TorchT>()), ldB,
      reinterpret_cast<CuteT *>(c.data_ptr<TorchT>()), ldC, stream);
}

} // namespace

torch::Tensor gemm(const torch::Tensor &mat_a, const torch::Tensor &mat_b) {
  TORCH_TENSOR_CHECK(mat_a);
  TORCH_TENSOR_CHECK(mat_b);

  TORCH_CHECK(mat_a.size(1) == mat_b.size(1),
              "Tensors dimensions are not compatible for matmul");
  TORCH_CHECK(mat_a.scalar_type() == mat_b.scalar_type(),
              "A and B must have the same dtype. Got ", mat_a.scalar_type(),
              " vs ", mat_b.scalar_type());
  TORCH_CHECK(mat_a.is_contiguous() && mat_b.is_contiguous(),
              "A and B must be row-major contiguous");

  TORCH_CHECK((mat_a.size(1) * mat_a.element_size()) % 16 == 0,
              "K * sizeof(dtype) must be a multiple of 16 bytes for TMA load "
              "of A. Got K=",
              mat_a.size(1), " dtype=", mat_a.dtype());
  TORCH_CHECK((mat_b.size(1) * mat_b.element_size()) % 16 == 0,
              "K * sizeof(dtype) must be a multiple of 16 bytes for TMA load "
              "of B. Got K=",
              mat_b.size(1), " dtype=", mat_b.dtype());

  unsigned int M = mat_a.size(0);
  unsigned int N = mat_b.size(0);
  unsigned int K = mat_a.size(1);

  torch::Tensor mat_c =
      torch::empty({M, N}, torch::dtype(mat_a.dtype()).device(torch::kCUDA));

  TORCH_CHECK((mat_c.size(1) * mat_c.element_size()) % 16 == 0,
              "N * sizeof(dtype) must be a multiple of 16 bytes for TMA store "
              "of C. Got N=",
              mat_c.size(1), " dtype=", mat_c.dtype());

  unsigned int ldA = mat_a.stride(0);
  unsigned int ldB = mat_b.stride(0);
  unsigned int ldC = mat_c.stride(0);

  auto device = mat_a.device();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  switch (mat_a.scalar_type()) {
  case torch::kHalf:
    launch_typed<cute::half_t, at::Half>(M, N, K, mat_a, ldA, mat_b, ldB, mat_c,
                                         ldC, stream);
    break;
  case torch::kBFloat16:
    launch_typed<cute::bfloat16_t, at::BFloat16>(M, N, K, mat_a, ldA, mat_b,
                                                 ldB, mat_c, ldC, stream);
    break;
  default:
    TORCH_CHECK(false, "sm90_tma_mma_gemm: unsupported dtype ",
                mat_a.scalar_type(), ". Supported: float16, bfloat16.");
  }

  return mat_c;
}

} // namespace cudabox::gemm::sm90_pipelined_tma_mma
