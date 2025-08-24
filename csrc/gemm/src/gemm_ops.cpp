#if defined(Py_LIMITED_API)
#include "cudabox_ops.hpp"
#include "python_utils.hpp"

TORCH_LIBRARY_FRAGMENT(cudabox, m) {
  // Simple Gemm
  m.def("simple_gemm(Tensor mat_a, Tensor mat_b) -> Tensor");
  m.impl("simple_gemm", torch::kCUDA, &cudabox::gemm::simple_gemm);

  // Tilled Gemm
  m.def("tiled_gemm(Tensor mat_a, Tensor mat_b) -> Tensor");
  m.impl("tiled_gemm", torch::kCUDA, &cudabox::gemm::tiled_gemm);
}

REGISTER_EXTENSION(gemm_ops)
#endif
