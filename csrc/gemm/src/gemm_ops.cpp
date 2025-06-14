#if defined(Py_LIMITED_API)
#include "cudabox_ops.hpp"
#include "python_utils.hpp"

TORCH_LIBRARY_FRAGMENT(cudabox, m) {
  // Simple Gemm
  m.def("sgemm(Tensor mat_a, Tensor mat_b) -> Tensor");
  m.impl("sgemm", torch::kCUDA, &cudabox::gemm::sgemm);

  // Tilled Gemm
  m.def("tgemm(Tensor mat_a, Tensor mat_b) -> Tensor");
  m.impl("tgemm", torch::kCUDA, &cudabox::gemm::tgemm);
}

REGISTER_EXTENSION(gemm_ops)
#endif
