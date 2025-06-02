#include "cudabox_ops.hpp"
#include "python_utils.hpp"

TORCH_LIBRARY_FRAGMENT(cudabox, m) {
    m.def("sgemm(Tensor mat_a, Tensor mat_b) -> Tensor");
    m.impl("sgemm", torch::kCUDA, &cudabox::gemm::sgemm);
}

REGISTER_EXTENSION(gemm_ops)
