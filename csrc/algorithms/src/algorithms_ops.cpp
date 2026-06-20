#if defined(Py_LIMITED_API)
#include "cudabox_ops.hpp"
#include "python_utils.hpp"

TORCH_LIBRARY_FRAGMENT(cudabox, m) {
  m.def("histogram(Tensor tensor, int num_bins) -> Tensor");
  m.impl("histogram", torch::kCUDA, &cudabox::algorithms::histogram);
}

REGISTER_EXTENSION(algorithms_ops)
#endif
