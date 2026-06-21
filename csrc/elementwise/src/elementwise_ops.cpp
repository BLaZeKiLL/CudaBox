#if defined(Py_LIMITED_API)
#include "cudabox_ops.hpp"
#include "python_utils.hpp"

TORCH_LIBRARY_FRAGMENT(cudabox, m) {
  m.def("softmax(Tensor tensor) -> Tensor");
  m.impl("softmax", torch::kCUDA, &cudabox::elementwise::softmax);

  m.def("online_softmax(Tensor tensor) -> Tensor");
  m.impl("online_softmax", torch::kCUDA, &cudabox::elementwise::online_softmax);

  m.def("rmsnorm(Tensor tensor, Tensor gamma, float eps) -> Tensor");
  m.impl("rmsnorm", torch::kCUDA, &cudabox::elementwise::rmsnorm);
}

REGISTER_EXTENSION(elementwise_ops)
#endif
