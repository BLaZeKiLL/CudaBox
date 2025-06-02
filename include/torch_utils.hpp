#pragma once

#include <torch/torch.h>

#define TORCH_TENSOR_CHECK(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
