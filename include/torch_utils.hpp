#pragma once

#include <torch/torch.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#define TORCH_TENSOR_CHECK(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
