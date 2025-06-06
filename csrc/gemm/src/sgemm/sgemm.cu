#include "cudabox_ops.hpp"
#include "torch_utils.hpp"
#include "cuda_utils.cuh"

namespace cudabox::gemm
{
    template <typename T>
    __global__ void sgemm_kernel(T* __restrict__ A, T* __restrict__ B, T* __restrict__ C, unsigned int M, unsigned int N, unsigned int K)
    {
        // Index calculation for the grid
        unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
        // Boundary check
        if ((row >= M) || (col >= N)) return;

        T accumulator = {0};

        for (int i = 0; i < K; i++) {
            accumulator += A[row * K + i] * B[i * N + col];
        }

        C[row * N + col] = accumulator;
    }

    torch::Tensor sgemm(const torch::Tensor &mat_a, const torch::Tensor &mat_b)
    {
        TORCH_TENSOR_CHECK(mat_a);
        TORCH_TENSOR_CHECK(mat_b);

        TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "Tensors dimmensions are not compatible for matmul");

        unsigned int M = mat_a.size(0);
        unsigned int N = mat_b.size(1);
        unsigned int K = mat_a.size(1);

        torch::Tensor mat_c = torch::ones({M, N}, torch::dtype(mat_a.dtype()).device(torch::kCUDA));

        auto device = mat_a.device();

        const c10::cuda::OptionalCUDAGuard device_guard(device);
        const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        dim3 nblks(ceil_div(N, 32), ceil_div(M, 32), 1);
        dim3 nthrs(32, 32, 1);

        sgemm_kernel<<<nblks, nthrs, 0, stream>>>(
            mat_a.data_ptr<float>(),
            mat_b.data_ptr<float>(),
            mat_c.data_ptr<float>(),
            M, N, K
        );

        cudaDeviceSynchronize();

        return mat_c;
    }
} // namespace cudabox::gemm
