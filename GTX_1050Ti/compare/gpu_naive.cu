#include <cuda_runtime.h>
#include "inc.h"

__global__
void GPU_naiveMatMulKernel(const float* A, const float* B, float* C,
                           int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int t = 0; t < k; t++) {
            sum += A[row * k + t] * B[t * n + col];
        }
        C[row * n + col] = sum;
    }
}

void GPU_naiveMatMul(const float* dA, const float* dB, float* dC,
                     int m, int n, int k)
{
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE,
              (m + TILE - 1) / TILE);

    GPU_naiveMatMulKernel<<<grid, block>>>(dA, dB, dC, m, n, k);
}
