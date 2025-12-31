#include <cuda_runtime.h>
#include "inc.h"

__global__
void GPU_optMatMulKernel(const float* A, const float* B, float* C,
                         int m, int n, int k)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (k + TILE - 1) / TILE;

    for (int tile = 0; tile < numTiles; tile++) {
        int Acol = tile * TILE + threadIdx.x;
        int Brow = tile * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < m && Acol < k) ? A[row * k + Acol] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (Brow < k && col < n) ? B[Brow * n + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = sum;
}

void GPU_optMatMul(const float* dA, const float* dB, float* dC,
                   int m, int n, int k)
{
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE,
              (m + TILE - 1) / TILE);

    GPU_optMatMulKernel<<<grid, block>>>(dA, dB, dC, m, n, k);
}
