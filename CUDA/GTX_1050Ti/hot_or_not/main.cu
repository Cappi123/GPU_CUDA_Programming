// OPT matmul benchmark in two modes: COLD (single) vs HOT (warm+avg)

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define M 1024
#define N 1024
#define K 1024
#define TILE 16

#define WARMUP 2
#define REPEATS 10

float timeGPU_cold(void (*launch)(const float*, const float*, float*, int, int, int),
                   const float* dA, const float* dB, float* dC,
                   int m, int n, int k);

float timeGPU_hot(void (*launch)(const float*, const float*, float*, int, int, int),
                  const float* dA, const float* dB, float* dC,
                  int m, int n, int k,
                  int warmup, int repeats);

// helpers
static float rand01(void) { return (float)(rand() % 1000) / 1000.0f; }

static void initMat(float* X, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++) X[i] = rand01();
}

static double gflops(int m, int n, int k, double ms)
{
    double flops = 2.0 * (double)m * (double)n * (double)k;
    return (flops / (ms / 1000.0)) / 1e9;
}

// optimized kernel
__global__ void optMatMulKernel(const float* A, const float* B, float* C,
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
        for (int i = 0; i < TILE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = sum;
}

// launcher (function pointer target)
void GPU_optMatMul(const float* dA, const float* dB, float* dC,
                   int m, int n, int k)
{
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    optMatMulKernel<<<grid, block>>>(dA, dB, dC, m, n, k);
}

int main(void)
{
    srand(0);

    printf("Opt MatMul: C[%d x %d] = A[%d x %d] * B[%d x %d]\n",
           M, N, M, K, K, N);
    printf("TILE=%d | WARMUP=%d | REPEATS=%d\n\n", TILE, WARMUP, REPEATS);

    static float hA[M * K];
    static float hB[K * N];

    initMat(hA, M, K);
    initMat(hB, K, N);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, (size_t)M * (size_t)K * sizeof(float));
    cudaMalloc(&dB, (size_t)K * (size_t)N * sizeof(float));
    cudaMalloc(&dC, (size_t)M * (size_t)N * sizeof(float));

    cudaMemcpy(dA, hA, (size_t)M * (size_t)K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, (size_t)K * (size_t)N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(dC, 0, (size_t)M * (size_t)N * sizeof(float));
    float cold_ms = timeGPU_cold(GPU_optMatMul, dA, dB, dC, M, N, K);
    printf("[COLD] %.3f ms | %.2f GFLOP/s\n", cold_ms, gflops(M, N, K, cold_ms));

    cudaMemset(dC, 0, (size_t)M * (size_t)N * sizeof(float));
    float hot_ms = timeGPU_hot(GPU_optMatMul, dA, dB, dC, M, N, K, WARMUP, REPEATS);
    printf("[HOT ] %.3f ms | %.2f GFLOP/s\n", hot_ms, gflops(M, N, K, hot_ms));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
