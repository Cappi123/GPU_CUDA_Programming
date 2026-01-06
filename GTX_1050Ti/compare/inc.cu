#include <stdlib.h>
#include <cuda_runtime.h>

#include "inc.h"

// -------------------- helper implementations --------------------

float rand01(void)
{
    return (float)(rand() % 1000) / 1000.0f;
}

void initMat(float* X, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
        X[i] = rand01();
}

void zeroMat(float* X, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
        X[i] = 0.0f;
}

double gflops(int m, int n, int k, double ms)
{
    // ~2*m*n*k FLOPs for GEMM
    double flops = 2.0 * (double)m * (double)n * (double)k;
    double s = ms / 1000.0;
    return (flops / s) / 1e9;
}

// -------------------- GPU timing --------------------

float timeGPU(void (*launch)(const float*, const float*, float*, int, int, int),
              const float* dA, const float* dB, float* dC,
              int m, int n, int k)
{
    cudaEvent_t start, stop;
    float ms = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch(dA, dB, dC, m, n, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}
