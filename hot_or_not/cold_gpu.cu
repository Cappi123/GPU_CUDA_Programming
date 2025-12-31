// Single-shot (cold) timing using CUDA events.

#include <cuda_runtime.h>

float timeGPU_cold(void (*launch)(const float*, const float*, float*, int, int, int),
                   const float* dA, const float* dB, float* dC,
                   int m, int n, int k)
{
    cudaEvent_t start, stop;
    float ms = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch(dA, dB, dC, m, n, k);     // ONE launch
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}
