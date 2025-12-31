// Warmup + repeats average (hot) timing using CUDA events.

#include <cuda_runtime.h>

float timeGPU_hot(void (*launch)(const float*, const float*, float*, int, int, int),
                  const float* dA, const float* dB, float* dC,
                  int m, int n, int k,
                  int warmup, int repeats)
{
    cudaEvent_t start, stop;
    float ms = 0.0f;

    for (int i = 0; i < warmup; i++)
        launch(dA, dB, dC, m, n, k);
    cudaDeviceSynchronize();

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeats; i++)
        launch(dA, dB, dC, m, n, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / repeats;
}
