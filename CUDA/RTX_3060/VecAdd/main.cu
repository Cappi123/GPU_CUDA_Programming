#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void vecAddKernel(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// -------------------- CPU-side helpers --------------------

__host__ float* host_alloc(int n)
{
    return (float*)malloc((size_t)n * sizeof(float));
}

__host__ void host_initVec(float* a, float* b, int n)
{
    for (int i = 0; i < n; i++) {
        a[i] = 1.0f + 0.001f * (float)(i % 1000);
        b[i] = 2.0f + 0.001f * (float)((i * 7) % 1000);
    }
}

__host__ void cpu_vecAdd(const float* a, const float* b, float* c, int n)
{
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

__host__ double cpu_timing(const float* a, const float* b, float* c, int n, int iters)
{
    // warmup
    cpu_vecAdd(a, b, c, n);

    clock_t t0 = clock();
    for (int rep = 0; rep < iters; rep++) {
        cpu_vecAdd(a, b, c, n);
    }
    clock_t t1 = clock();

    double total_s = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
    return total_s / (double)iters; // seconds per iteration
}

__host__ double gpu_timing(const float* h_a, const float* h_b, float* h_c,
                           int n, int iters, int block)
{
    size_t bytes = (size_t)n * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int grid = (n + block - 1) / block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    vecAddKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // kernel-only timing
    cudaEventRecord(start);
    for (int rep = 0; rep < iters; rep++) {
        vecAddKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, start, stop);
    double s_per_iter = ((double)ms_total / 1000.0) / (double)iters;

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return s_per_iter;
}

__host__ float host_maxAbsError(const float* ref, const float* out, int n)
{
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(out[i] - ref[i]);
        if (e > m) m = e;
    }
    return m;
}

__host__ double gflops_vecAdd(int n, double seconds_per_iter)
{
    // vec add = 1 FLOP per element
    return ((double)n / seconds_per_iter) / 1e9;
}

// -------------------- main --------------------

int main()
{
    // Hardcoded for simplicity (change these)
    int n     = 1 << 26;   // ~67 million
    int iters = 50;
    int block = 256;

    printf("N=%d, iters=%d, block=%d\n", n, iters, block);

    float* h_a = host_alloc(n);
    float* h_b = host_alloc(n);
    float* h_c_cpu = host_alloc(n);
    float* h_c_gpu = host_alloc(n);

    host_initVec(h_a, h_b, n);

    double cpu_s = cpu_timing(h_a, h_b, h_c_cpu, n, iters);
    double gpu_s = gpu_timing(h_a, h_b, h_c_gpu, n, iters, block);

    float err = host_maxAbsError(h_c_cpu, h_c_gpu, n);

    printf("\nPer-iteration:\n");
    printf("CPU: %.3f ms | %.3f GFLOPS\n", cpu_s * 1000.0, gflops_vecAdd(n, cpu_s));
    printf("GPU: %.3f ms | %.3f GFLOPS (kernel-only)\n", gpu_s * 1000.0, gflops_vecAdd(n, gpu_s));
    printf("Max abs error: %g\n", err);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    return 0;
}

