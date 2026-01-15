#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#include "inc.h"

int main(void)
{
    srand(0);

    printf("MatMul: C[%d x %d] = A[%d x %d] * B[%d x %d]\n",
           M, N, M, K, K, N);
    printf("TILE=%d\n\n", TILE);

    // Host matrices (static to avoid stack overflow)
    static float hA[M * K];
    static float hB[K * N];
    static float hC_cpu[M * N];
    static float hC_naive[M * N];
    static float hC_opt[M * N];

    initMat(hA, M, K);
    initMat(hB, K, N);
    zeroMat(hC_cpu, M, N);
    zeroMat(hC_naive, M, N);
    zeroMat(hC_opt, M, N);

    // ---------------- CPU ----------------
    clock_t t0 = clock();
    CPU_matMul(hA, hB, hC_cpu, M, N, K);
    clock_t t1 = clock();

    double cpu_ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
    printf("[CPU]       %.3f ms | %.2f GFLOP/s\n",
           cpu_ms, gflops(M, N, K, cpu_ms));

    // ---------------- GPU ----------------

    //set-up
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(hA));
    cudaMalloc(&dB, sizeof(hB));
    cudaMalloc(&dC, sizeof(hC_cpu));

    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);

    // GPU naive 
    cudaMemset(dC, 0, sizeof(hC_naive));
    float naive_ms = timeGPU(GPU_naiveMatMul, dA, dB, dC, M, N, K);
    cudaMemcpy(hC_naive, dC, sizeof(hC_naive), cudaMemcpyDeviceToHost);

    printf("[GPU naive] %.3f ms | %.2f GFLOP/s\n",
   	naive_ms, gflops(M, N, K, naive_ms));

    // GPU optimized
    cudaMemset(dC, 0, sizeof(hC_opt));
    float opt_ms = timeGPU(GPU_optMatMul, dA, dB, dC, M, N, K);
    cudaMemcpy(hC_opt, dC, sizeof(hC_opt), cudaMemcpyDeviceToHost);

    printf("[GPU opt]   %.3f ms | %.2f GFLOP/s\n",
       opt_ms, gflops(M, N, K, opt_ms));


    //clean-up 
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
