#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_from_gpu() {
    printf("Hello from GPU! block=%d thread=%d\n", blockIdx.x, threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");

    hello_from_gpu<<<1, 4>>>();   
    cudaDeviceSynchronize();     
    return 0;
}
