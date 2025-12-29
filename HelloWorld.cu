#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
	printf("Hello from GPU! block=%d thread=%d\n", blockIdx.x, threadIdx.x);
}

int main() {
    	// launch
    	hello_kernel<<<1, 3>>>();
    

    	// wait for GPU
    	cudaDeviceSynchronize();

	return 0;
}
