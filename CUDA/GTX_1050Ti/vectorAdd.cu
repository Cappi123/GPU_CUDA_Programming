#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 5

__global__
void kernelAdd(float* v1, float* v2, float* res)
{
	int i = threadIdx.x;
	res[i] = v1[i] + v2[i];	
}


__host__
void initVec(float* v)
{
	for(int i = 0; i < SIZE; i++){
		v[i] = i;
	}	
}

__host__
void printVec(float* v)
{
	for(int i = 0; i < SIZE; i++){
		printf("v[%d] = %.1f \n",i, v[i]);
	}	
}

int main()
{
	float h_vA[SIZE], h_vB[SIZE], h_res[SIZE];
	float *d_vA = nullptr, *d_vB = nullptr, *d_res = nullptr;

	initVec(h_vA);
	initVec(h_vB);

	//Allocate Device Memory
	cudaMalloc(&d_vA, SIZE*sizeof(float));
	cudaMalloc(&d_vB, SIZE*sizeof(float));
	cudaMalloc(&d_res, SIZE*sizeof(float));

	//Copy Host -> Device
	cudaMemcpy(d_vA, h_vA, SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vB, h_vB, SIZE*sizeof(float), cudaMemcpyHostToDevice);
	
	//Launch Kernel
	kernelAdd<<<1,SIZE>>>(d_vA, d_vB, d_res);
	
	//Wait for GPU to finish
	cudaDeviceSynchronize();

	//Copy Device -> Host
	cudaMemcpy(h_res, d_res, SIZE*sizeof(float), cudaMemcpyDeviceToHost);

	//Clean-up
	cudaFree(d_vA);
	cudaFree(d_vB);
	cudaFree(d_res);

	//Print Result
	printf("h_res :\n");
	printVec(h_res);
	
	return 0;

}

