
/*

naive : 
Only global memory is used, not made for optimisation.

BLOCK:
Threads are grouped into 16x16 blocks.
Each block computes a small tile of the output matrix.
Each thread inside the block computes one C[row][col].

GRID:
The grid is sized to cover the entire output matrix.
Ceiling division is used since M and N may not be multiples of the block size.
Extra threads may be launched and exit early via a guard.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 3
#define K 2
#define N 3

#define BLOCK_X 16
#define BLOCK_Y 16

__global__
void matMul(const float* A, const float* B, float* C )
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	float sum = 0.0;
	
	// guard condition
	// returns nothing for un-used threads
	if( row >= M || col >= N ){return;}

	for(int k = 0; k < K; k++)
	{
		 sum += A[ row*K + k ]*B[ k*N + col];
	}
	C[ row*N + col ] = sum;
}


__host__
void initMatrix(float* Mat, int size )
{
	 for(int i =0; i<size; i++)
	 {
		Mat[i] = (float)(rand()%100) / 100.0f;
	 }
}

__host__
void printMatrix(float* Mat, int rows, int cols)
{
	for(int r =0; r<rows;r++)
	{
		for(int c =0; c<cols;c++)
		{
			printf("%6.2f ", Mat[r*cols + c]);
		}
		printf("\n");
	}
	printf("\n");
}

int main()
{
	srand(12);

	float h_MA[M*K], h_MB[K*N], h_MC[M*N];
	float* d_MA, *d_MB, *d_MC;

	initMatrix(h_MA,M*K);
	initMatrix(h_MB,K*N);

	printMatrix( h_MA, M, K);
	printMatrix( h_MB, K, N);


	//Allocate Device Memory
	cudaMalloc(&d_MA,M*K*sizeof(float));
	cudaMalloc(&d_MB,K*N*sizeof(float));
	cudaMalloc(&d_MC,M*N*sizeof(float));

	// Copy Memory Host -> Device
	cudaMemcpy(d_MA, h_MA, M*K*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_MB, h_MB, K*N*sizeof(float), cudaMemcpyHostToDevice);


	// Configure Thread & Block Settings
	dim3 block(BLOCK_X, BLOCK_Y);
	dim3 grid(
		(N + BLOCK_X -1 )/BLOCK_X,
		(M + BLOCK_Y -1 )/BLOCK_Y
	);
	
	//Launch Kernel
	matMul<<<grid, block>>>(d_MA,d_MB,d_MC);
	cudaDeviceSynchronize();

	// Copy Memory Device -> Host
	cudaMemcpy( h_MC, d_MC, M*N*sizeof(float), cudaMemcpyDeviceToHost);
	printMatrix(h_MC, M, N );

	cudaFree(d_MA);
	cudaFree(d_MB);
	cudaFree(d_MC);

	return 0;
}

