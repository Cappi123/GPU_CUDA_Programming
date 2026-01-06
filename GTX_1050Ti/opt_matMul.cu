#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 2
#define K 2
#define N 2

#define BLOCK_SIZE 16

__global__
void matMul( const float* A, const float* B, float* C)
{
	
	// Shared memory tiles
	// each block loads s small tile ( portion ) of matrix A and B
	__shared__
	float As[BLOCK_SIZE][BLOCK_SIZE];

	__shared__
	float Bs[BLOCK_SIZE][BLOCK_SIZE];

	int row = threadIdx.y + blockDim.y*blockIdx.y;
	int col = threadIdx.x + blockDim.x*blockIdx.x;

	// Guard condition. threads outside matrix do nothing
	if (row >= M) return;
    	if (col >= N) return;

	float sum = 0.0f;

	// Calculate the number of tiles needed
	int numTiles = ( K + BLOCK_SIZE - 1 ) / BLOCK_SIZE;

	// matrix multiplication done one tile at a time
	for( int t=0; t<numTiles; t++)
	{
		// global index of the thread
		int aCol = t*BLOCK_SIZE + threadIdx.x;
		int bRow = t*BLOCK_SIZE + threadIdx.y;

		//Load A tile elements into shared memory, if outside load 0
		As[threadIdx.y][threadIdx.x] = (aCol < K) ? A[row*K + aCol] : 0.0f;

		//Load B tile elements
		Bs[threadIdx.y][threadIdx.x] = (bRow < K) ? B[bRow*N + col] : 0.0f;
		
		// Wait until all threads finish loading, *race condition
		__syncthreads();

		// Compute partial dot prod for this tile
		for( int i=0; i<BLOCK_SIZE; i++ )
		{
			sum += As[threadIdx.y][i]*Bs[i][threadIdx.x];
		}
		
		// wait before loading the next tile
		__syncthreads();
	}
	
	//write results to matrix C
	C[ row*N + col ] = sum;

}

__host__
void initMatrix(float* mat, int size)
{
	for(int i=0; i<size; i++)
	{
		//0.0 - 9.9
		mat[i] = (float)( rand()%100) / 10.0;  
	}
}

__host__
void printMatrix(float* mat, int rows, int cols)
{
	for(int i = 0; i<rows; i++)
	{
		for(int j=0; j<cols; j++) 
		{
			printf( "%6.2f ", mat[i*cols + j] );
		}
		printf("\n");
	}
	printf("\n");
}

int main()
{
	srand(42);

    	// Host matrices (stack)
    	float h_MA[M*K], h_MB[K*N], h_MC[M*N];

    	initMatrix(h_MA, M * K);
    	initMatrix(h_MB, K * N);
  	
	printMatrix(h_MA, M, K);
	printMatrix(h_MB, K, N);

   	 // Device allocations
    	float *d_MA = NULL, *d_MB = NULL, *d_MC = NULL;

    	cudaMalloc((void**)&d_MA, M * K * sizeof(float));
    	cudaMalloc((void**)&d_MB, K * N * sizeof(float));
    	cudaMalloc((void**)&d_MC, M * N * sizeof(float));

    	// Copy Memory Host -> Device
    	cudaMemcpy(d_MA, h_MA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    	cudaMemcpy(d_MB, h_MB, K * N * sizeof(float), cudaMemcpyHostToDevice);

    	// Configure Thread & Block Settings
    	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    	dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    	// Run kernel
    	matMul<<<grid, block>>>(d_MA, d_MB, d_MC);

    	// Copy result Device -> Host
    	cudaMemcpy(h_MC, d_MC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    	// Print result
    	printMatrix(h_MC, M, N);   

    	// Cleanup
    	cudaFree(d_MA);
    	cudaFree(d_MB);
    	cudaFree(d_MC);

    	return 0;
}

