#pragma once

#include <math.h>

// -------------------- compile-time constants --------------------
#define M 1024
#define N 1024
#define K 1024

#define TILE 16

// -------------------- helpers --------------------
float rand01(void);
void initMat(float* X, int rows, int cols);
void zeroMat(float* X, int rows, int cols);
double gflops(int m, int n, int k, double ms);

// Generic GPU timer for a matmul launcher (naive or opt)
float timeGPU(void (*launch)(const float*, const float*, float*, int, int, int),
              const float* dA, const float* dB, float* dC,
              int m, int n, int k);

// -------------------- compute APIs --------------------
void CPU_matMul(const float* A, const float* B, float* C,
                int m, int n, int k);

void GPU_naiveMatMul(const float* dA, const float* dB, float* dC,
                     int m, int n, int k);

void GPU_optMatMul(const float* dA, const float* dB, float* dC,
                   int m, int n, int k);
