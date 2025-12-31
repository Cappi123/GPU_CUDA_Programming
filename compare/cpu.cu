#include "inc.h"

void CPU_matMul(const float* A, const float* B, float* C,
                int m, int n, int k)
{
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int t = 0; t < k; t++) {
                sum += A[row * k + t] * B[t * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}
