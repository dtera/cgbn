//
// Created by root on 4/3/24.
//

#include "demo.cuh"

// 两个向量加法kernel，grid和block均为一维
__global__ void kernel_add(float *x, float *y, float *z, int n) {
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        z[i] = x[i] + y[i];
    }
}

__global__ void kernel_matrix_add(float **A, float **B, float **C, int M, int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        C[i][j] = A[i][j] + B[i][j];
    }
}


__global__ void kernel_histograms_add(float *x, float **y, int m, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        x[i] = x[i] + y[i][j];
    }
}