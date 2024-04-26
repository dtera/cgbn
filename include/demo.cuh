//
// Created by root on 4/3/24.
//

#pragma once

__global__ void kernel_add(float *x, float *y, float *z, int n);

__global__ void kernel_matrix_add(float **A, float **B, float **C, int M, int N);

__global__ void kernel_histograms_add(float *x, float **y, int m, int n);