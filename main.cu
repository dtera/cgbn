#include <iostream>
#include "demo.hpp"
#include "demo.cuh"
#include "stopwatch.hpp"

void show_cuda_info() {
    cudaDeviceProp devProp;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
}

void add_v1() {
    StopWatch sw;
    int N = 1 << 27, len = 100;
    int nBytes = N * sizeof(float);
    // 申请host内存
    float *x, *y, *z;
    x = (float *) malloc(nBytes);
    y = (float *) malloc(nBytes);
    z = (float *) malloc(nBytes);

    // 初始化数据
    for (int i = 0; i < N; ++i) {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    sw.Mark("add");
    for (int i = 0; i < len; i++) {
        add(x, y, z, N);
    }
    sw.PrintWithMills("add");

    // 申请device内存
    float *d_x, *d_y, *d_z;
    cudaMalloc((void **) &d_x, nBytes);
    cudaMalloc((void **) &d_y, nBytes);
    cudaMalloc((void **) &d_z, nBytes);

    sw.Mark("cuda_add_all");
    // 将host数据拷贝到device
    cudaMemcpy((void *) d_x, (void *) x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_y, (void *) y, nBytes, cudaMemcpyHostToDevice);
    // 定义kernel的执行配置
    dim3 blockSize(512);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    sw.Mark("cuda_add");
    for (int i = 0; i < len * 1; i++) {
        kernel_add<<< gridSize, blockSize >>>(d_x, d_y, d_z, N);
    }
    sw.PrintWithMills("cuda_add");

    // 将device得到的结果拷贝到host
    cudaMemcpy((void *) z, (void *) d_z, nBytes, cudaMemcpyDeviceToHost);
    sw.PrintWithMills("cuda_add_all");

    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放device内存
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    // 释放host内存
    free(x);
    free(y);
    free(z);
    std::cout << std::endl;
}

void add_v2() {
    StopWatch sw;
    int deviceId;
    cudaGetDevice(&deviceId);
    int N = 1 << 27, len = 100;
    int nBytes = N * sizeof(float);
    // 申请host内存
    float *x, *y, *z;
    x = (float *) malloc(nBytes);
    y = (float *) malloc(nBytes);
    z = (float *) malloc(nBytes);
    cudaMallocManaged(&x, nBytes);
    cudaMallocManaged(&y, nBytes);
    cudaMallocManaged(&z, nBytes);
//    cudaMemPrefetchAsync(x, nBytes, cudaCpuDeviceId);
//    cudaMemPrefetchAsync(y, nBytes, cudaCpuDeviceId);
//    cudaMemPrefetchAsync(z, nBytes, cudaCpuDeviceId);

    // 初始化数据
    for (int i = 0; i < N; ++i) {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    sw.Mark("add");
    for (int i = 0; i < len; i++) {
        add(x, y, z, N);
    }
    sw.PrintWithMills("add");

    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    sw.Mark("cuda_add");
    for (int i = 0; i < len * 1; i++) {
        kernel_add<<< gridSize, blockSize >>>(x, y, z, N);
    }
    sw.PrintWithMills("cuda_add");

    // 同步device 保证结果能正确访问
    // cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放内存
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    std::cout << std::endl;
}

void matrix_add_v1() {
    StopWatch sw;
    int deviceId;
    cudaGetDevice(&deviceId);
    const long M = 1 << 12, N = 1 << 12, len = 10;
    long nBytes = M * N * sizeof(float);
    // 申请host内存
    float **x = new float *[M], **y = new float *[M], **z = new float *[M];
    for (int i = 0; i < M; ++i) {
        x[i] = new float[N];
        y[i] = new float[N];
        z[i] = new float[N];
    }
    cudaMallocManaged(&x, nBytes);
    cudaMallocManaged(&y, nBytes);
    cudaMallocManaged(&z, nBytes);

    // 初始化数据
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            x[i][j] = 10.0;
            y[i][j] = 20.0;
        }
    }

    sw.Mark("matrix_add");
    for (int i = 0; i < len; i++) {
        matrix_add(x, y, z, M, N);
    }
    sw.PrintWithMills("matrix_add");

    // 定义kernel的执行配置
    dim3 blockSize(64, 64);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);
    // 执行kernel
    sw.Mark("cuda_matrix_add");
    for (int i = 0; i < len * 1; i++) {
        kernel_matrix_add<<< gridSize, blockSize >>>(x, y, z, M, N);
    }
    sw.PrintWithMills("cuda_matrix_add");

    // 同步device 保证结果能正确访问
    // cudaDeviceSynchronize();
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i][i] - 30.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放内存
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    std::cout << std::endl;
}

int main() {
    //show_cuda_info();
    add_v1();
    add_v2();
    //matrix_add_v1();
    return 0;
}
