#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}


int main() {
    int N; // 向量长度
    std::cout << "Enter the length of the vectors: ";
    std::cin >> N;

    size_t size = N * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    std::cout << "Enter elements of vector A:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cin >> h_A[i];
    }

    std::cout << "Enter elements of vector B:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cin >> h_B[i];
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 定义线程块和网格的维度
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 启动CUDA核函数
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result vector C:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
