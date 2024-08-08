#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorMultiply(const float* A, const float* B, float* C, int N) {
    extern __shared__ float shared_data[]; // 共享内存声明
    float* shared_A = shared_data;         // 指向共享内存的前半部分
    float* shared_B = shared_data + blockDim.x; // 指向共享内存的后半部分

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 将全局内存的数据拷贝到共享内存
    if (idx < N) {
        shared_A[threadIdx.x] = A[idx];
        shared_B[threadIdx.x] = B[idx];
    }
    __syncthreads(); // 同步，确保所有线程都拷贝完数据

    // 进行向量乘法计算
    if (idx < N) {
        C[idx] = shared_A[threadIdx.x] * shared_B[threadIdx.x];
    }
}

int main() {
    int N;
    std::cout << "Enter the length of the vectors: ";
    std::cin >> N;

    size_t size = N * sizeof(float);

    // 使用cudaMallocHost分配页锁定内存
    float* h_A;
    float* h_B;
    float* h_C;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C, size);

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

    // 使用异步内存传输和流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 每个线程块的共享内存大小
    size_t sharedMemSize = 2 * threadsPerBlock * sizeof(float);

    vectorMultiply<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream>>>(d_A, d_B, d_C, N);

    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);

    // 等待所有流操作完成
    cudaStreamSynchronize(stream);

    std::cout << "Result vector C:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // 释放流
    cudaStreamDestroy(stream);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
