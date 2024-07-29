#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// CUDA核函数，用于执行矩阵加法
__global__ void matrixAdd(const float *A, const float *B, float *C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int idx = row * numCols + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int numRows, numCols;

    std::cout << "Enter the number of rows: ";
    std::cin >> numRows;
    std::cout << "Enter the number of columns: ";
    std::cin >> numCols;

    int size = numRows * numCols * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    std::cout << "Enter elements of matrix A:" << std::endl;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            std::cin >> h_A[i * numCols + j];
        }
    }

    std::cout << "Enter elements of matrix B:" << std::endl;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            std::cin >> h_B[i * numCols + j];
        }
    }


    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numRows, numCols);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result matrix: " << std::endl;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            std::cout << h_C[i * numCols + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
