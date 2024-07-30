#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixTranspose(const float* A, float* B, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int idx_in = row * numCols + col;
        int idx_out = col * numRows + row;
        B[idx_out] = A[idx_in];
    }
}

int main() {
    int numRows, numCols;

    std::cout << "Enter the number of rows: ";
    std::cin >> numRows;
    std::cout << "Enter the number of columns: ";
    std::cin >> numCols;

    int size = numRows * numCols * sizeof(float);

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    std::cout << "Enter elements of matrix A:" << std::endl;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            std::cin >> h_A[i * numCols + j];
        }
    }

    float* d_A, * d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, numRows, numCols);

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    std::cout << "Transposed matrix B:" << std::endl;
    for (int i = 0; i < numCols; ++i) {
        for (int j = 0; j < numRows; ++j) {
            std::cout << h_B[i * numRows + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}
