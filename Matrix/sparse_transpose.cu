#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void countNonZeroPerColumn(const int* A_col_indices, int* col_count, int nnz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        atomicAdd(&col_count[A_col_indices[idx]], 1);
    }
}

__global__ void scanRowPtr(int* row_ptr, int numCols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numCols + 1) {
        for (int i = 1; i < numCols + 1; ++i) {
            row_ptr[i] += row_ptr[i - 1];
        }
    }
}

__global__ void transposeCSR(const int* A_row_ptr, const int* A_col_indices, const float* A_values,
                             int* B_row_ptr, int* B_col_indices, float* B_values,
                             int numRows, int nnz) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows) {
        for (int i = A_row_ptr[row]; i < A_row_ptr[row + 1]; ++i) {
            int col = A_col_indices[i];
            int dest = atomicAdd(&B_row_ptr[col], 1);
            B_col_indices[dest] = row;
            B_values[dest] = A_values[i];
        }
    }
}

void transposeCSRHost(const std::vector<int>& h_A_row_ptr, const std::vector<int>& h_A_col_indices, const std::vector<float>& h_A_values,
                      std::vector<int>& h_B_row_ptr, std::vector<int>& h_B_col_indices, std::vector<float>& h_B_values,
                      int numRows, int numCols, int nnzA) {

    // Allocate device memory
    int* d_A_row_ptr;
    int* d_A_col_indices;
    float* d_A_values;
    int* d_B_row_ptr;
    int* d_B_col_indices;
    float* d_B_values;

    cudaMalloc((void**)&d_A_row_ptr, (numRows + 1) * sizeof(int));
    cudaMalloc((void**)&d_A_col_indices, nnzA * sizeof(int));
    cudaMalloc((void**)&d_A_values, nnzA * sizeof(float));
    cudaMalloc((void**)&d_B_row_ptr, (numCols + 1) * sizeof(int));
    cudaMalloc((void**)&d_B_col_indices, nnzA * sizeof(int));
    cudaMalloc((void**)&d_B_values, nnzA * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A_row_ptr, h_A_row_ptr.data(), (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_col_indices, h_A_col_indices.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, h_A_values.data(), nnzA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_B_row_ptr, 0, (numCols + 1) * sizeof(int));

    // Count non-zero elements per column
    int blockSize = 256;
    int numBlocks = (nnzA + blockSize - 1) / blockSize;
    countNonZeroPerColumn<<<numBlocks, blockSize>>>(d_A_col_indices, d_B_row_ptr, nnzA);

    // Scan row pointers
    scanRowPtr<<<1, numCols + 1>>>(d_B_row_ptr, numCols);

    // Transpose the matrix
    numBlocks = (numRows + blockSize - 1) / blockSize;
    transposeCSR<<<numBlocks, blockSize>>>(d_A_row_ptr, d_A_col_indices, d_A_values,
                                           d_B_row_ptr, d_B_col_indices, d_B_values,
                                           numRows, nnzA);

    // Copy results back to host
    cudaMemcpy(h_B_row_ptr.data(), d_B_row_ptr, (numCols + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_col_indices.data(), d_B_col_indices, nnzA * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_values.data(), d_B_values, nnzA * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A_row_ptr);
    cudaFree(d_A_col_indices);
    cudaFree(d_A_values);
    cudaFree(d_B_row_ptr);
    cudaFree(d_B_col_indices);
    cudaFree(d_B_values);
}

int main() {
    int numRows, numCols, nnzA;

    std::cout << "Enter the number of rows: ";
    std::cin >> numRows;
    std::cout << "Enter the number of columns: ";
    std::cin >> numCols;
    std::cout << "Enter the number of non-zero elements in matrix A: ";
    std::cin >> nnzA;

    std::vector<int> h_A_row_ptr(numRows + 1), h_A_col_indices(nnzA);
    std::vector<float> h_A_values(nnzA);

    std::cout << "Enter row pointers for matrix A: ";
    for (int i = 0; i <= numRows; ++i) {
        std::cin >> h_A_row_ptr[i];
    }

    std::cout << "Enter column indices and values for matrix A: ";
    for (int i = 0; i < nnzA; ++i) {
        std::cin >> h_A_col_indices[i] >> h_A_values[i];
    }

    // 预估结果矩阵的最大非零元素数量
    std::vector<int> h_B_row_ptr(numCols + 1, 0), h_B_col_indices(nnzA);
    std::vector<float> h_B_values(nnzA);

    // Transpose the matrix using CUDA
    transposeCSRHost(h_A_row_ptr, h_A_col_indices, h_A_values, h_B_row_ptr, h_B_col_indices, h_B_values, numRows, numCols, nnzA);

    std::cout << "Transposed matrix row pointers: ";
    for (int i = 0; i <= numCols; ++i) {
        std::cout << h_B_row_ptr[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Transposed matrix column indices and values: ";
    for (int i = 0; i < nnzA; ++i) {
        std::cout << "(" << h_B_col_indices[i] << ", " << h_B_values[i] << ") ";
    }
    std::cout << std::endl;

    return 0;
}
