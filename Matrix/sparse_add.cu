#include <cuda_runtime.h>
#include <iostream>
#include <vector>


__global__ void sparseMatrixAdd(int* A_row_ptr, int* A_col_indices, float* A_values,
                                int* B_row_ptr, int* B_col_indices, float* B_values,
                                int* C_row_ptr, int* C_col_indices, float* C_values,
                                int numRows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows) {
        int A_start = A_row_ptr[row];
        int A_end = A_row_ptr[row + 1];
        int B_start = B_row_ptr[row];
        int B_end = B_row_ptr[row + 1];

        int C_start = C_row_ptr[row];
        int C_index = C_start;

        while (A_start < A_end && B_start < B_end) {
            if (A_col_indices[A_start] < B_col_indices[B_start]) {
                C_col_indices[C_index] = A_col_indices[A_start];
                C_values[C_index] = A_values[A_start];
                A_start++;
            } else if (A_col_indices[A_start] > B_col_indices[B_start]) {
                C_col_indices[C_index] = B_col_indices[B_start];
                C_values[C_index] = B_values[B_start];
                B_start++;
            } else {
                C_col_indices[C_index] = A_col_indices[A_start];
                C_values[C_index] = A_values[A_start] + B_values[B_start];
                A_start++;
                B_start++;
            }
            C_index++;
        }

        while (A_start < A_end) {
            C_col_indices[C_index] = A_col_indices[A_start];
            C_values[C_index] = A_values[A_start];
            A_start++;
            C_index++;
        }

        while (B_start < B_end) {
            C_col_indices[C_index] = B_col_indices[B_start];
            C_values[C_index] = B_values[B_start];
            B_start++;
            C_index++;
        }
    }
}

int main() {
    int numRows;
    int nnzA, nnzB; // 非零元素数量

    std::cout << "Enter the number of rows: ";
    std::cin >> numRows;
    std::cout << "Enter the number of non-zero elements in matrix A: ";
    std::cin >> nnzA;
    std::cout << "Enter the number of non-zero elements in matrix B: ";
    std::cin >> nnzB;

    // 分配主机内存
    std::vector<int> h_A_row_ptr(numRows + 1), h_B_row_ptr(numRows + 1);
    std::vector<int> h_A_col_indices(nnzA), h_B_col_indices(nnzB);
    std::vector<float> h_A_values(nnzA), h_B_values(nnzB);

    std::cout << "Enter row pointers for matrix A: ";
    for (int i = 0; i <= numRows; ++i) {
        std::cin >> h_A_row_ptr[i];
    }

    std::cout << "Enter column indices and values for matrix A: ";
    for (int i = 0; i < nnzA; ++i) {
        std::cin >> h_A_col_indices[i] >> h_A_values[i];
    }

    std::cout << "Enter row pointers for matrix B: ";
    for (int i = 0; i <= numRows; ++i) {
        std::cin >> h_B_row_ptr[i];
    }

    std::cout << "Enter column indices and values for matrix B: ";
    for (int i = 0; i < nnzB; ++i) {
        std::cin >> h_B_col_indices[i] >> h_B_values[i];
    }

    // 预估结果矩阵的最大非零元素数量
    int nnzC = nnzA + nnzB;

    std::vector<int> h_C_row_ptr(numRows + 1, 0);
    std::vector<int> h_C_col_indices(nnzC);
    std::vector<float> h_C_values(nnzC);

    int *d_A_row_ptr, *d_A_col_indices, *d_B_row_ptr, *d_B_col_indices, *d_C_row_ptr, *d_C_col_indices;
    float *d_A_values, *d_B_values, *d_C_values;
    cudaMalloc(&d_A_row_ptr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_A_col_indices, nnzA * sizeof(int));
    cudaMalloc(&d_A_values, nnzA * sizeof(float));
    cudaMalloc(&d_B_row_ptr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_B_col_indices, nnzB * sizeof(int));
    cudaMalloc(&d_B_values, nnzB * sizeof(float));
    cudaMalloc(&d_C_row_ptr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_C_col_indices, nnzC * sizeof(int));
    cudaMalloc(&d_C_values, nnzC * sizeof(float));

    cudaMemcpy(d_A_row_ptr, h_A_row_ptr.data(), (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_col_indices, h_A_col_indices.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, h_A_values.data(), nnzA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_row_ptr, h_B_row_ptr.data(), (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_col_indices, h_B_col_indices.data(), nnzB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_values, h_B_values.data(), nnzB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_row_ptr, h_C_row_ptr.data(), (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numRows + blockSize - 1) / blockSize;

    // 启动CUDA核函数
    sparseMatrixAdd<<<numBlocks, blockSize>>>(d_A_row_ptr, d_A_col_indices, d_A_values,
                                              d_B_row_ptr, d_B_col_indices, d_B_values,
                                              d_C_row_ptr, d_C_col_indices, d_C_values,
                                              numRows);

    cudaMemcpy(h_C_row_ptr.data(), d_C_row_ptr, (numRows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_col_indices.data(), d_C_col_indices, nnzC * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_values.data(), d_C_values, nnzC * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result matrix row pointers: ";
    for (int i = 0; i <= numRows; ++i) {
        std::cout << h_C_row_ptr[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Result matrix column indices and values: ";
    for (int i = 0; i < nnzC; ++i) {
        std::cout << "(" << h_C_col_indices[i] << ", " << h_C_values[i] << ") ";
    }
    std::cout << std::endl;

    cudaFree(d_A_row_ptr);
    cudaFree(d_A_col_indices);
    cudaFree(d_A_values);
    cudaFree(d_B_row_ptr);
    cudaFree(d_B_col_indices);
    cudaFree(d_B_values);
    cudaFree(d_C_row_ptr);
    cudaFree(d_C_col_indices);
    cudaFree(d_C_values);

    return 0;
}
