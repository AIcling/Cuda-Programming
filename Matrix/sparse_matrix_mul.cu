#include <iostream>
#include <cuda_runtime.h>
#include <vector>

__global__ void spmm_csr(const int* A_row_ptr, const int* A_col_indices, const float* A_values, 
                         const int* B_row_ptr, const int* B_col_indices, const float* B_values, 
                         float* C, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        for (int i = A_row_ptr[row]; i < A_row_ptr[row + 1]; ++i) {
            int A_col = A_col_indices[i];
            float A_val = A_values[i];

            for (int j = B_row_ptr[A_col]; j < B_row_ptr[A_col + 1]; ++j) {
                int B_col = B_col_indices[j];
                atomicAdd(&C[row * N + B_col], A_val * B_values[j]);
            }
        }
    }
}

int main() {
    int M, K, N;
    std::cout << "Enter dimensions M, K, N (for matrices A[MxK] and B[KxN]): ";
    std::cin >> M >> K >> N;

    std::vector<int> h_A_row_ptr(M + 1), h_A_col_indices, h_B_row_ptr(K + 1), h_B_col_indices;
    std::vector<float> h_A_values, h_B_values;

    std::cout << "Enter A's row pointers: ";
    for (int i = 0; i <= M; ++i) {
        std::cin >> h_A_row_ptr[i];
    }
    h_A_col_indices.resize(h_A_row_ptr[M]);
    h_A_values.resize(h_A_row_ptr[M]);
    std::cout << "Enter A's column indices and values: ";
    for (int i = 0; i < h_A_row_ptr[M]; ++i) {
        std::cin >> h_A_col_indices[i] >> h_A_values[i];
    }

    std::cout << "Enter B's row pointers: ";
    for (int i = 0; i <= K; ++i) {
        std::cin >> h_B_row_ptr[i];
    }
    h_B_col_indices.resize(h_B_row_ptr[K]);
    h_B_values.resize(h_B_row_ptr[K]);
    std::cout << "Enter B's column indices and values: ";
    for (int i = 0; i < h_B_row_ptr[K]; ++i) {
        std::cin >> h_B_col_indices[i] >> h_B_values[i];
    }

    std::vector<float> h_C(M * N, 0.0f);

    int *d_A_row_ptr, *d_A_col_indices, *d_B_row_ptr, *d_B_col_indices;
    float *d_A_values, *d_B_values, *d_C;

    cudaMalloc(&d_A_row_ptr, (M + 1) * sizeof(int));
    cudaMalloc(&d_A_col_indices, h_A_col_indices.size() * sizeof(int));
    cudaMalloc(&d_A_values, h_A_values.size() * sizeof(float));
    cudaMalloc(&d_B_row_ptr, (K + 1) * sizeof(int));
    cudaMalloc(&d_B_col_indices, h_B_col_indices.size() * sizeof(int));
    cudaMalloc(&d_B_values, h_B_values.size() * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A_row_ptr, h_A_row_ptr.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_col_indices, h_A_col_indices.data(), h_A_col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, h_A_values.data(), h_A_values.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_row_ptr, h_B_row_ptr.data(), (K + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_col_indices, h_B_col_indices.data(), h_B_col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_values, h_B_values.data(), h_B_values.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float));  // 初始化结果矩阵C

    int blockSize = 256;
    int numBlocks = (M + blockSize - 1) / blockSize;
    spmm_csr<<<numBlocks, blockSize>>>(d_A_row_ptr, d_A_col_indices, d_A_values, 
                                       d_B_row_ptr, d_B_col_indices, d_B_values, 
                                       d_C, M, N);

    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result matrix C (first 5 elements): ";
    for (int i = 0; i < 5 && i < M * N; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_A_row_ptr);
    cudaFree(d_A_col_indices);
    cudaFree(d_A_values);
    cudaFree(d_B_row_ptr);
    cudaFree(d_B_col_indices);
    cudaFree(d_B_values);
    cudaFree(d_C);

    return 0;
}
