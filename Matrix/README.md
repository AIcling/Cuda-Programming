## 关于矩阵乘法的几种优化思路

### 1. 基于分块矩阵的优化方法

分块乘法的核心思想是将矩阵分成多个小块（blocks），然后在每个小块上进行局部计算，从而减少数据传输和提高缓存命中率。

```cuda
#define TILE_SIZE 16  // 定义每个块的大小

// 核函数：分块矩阵乘法
__global__ void matrixMultiplyTiled(const float* A, const float* B, float* C, int N) {
    // 计算当前线程负责计算的C矩阵的位置
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 中间结果存储在局部变量中
    float value = 0.0f;

    // 分块计算矩阵乘法
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 取出当前块中对应位置的元素（此处尚未使用共享内存）
        float A_element = 0.0f;
        float B_element = 0.0f;
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N) {
            A_element = A[row * N + t * TILE_SIZE + threadIdx.x];
        }
        if (col < N && (t * TILE_SIZE + threadIdx.y) < N) {
            B_element = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        }

        // 累加计算
        value += A_element * B_element;
    }

    // 将计算结果写入结果矩阵C
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}
```

### 2. 基于共享内存的优化方法

在传统的矩阵乘法中，每个线程会从全局内存中读取多个数据元素。对于一个 **16**×**16** 的线程块，计算过程中需要频繁地从全局内存中获取这些元素。然而，通过共享内存，这些数据只需要从全局内存中读取一次，然后所有线程都可以重复使用它们进行计算。通过将数据块加载到共享内存中，后续计算可以直接从共享内存中读取数据，大大减少了全局内存的访问次数。

__(1) 基于dense tiled Matrix Multiplication的共享内存优化__：[`opt_matrix_mul.cu`](opt_matrix_mul.cu)

在稠密矩阵乘法的分块矩阵乘法 **（Tiled Matrix Multiplication）**。通过将这些小块的数据加载到共享内存中，我们可以显著减少全局内存访问次数，并提高计算效率。

__(2)基于Sparse Matrix Multiplication的共享内存优化__：[`opt_sparse_mul.cu`](opt_sparse_mul.cu)

在稀疏矩阵乘法中，使用共享内存进行优化相对复杂，因为稀疏矩阵中每行的非零元素数量和位置都不一致。但是，通过将某些稀疏矩阵块加载到共享内存中，仍然可以减少全局内存访问，提升效率。
