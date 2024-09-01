#include <cuda_runtime.h>

__global__ void mean_kernel(float* mean, const float* inp, int B, int T, int C, const int block_size){
    int b = blockIdx.x;  // batch index
    int t = blockIdx.y;  // sequence index
    int thread_idx = threadIdx.x;  // feature index

    extern __shared__ float shared_data[];

    float sum = 0.0f;
    for (int i = thread_idx; i < C; i += blockDim.x) {
        sum += inp[b * T * C + t * C + i];
    }
    // Reduction to calculate mean
    shared_data[thread_idx] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_idx < stride) {
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        mean[b * T + t] = shared_data[0] / C;
    }
}