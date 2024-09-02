#include <cuda_runtime.h>

__global__ void mean_kernel(float* mean, const float* inp, int B, int T, int C){
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

__global__ void rstd_kernel(float* rstd, const float* inp, const float* mean, int B, int T, int C){
    int b = blockIdx.x;  // batch index
    int t = blockIdx.y;  // sequence index
    int thread_idx = threadIdx.x;  // feature index    

    extern __shared__ float shared_data[];

    float sum = 0.0f;
    float mean_val = mean[b * T + t];

    for (int i = thread_idx; i < C; i += blockDim.x) {
        float diff = inp[b * T * C + t * C + i] - mean_val;
        sum += diff * diff;
    }    

    shared_data[thread_idx] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread_idx < stride) {
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        }
        __syncthreads();
    }    

    if (thread_idx == 0) {
        float variance = shared_data[0] / C;
        rstd[b * T + t] = rsqrtf(variance + 1e-5);
    }
}

__global__ void layernorm_forward_kernel(float* out, const float* inp, const float* mean, const float* rstd, const float* weight, const float* bias, int B, int T, int C) {
    int b = blockIdx.x;  // batch index
    int t = blockIdx.y;  // sequence index
    int c = threadIdx.x;  // feature index

    if (c < C) {
        float normalized = (inp[b * T * C + t * C + c] - mean[b * T + t]) * rstd[b * T + t];
        if (weight != nullptr && bias != nullptr) {
            out[b * T * C + t * C + c] = normalized * weight[c] + bias[c];
        } else {
            out[b * T * C + t * C + c] = normalized;
        }
    }
}
