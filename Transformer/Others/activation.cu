#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void relu_activation(const float* inp, float* out, int B, int T, int C) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_num =B * T * C;

    if(idx<total_num){
        out[idx] = fmaxf(0.0f,inp[idx]);
    }

}

__global__ void gelu_activation(const float* inp, float* out, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * T * C;

    if (idx < total_elements) {
        float x = inp[idx];
        // GELU Approximation
        float c = 0.044715f;
        float sqrt_2_over_pi = sqrtf(2.0f / CUDART_PI_F);
        float x_cube = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + c * x_cube);
        float tanh_res = tanhf(tanh_arg);
        out[idx] = 0.5f * x * (1.0f + tanh_res);
    }
}