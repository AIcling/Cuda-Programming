#include <cuda_runtime.h>

__global__ void softmax_kernel(float* out, const float* inp, int B, int T, int C){
    int b = blockIdx.x;  // batch index
    int t = blockIdx.y;  // sequence index
    int c = threadIdx.x;  // feature index

    extern __shared__ float shared_data[];

    //reduce to cal max_val
    float max_val = -INFINITY;
    for(int i=c;i<C;i+=blockDim.x){
        max_val = fmaxf(max_val,inp[b * T * C + t * C + i]);
    }
    shared_data[threadIdx.x] = max_val;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride /= 2){
        if(stride>threadIdx.x){
            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x],shared_data[threadIdx.x]+stride);
        }
        __syncthreads();
    }
    max_val = shared_data[0];

    //reduce to cal exp and sum
    float sum_exp = 0.0f;
    for(int i=c;i<C;i+=blockDim.x){
        shared_data[threadIdx.x] = __expf(inp[b * T * C + t * C + i]-max_val);
        sum_exp += shared_data[threadIdx.x];
        out[b * T * C + t * C + i] = shared_data[threadIdx.x];
    }
    shared_data[threadIdx.x] = sum_exp;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride /= 2){
        if(stride>threadIdx.x){
            shared_data[threadIdx.x] = shared_data[threadIdx.x] + shared_data[threadIdx.x+stride];
        }
        __syncthreads();
    }
    sum_exp = shared_data[0];

    //norm
    for(int i=c;i<C;i+=blockDim.x){
        out[b * T * C + t * C + i] /= sum_exp;
    }
}

__global__ void online_softmax_kernel(float* out, const float* inp, int B, int T, int C) {
    int b = blockIdx.x;  // Batch index
    int t = blockIdx.y;  // Sequence index
    int c = threadIdx.x;  // Feature index

    if (b < B && t < T) {
        const float* inp_row = inp + (b * T + t) * C;
        float* out_row = out + (b * T + t) * C;

        float maxval = -INFINITY;
        double sum = 0.0;

        for (int j = 0; j < C; j++) {
            float maxval_prev = maxval;
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
                sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
            } else {
                sum += expf(inp_row[j] - maxval);
            }
        }

        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval) / sum;
        }
    }
}
