#include <cuda_runtime.h>

__global__ void linear_kernel(float* input, float* weights, float* output, int batch_size, 
int input_dim, int output_dim){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if(row<batch_size && col<output_dim){
        float value = 0.0f;
        for(int i=0;i<input_dim;i++){
            value += input[row*input_dim + i] * weights[i*output_dim + col];
        }
        weights[row*output_dim+col] = value;
    }
}

void linear_forward_cuda(float* input, float* weights, float* output, int batch_size, int input_dim, int output_dim){
    dim3 thredsPerblock(32,32);
    dim3 numBlocks((output_dim + thredsPerblock.x - 1) / thredsPerblock.x,
                   (batch_size + thredsPerblock.y - 1) / thredsPerblock.y);
    linear_kernel<<<numBlocks,thredsPerblock>>>(input,weights,output,batch_size,input_dim,output_dim);
}   