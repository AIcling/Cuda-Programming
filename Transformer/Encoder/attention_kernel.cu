#include <cuda_runtime.h>

__global__ void scaled_dot_product_attention_kernel(
    const float* query, const float* key, const float* value,
    float* attention_scores, float* output,
    int batch_size, int num_heads, int seq_length, int head_dim, float scale){


}