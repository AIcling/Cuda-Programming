#include <torch/extension.h>

// CUDA forward declarations
void scaled_dot_product_attention_forward_cuda(
    const float* query, const float* key, const float* value,
    float* attention_scores, float* output,
    int batch_size, int num_heads, int seq_length, int head_dim, float scale);

torch::Tensor scaled_dot_product_attention_forward(
    torch::Tensor query, torch::Tensor key, torch::Tensor value, float scale) {

    auto batch_size = query.size(0);
    auto num_heads = query.size(1);
    auto seq_length = query.size(2);
    auto head_dim = query.size(3);

    auto attention_scores = torch::zeros({batch_size, num_heads, seq_length, head_dim}, torch::kCUDA);
    auto output = torch::zeros_like(attention_scores);

    scaled_dot_product_attention_forward_cuda(
        query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
        attention_scores.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, num_heads, seq_length, head_dim, scale);

    return output;
}

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scaled_dot_product_attention_forward", &scaled_dot_product_attention_forward, "Scaled Dot-Product Attention forward (CUDA)");
}
