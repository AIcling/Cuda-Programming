#include <torch/extension.h>

// CUDA forward declarations
void linear_forward_cuda(float* input, float* weights, float* output, int batch_size, int input_dim, int output_dim);

torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weights) {
    auto output = torch::zeros({input.size(0), weights.size(1)}, torch::kCUDA);
    linear_forward_cuda(input.data_ptr<float>(), weights.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1), weights.size(1));
    return output;
}

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Linear forward (CUDA)");
}
