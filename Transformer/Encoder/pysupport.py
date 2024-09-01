import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

linear_cpp = load(name="linear_cpp", sources=["qkv.cpp", "qkv_kernel.cu"])
attention_cpp = load(name="attention_cpp", sources=["attention.cpp", "attention_kernel.cu"])

class ComputeQKV(Function):
    @staticmethod
    def forward(ctx, input, weights):
        ctx.save_for_backward(input, weights)
        
        output = linear_cpp.linear_forward(input, weights)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        
        grad_input = grad_output.mm(weights.t())
        grad_weights = input.t().mm(grad_output)
        
        return grad_input, grad_weights
