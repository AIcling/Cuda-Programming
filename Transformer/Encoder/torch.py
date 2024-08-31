import torch
import torch.nn as nn
import torch.nn.functional as F
from pysupport import *

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Q, K, V projections
        self.q_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.k_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.v_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        # This part is intended to be replaced with a C++/CUDA implementation
        # q = self.q_proj(query)
        # k = self.k_proj(key)
        # v = self.v_proj(value)

        # q, k, v are now of shape (batch_size, num_heads, seq_length, head_dim)
        q, k, v = self.compute_qkv(query, key, value)

        # Scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads and apply final linear projection
        attn_output = attn_output.transpose(1, 2).reshape(query.size(0), -1, self.embed_dim)
        output = self.out_proj(attn_output)
        return output

    def compute_qkv(self, query, key, value):
        # This is where you would use C++/CUDA to perform the Q, K, V calculations
        q = ComputeQKV.apply(query, self.q_proj)
        k = ComputeQKV.apply(key, self.k_proj)
        v = ComputeQKV.apply(value, self.v_proj)

        # Reshape for multi-head attention
        q = q.reshape(query.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(key.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(value.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        return q, k, v

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # This is another area that can be implemented in C++/CUDA
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        # This part can also be replaced with a C++/CUDA implementation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim)

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention sublayer
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(src2)
        src = self.layernorm1(src)

        # Feed-forward network sublayer
        src2 = self.ffn(src)
        src = src + self.dropout(src2)
        src = self.layernorm2(src)

        return src

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src
