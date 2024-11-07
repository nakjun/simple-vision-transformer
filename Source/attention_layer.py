import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(AttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads."

        # Query, Key, Value projection layers
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (batch_size, num_patches, embed_dim * 3)
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, num_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Separate Q, K, V

        # Scaled Dot-Product Attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, num_patches, num_patches)
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Weighted sum of values
        attn_output = (attn_probs @ v)  # (batch_size, num_heads, num_patches, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, num_patches, embed_dim)

        # Output projection
        output = self.out_proj(attn_output)
        return output