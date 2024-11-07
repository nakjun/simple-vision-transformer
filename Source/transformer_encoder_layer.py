import torch
import torch.nn as nn
import torch.nn.functional as F
from Source.attention_layer import AttentionLayer

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = AttentionLayer(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-Forward Network (FFN)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-Attention with residual connection
        x = x + self.attention(self.norm1(x))
        # Feed-Forward Network with residual connection
        x = x + self.mlp(self.norm2(x))
        return x