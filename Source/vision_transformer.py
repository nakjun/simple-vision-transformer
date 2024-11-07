from Source.patch_embedding import PatchEmbedding
from Source.position_embedding import PositionEmbedding
from Source.transformer_encoder_layer import TransformerEncoderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, num_classes: int, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.position_embed = PositionEmbedding(num_patches, embed_dim)

        # Classification token (CLS token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch Embedding and Position Embedding
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        batch_size = x.size(0)

        # Add CLS token to the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)

        # Add Position Embedding
        x = self.position_embed(x)

        # Pass through Transformer Encoder Layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Apply LayerNorm
        x = self.norm(x)

        # Use CLS token for classification
        cls_token_final = x[:, 0]  # (batch_size, embed_dim)
        output = self.classifier(cls_token_final)  # (batch_size, num_classes)

        return output