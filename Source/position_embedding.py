import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int):
        super(PositionEmbedding, self).__init__()
        # 학습 가능한 위치 임베딩 벡터
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        # x: (batch_size, num_patches, embed_dim)
        x = x + self.position_embeddings  # 위치 임베딩을 더함
        return x