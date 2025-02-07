import math

import torch
import torch.nn as nn

from src.models.components.transformer import SinusoidalEncoding


class EmbeddingPool(nn.Module):
    def __init__(self, embed_dim: int, d_model: int, num_heads: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model, num_heads, kdim=embed_dim, vdim=embed_dim, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, embed: torch.Tensor):
        embed = embed.permute(0, 2, 1)

        embed = embed + self.positional_encoding
        query = self.query.repeat(embed.shape[0], 1, 1)
        embed, _ = self.attn(query=query, key=embed, value=embed)

        return embed.squeeze(1)
