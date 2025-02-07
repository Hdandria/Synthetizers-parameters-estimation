import math

import torch
import torch.nn as nn


class EmbeddingPool(nn.Module):
    def __init__(self, embed_dim: int, d_model: int, num_heads: int):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.resid = nn.Linear(embed_dim, d_model, False)

        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))

    def forward(self, embed: torch.Tensor):
        embed = embed.permute(0, 2, 1)
        embed = self.ffn(embed) + self.resid(embed)
        query = self.query.repeat(embed.shape[0], 1, 1)
        output, _ = self.attn(query, embed, embed)

        return output + query
