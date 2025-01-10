from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
from einops import rearrange

from src.models.components.sinkhorn import (SinkhornAttention, sinkhorn,
                                            sinkhorn_C)


class PositionalEncoding(nn.Module):
    def __init__(self, size: int, num_pos: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, num_pos, size))

    def penalty(self) -> torch.Tensor:
        # structured sparsity
        return self.pe.norm(2.0, dim=-1).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe[:, : x.shape[1], :]
        return x + pe


def _make_block_diagonal_projections(
    param_group_lengths: Sequence[int], d_model: int
) -> nn.Module:
    n_params = sum(param_group_lengths)
    n_groups = len(param_group_lengths)
    params = torch.zeros(2, n_params, d_model * n_groups)

    i, j = 0, 0
    for length in param_group_lengths:
        data = torch.empty(2, length, d_model)
        nn.init.kaiming_uniform_(data)
        params[:, i : i + length, j : j + d_model].copy_(data)
        i += length
        j += d_model

    return nn.Parameter(params)


def _make_block_diagonal_mask(
    param_group_lengths: Sequence[int], d_model: int
) -> nn.Module:
    n_params = sum(param_group_lengths)
    n_groups = len(param_group_lengths)
    params = torch.ones(1, n_params, d_model * n_groups)

    i, j = 0, 0
    for length in param_group_lengths:
        params[:, i : i + length, j : j + d_model].fill_(0.0)
        i += length
        j += d_model

    return nn.Parameter(params)


class ParamToTokenProjection(nn.Module):
    def __init__(self, d_model: int, param_group_lengths: Sequence[int]):
        super().__init__()
        self.proj = _make_block_diagonal_projections(param_group_lengths, d_model)
        self.register_buffer(
            "penalty_mask", _make_block_diagonal_mask(param_group_lengths, d_model)
        )
        self.n_groups = len(param_group_lengths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, self.proj[0])
        x = rearrange(x, "b (k d) -> b k d", k=self.n_groups)

        return x

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b k d -> b (k d)")
        return torch.matmul(x, self.proj[1].transpose(0, 1))

    def penalty(self) -> torch.Tensor:
        masked = self.proj * self.penalty_mask

        return masked.abs().mean()


class KSinParamToTokenProjection(nn.Module):
    def __init__(self, d_model: int, k: int):
        super().__init__()
        self.forward_proj = nn.Linear(2, d_model)
        self.backward_proj = nn.Linear(d_model, 2)

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        k = x.shape[-1] // 2
        x = rearrange(x, "b (d k) -> b k d", k=k)
        return self.forward_proj(x)

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backward_proj(x)
        x = rearrange(x, "b k d -> b (d k)", d=2)
        return x

    def penalty(self) -> torch.Tensor:
        return 0.0


class LearntProjection(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_params: int,
        num_tokens: int,
        init: Literal["zero", "normal", "kaiming"],
        penalty: Literal["l1", "group"] = "l1",
        tying: bool = False,
    ):
        super().__init__()
        proj_in = torch.empty(num_tokens, d_model, num_params)
        proj_out = torch.empty(d_model, num_tokens, num_params)

        if init == "zero":
            nn.init.zeros_(proj_in)
            nn.init.zeros_(proj_out)
        elif init == "normal":
            nn.init.normal_(proj_in)
            nn.init.normal_(proj_out)
        elif init == "kaiming":
            nn.init.kaiming_uniform_(proj_in)
            nn.init.kaiming_uniform_(proj_out)

        self.proj_in = nn.Parameter(proj_in)
        self.proj_out = nn.Parameter(proj_out) if not tying else None
        self.penalty_type = penalty

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b n)
        return torch.einsum("bn,kdn->bkd", x, self.proj_in)

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b k d)
        if self.proj_out is None:
            # we are using tied weights
            return torch.einsum("bkd,kdn->bn", x, self.proj_in)

        # otherwise
        return torch.einsum("bkd,dkn->bn", x, self.proj_out)

    def _l1_penalty(self) -> torch.Tensor:
        penalty = self.proj_in.abs().mean()
        if self.proj_out is not None:
            penalty = penalty + self.proj_out.abs().mean()

        return penalty

    def _group_penalty(self) -> torch.Tensor:
        penalty = self.proj_in.norm(2.0, dim=1).mean()
        if self.proj_out is not None:
            penalty = penalty + self.proj_out.norm(2.0, dim=0).mean()

        return penalty

    def penalty(self) -> torch.Tensor:
        if self.penalty_type == "l1":
            return self._l1_penalty()
        elif self.penalty_type == "group":
            return self._group_penalty()

        raise ValueError(f"Unknown penalty {self.penalty}")


class LearntProjectionII(nn.Module):
    """Smarter learnt projection that factorises into:
    (i) assignment matrix
    (ii) value coding
    (iii) place tokens
    """

    def __init__(
        self,
        d_model: int,
        num_params: int,
        num_tokens: int,
    ):
        super().__init__()

        assignment = torch.rand(num_tokens, num_params)
        assignment = assignment / assignment.sum(1, keepdim=True)
        self.assignment = nn.Parameter(assignment)

        self.value_encoding = nn.Parameter(torch.randn(num_params, d_model))
        self.out_projection = nn.Parameter(torch.randn(d_model, num_params))

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        values = torch.einsum("bn,nd->bnd", x, self.value_encoding)
        return torch.einsum("bnd,kn->bkd", values, self.assignment)

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        deassigned = torch.einsum("bkd,kn->bnd", x, self.assignment)
        return torch.einsum("bnd,dn->bn", deassigned, self.out_projection)

    def penalty(self) -> torch.Tensor:
        # we apply L1 penalty to the assignment matrix
        return self.assignment.abs().mean()


def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = x.norm(2.0, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


def slerp(
    x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    x = normalize(x, eps)
    y = normalize(y, eps)

    cos_theta = torch.einsum("bnd,bnd->bn", x, y)[:, :, None]  # (1 n 1)
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    sin_theta = sin_theta.clamp_min(eps)

    # t has shape (b n 1)
    s0 = torch.sin((1.0 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    return s0 * x + s1 * y


class SinusoidalEncoding(nn.Module):
    """A sinusoidal encoding of scalar values centered around zero."""

    def __init__(self, d_model: int):
        super().__init__()

        k = torch.arange(0, d_model // 2) + 1
        basis = 1 / torch.pow(10000, 2 * k / d_model)

        self.register_buffer("basis", basis[None, None, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (b t)
        cos_part = torch.cos(x[:, :, None] * self.basis)
        sin_part = torch.sin(x[:, :, None] * self.basis)
        return torch.cat([cos_part, sin_part], dim=-1)


class SinsPlusSinkhornAttn(nn.Module):
    """
    Each scalar parameter is given a sinusoidal embedding. Each parameter is given a
    matrix initialised to zeros, which is able to learn a projection, enabling
    the model to break symmetry if necessary.
    Then, to map to tokens, we use sinnkhorn cross-attention between these embeddings
    and a set of learnable query tokens.
    Finally, the out projection (i.e. return from tokens to vector) is a set of
    token-wise matrices.
    """

    def __init__(
        self,
        num_params: int,
        num_tokens: int,
        d_model: int,
        d_embed: int,
        sinkhorn_iters: int = 5,
        sinkhorn_reg: float = 1.0,
    ):
        super().__init__()

        # we encode in a lower dimensional space to save on parameters, as projections
        # can learn different subspaces anyway.
        self.sin_encoding = SinusoidalEncoding(d_embed)
        self.projections = nn.Parameter(torch.zeros(num_params, d_model, d_embed))

        self.query_tokens = nn.Parameter(torch.randn(num_tokens, d_model))
        out_proj = torch.empty(num_tokens, d_model, num_params)
        nn.init.kaiming_normal_(out_proj)
        self.out_proj = nn.Parameter(out_proj)

        self.attn = SinkhornAttention(
            d_model,
            d_model,
            d_model,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_reg=sinkhorn_reg,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b n)
        # x is typically defined on [-1, 1] so our slerp embeddings (p0, p1) define the
        # midpoint and one end. there is a risk of some slerp "aliasing" if the embeds
        # end up too far apart and we get out of interval values (i.e. due to the
        # flow source distribution).
        # TODO: figure out if this is a real problem
        encs = self.sin_encoding(x)
        embeds = torch.einsum("bne,nde->bnd", encs, self.projections)
        tokens = self.attn(self.query_tokens, embeds, embeds)

        return tokens

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b k d)
        return torch.einsum("bkd,kdn->bn", x, self.out_proj)

    def penalty(self) -> torch.Tensor:
        return self.out_proj.norm(2.0, dim=1).mean()


class GeodesicPlusSinkhornAttn(nn.Module):
    """
    Each scalar parameter is associated to two vectors. To compute the embedding, we
    take the hypersphere geodesic (i.e. spherical linear interpolation) between the
    normalised vectors. This means that every parameter can be associated to some arc
    on the sphere.
    Then, to map to tokens, we use sinnkhorn cross-attention between these embeddings
    and a set of learnable query tokens.
    Finally, the out projection (i.e. return from tokens to vector) is a set of
    token-wise matrices.
    """

    def __init__(
        self,
        num_params: int,
        num_tokens: int,
        d_model: int,
        d_embed: int,
        sinkhorn_iters: int = 5,
        sinkhorn_reg: float = 1.0,
    ):
        super().__init__()

        self.p0 = nn.Parameter(torch.randn(1, num_params, d_embed))
        self.p1 = nn.Parameter(torch.randn(1, num_params, d_embed))
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, d_model))
        out_proj = torch.empty(num_tokens, d_model, num_params)
        nn.init.kaiming_normal_(out_proj)
        self.out_proj = nn.Parameter(out_proj)

        self.attn = SinkhornAttention(
            d_model,
            d_embed,
            d_embed,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_reg=sinkhorn_reg,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b n)
        # x is typically defined on [-1, 1] so our slerp embeddings (p0, p1) define the
        # midpoint and one end. there is a risk of some slerp "aliasing" if the embeds
        # end up too far apart and we get out of interval values (i.e. due to the
        # flow source distribution).
        # TODO: figure out if this is a real problem
        embeds = slerp(self.p0, self.p1, x[:, :, None])
        tokens = self.attn(self.query_tokens, embeds, embeds)

        return tokens

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b k d)
        return torch.einsum("bkd,kdn->bn", x, self.out_proj)

    def penalty(self) -> torch.Tensor:
        return self.out_proj.norm(2.0, dim=1).mean()


class AdaptiveLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, conditioning_dim: int, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)
        self.shift = nn.Linear(conditioning_dim, dim)
        self.scale = nn.Linear(conditioning_dim, dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        shift = self.shift(z)
        scale = self.scale(z)
        x = super().forward(x)
        return x * scale + shift


class DiTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        conditioning_dim: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.cond = nn.Sequential(
            nn.Linear(conditioning_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model * 6),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        g1, b1, a1, g2, b2, a2 = self.cond(z)[:, None].chunk(6, dim=-1)

        res = x
        x = self.norm1(x)
        x = g1 * x + b1
        x = self.attn(x, x, x)[0]
        x = a1 * x + res

        res = x
        x = self.norm2(x)
        x = g2 * x + b2
        x = self.ff(x)
        x = a2 * x + res

        return x


class ApproxEquivTransformer(nn.Module):
    """Implements a simple transformer that enforces approximate permutation
    equivariance. This happens by:
        (i) Positional encodings initialised to zero (enforces equivariance at
            init)
        (ii) Regularization on the positional encodings (encourages equivariance)
        (iii) Block diagonal parameter-to-token projection matrix (enforces separation
              between units of synthesiser)
        (iv) Regularization on the off-diagonal elements of the projection matrix
             (encourages separation of units of synthesiser)
    """

    def __init__(
        self,
        projection: nn.Module,
        num_layers: int = 5,
        d_model: int = 1024,
        conditioning_dim: int = 128,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.0,
        num_tokens: int = 32,
        learn_pe: bool = False,
        learn_projection: bool = False,
        pe_type: Literal["initial", "layerwise"] = "initial",
        pe_penalty: float = 0.0,
        projection_penalty: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DiTransformerBlock(
                    d_model, conditioning_dim + 1, num_heads, d_ff, dropout
                )
                for _ in range(num_layers)
            ]
        )

        if pe_type == "initial":
            self.pe = PositionalEncoding(d_model, num_tokens)
            if not learn_pe:
                self.pe.pe.requires_grad = False

        elif pe_type == "layerwise":
            self.pe = nn.ModuleList(
                [PositionalEncoding(d_model, num_tokens) for _ in range(num_layers)]
            )
            if not learn_pe:
                for pe in self.pe:
                    pe.pe.requires_grad = False
        elif pe_type == "none":
            self.pe = None

        self.pe_type = pe_type

        self.projection = projection

        if not learn_projection:
            self.projection.proj.requires_grad = False

        self.pe_penalty = pe_penalty
        self.projection_penalty = projection_penalty

        self.cfg_dropout_token = nn.Parameter(torch.randn(1, conditioning_dim))

    def apply_dropout(self, z: torch.tensor, rate: float = 0.1):
        if rate == 0.0:
            return z

        dropout_mask = torch.rand(z.shape[0], 1, device=z.device) > rate
        return z.where(dropout_mask, self.cfg_dropout_token)

    def penalty(self) -> torch.Tensor:
        penalty = 0.0

        if self.pe_type != "none" and self.pe_penalty > 0.0:
            if self.pe_type == "initial":
                pe_penalty = self.pe.penalty()
            elif self.pe_type == "layerwise":
                pe_penalty = 0.0
                for pe in self.pe:
                    pe_penalty += pe.penalty()

            penalty += pe_penalty * self.pe_penalty

        if self.projection_penalty > 0.0:
            projection_penalty = self.projection.penalty()
            penalty += projection_penalty * self.projection_penalty

        return penalty

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if conditioning is None:
            conditioning = self.cfg_dropout_token.expand(x.shape[0], -1)
        x = self.projection.param_to_token(x)
        z = torch.cat((conditioning, t), dim=-1)

        if self.pe_type == "initial":
            x = self.pe(x)
        for i, layer in enumerate(self.layers):
            if self.pe_type == "layerwise":
                x = self.pe[i](x)
            x = layer(x, z)

        x = self.projection.token_to_param(x)

        return x
