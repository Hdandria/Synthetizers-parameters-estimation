import math
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
from einops import rearrange

from src.models.components.sinkhorn import SinkhornAttention, sinkhorn_C


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
    def __init__(self, d_model: int, filler_tokens: int = 0, params_per_token: int = 2):
        super().__init__()
        self.forward_proj = nn.Linear(params_per_token, d_model)
        self.backward_proj = nn.Linear(d_model, params_per_token)
        self.params_per_token = params_per_token

        if filler_tokens > 0:
            self.filler_tokens = nn.Parameter(torch.randn(1, filler_tokens, d_model))
        else:
            self.filler_tokens = None

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        k = x.shape[-1] // self.params_per_token
        x = rearrange(x, "b (d k) -> b k d", k=k)

        x = self.forward_proj(x)

        if self.filler_tokens is not None:
            filler_tokens = self.filler_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat([x, filler_tokens], dim=1)

        return x

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        if self.filler_tokens is not None:
            num_filler = self.filler_tokens.shape[1]
            x = x[:, :-num_filler, :]

        x = self.backward_proj(x)
        x = rearrange(x, "b k d -> b (d k)", d=self.params_per_token)
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


class SemiReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0.0)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        positive_mask = x > 0.0
        direction_mask = grad_output <= 0.0
        mask = positive_mask | direction_mask

        grad_input = torch.where(mask, grad_output, 0.0)
        return grad_input


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
        value_code: Literal["scale", "sin", "proto"] = "scale",
        assignment_type: Literal["linear", "sinkhorn", "exp", "semi_relu"] = "linear",
        sym_init: bool = True,
        filler_tokens: int = 0,
        sinkhorn_iters: int = 10,
        sinkhorn_reg: float = 0.2,
        var_penalty: bool = False,
        initial_ffn: bool = False,
        final_ffn: bool = False,
        num_prototypes: int = 2,
    ):
        super().__init__()

        if assignment_type == "linear":
            # assignment = torch.rand(num_tokens, num_params)
            # assignment = assignment / assignment.sum(1, keepdim=True)
            assignment = torch.full(
                (num_tokens, num_params), 1.0 / math.sqrt(num_tokens * num_params)
            )
            assignment = assignment + 1e-4 * torch.randn_like(assignment)
        elif assignment_type == "sinkhorn":
            assignment = torch.randn(num_tokens, num_params) * 0.1
        elif assignment_type == "exp":
            assignment = torch.randn(num_tokens, num_params)
            assignment = assignment - torch.logsumexp(assignment, dim=(0, 1))
        elif assignment_type == "semi_relu":
            assignment = torch.rand(num_tokens, num_params)
            assignment = assignment / assignment.sum()
        else:
            raise ValueError(f"Unknown assignment type {assignment_type}")

        self._assignment = nn.Parameter(assignment)

        # self._assignment = nn.Parameter(torch.randn(num_tokens, num_params) * 0.1)
        # self._assignment = nn.Parameter(torch.empty(num_tokens, num_params))
        # nn.init.xavier_normal_(self._assignment)

        # proj = torch.randn(1, d_model) / math.sqrt(d_model)
        # self.value_encoding = nn.Parameter(proj.repeat(num_params, 1))

        if value_code == "sin":
            self.sinusoidal_encoding = SinusoidalEncoding(d_model)
            proj = torch.randn(num_params, d_model) / math.sqrt(d_model)
            self._in_projection = nn.Parameter(proj)
            self._out_projection = nn.Parameter(proj.T)
        elif value_code == "proto":
            proto = torch.randn(num_prototypes, d_model) / math.sqrt(d_model)
            self.in_prototypes = nn.Parameter(proto)
            self.out_prototypes = nn.Parameter(proto)
            mappings = torch.rand(num_params, num_prototypes)
            self.in_proto_mappings = nn.Parameter(mappings)
            self.out_proto_mappings = nn.Parameter(mappings)
        elif value_code == "scale":
            if sym_init:
                # proj = torch.randn(1, d_model) / math.sqrt(d_model)
                #
                # self._in_projection = nn.Parameter(proj.repeat(num_params, 1))
                # self._out_projection = nn.Parameter(proj.T.repeat(1, num_params))

                proj = torch.randn(1, d_model) / math.sqrt(d_model)
                proj = proj.repeat(num_params, 1)
                proj = proj + 1e-4 * torch.randn_like(proj)

                self._in_projection = nn.Parameter(proj)
                self._out_projection = nn.Parameter(proj.T)
            else:
                proj = torch.randn(num_params, d_model) / math.sqrt(d_model)
                self._in_projection = nn.Parameter(proj)
                self._out_projection = nn.Parameter(proj.T)

        self.value_code = value_code

        self.assignment_type = assignment_type
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_reg = sinkhorn_reg
        self.var_penalty = var_penalty

        if initial_ffn:
            self.initial_ffn = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.initial_ffn = None

        if final_ffn:
            self.final_ffn = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.final_ffn = None

        if filler_tokens > 0:
            self.filler_tokens = nn.Parameter(
                torch.randn(1, filler_tokens, d_model) / math.sqrt(d_model)
            )
        else:
            self.filler_tokens = None

    @property
    def assignment(self):
        if self.assignment_type == "linear":
            return self._assignment
        elif self.assignment_type == "exp":
            return torch.exp(self._assignment)
        elif self.assignment_type == "sinkhorn":
            return sinkhorn_C(self._assignment, self.sinkhorn_iters, self.sinkhorn_reg)
        elif self.assignment_type == "semi_relu":
            return SemiReLU.apply(self._assignment)

    @property
    def in_projection(self):
        if self.value_code == "scale":
            return self._in_projection
        elif self.value_code == "sin":
            return self._in_projection
        elif self.value_code == "proto":
            return torch.einsum("np,pd->nd", self.in_proto_mappings, self.in_prototypes)

    @property
    def out_projection(self):
        if self.value_code == "scale":
            return self._out_projection
        elif self.value_code == "sin":
            return self._out_projection
        elif self.value_code == "proto":
            return torch.einsum(
                "np,pd->dn", self.out_proto_mappings, self.out_prototypes
            )

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        if self.value_code == "sin":
            values = self.sinusoidal_encoding(x)
            values = values * self.in_projection
        else:
            values = torch.einsum("bn,nd->bnd", x, self.in_projection)

        if self.initial_ffn is not None:
            values = self.initial_ffn(values)

        tokens = torch.einsum("bnd,kn->bkd", values, self.assignment)

        if self.filler_tokens is None:
            return tokens

        filler = self.filler_tokens.repeat(x.shape[0], 1, 1)
        tokens = torch.cat([tokens, filler], dim=1)
        return tokens

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        if self.filler_tokens is not None:
            num_filler = self.filler_tokens.shape[1]
            x = x[:, :-num_filler]

        deassigned = torch.einsum("bkd,kn->bnd", x, self.assignment)

        if self.final_ffn is not None:
            deassigned = self.final_ffn(deassigned)

        return torch.einsum("bnd,dn->bn", deassigned, self.out_projection)

    def penalty(self) -> torch.Tensor:
        # we apply L1 penalty to the assignment matrix
        penalty = self.assignment.abs().mean()
        if self.var_penalty:
            var_penalty = self._in_projection.std(dim=0).mean()
            penalty = penalty + var_penalty

        return penalty


class LearntProjectionIII(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_params: int,
        num_tokens: int,
        num_heads: int = 4,
        init_scale: float = 1e-4,
        attn_type: Literal["cross", "cat"] = "cross",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_params = num_params
        self.num_tokens = num_tokens

        _param_embeds = torch.randn(1, d_model).repeat(num_params, 1)
        _param_embeds = _param_embeds + init_scale * torch.randn_like(_param_embeds)
        _param_outputs = _param_embeds

        self._p2t_tokens = nn.Parameter(torch.randn(1, num_tokens, d_model))
        self._t2p_tokens = nn.Parameter(torch.randn(1, num_params, d_model))
        self._param_embeds = nn.Parameter(_param_embeds)
        self._param_outputs = nn.Parameter(_param_outputs)

        self.p2t_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.t2p_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.p2t_norm = nn.LayerNorm(d_model)
        self.t2p_norm = nn.LayerNorm(d_model)

        self.p2t_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.t2p_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.attn_type = attn_type

    def _do_attn(
        self, attn: nn.MultiheadAttention, q: torch.Tensor, kv: torch.Tensor
    ) -> torch.Tensor:
        if self.attn_type == "cross":
            return attn(q, kv, kv)[0]
        elif self.attn_type == "cat":
            seq = torch.cat([q, kv], dim=1)
            toks = attn(seq, seq, seq)[0]
            return toks[:, : q.shape[1], :]

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        # project scalars to vectors
        params = torch.einsum("bn,nd->bnd", x, self._param_embeds)

        # pass thru FFN with residual (no norm)
        params = self.p2t_ffn(params) + params

        # cross attention bit
        params = self.p2t_norm(params)
        query = self._p2t_tokens.repeat(x.shape[0], 1, 1)
        tokens = self._do_attn(self.p2t_attn, query, params)
        tokens = tokens + query

        return tokens

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        # cross attn
        x = self.t2p_norm(x)
        query = self._t2p_tokens.repeat(x.shape[0], 1, 1)
        params = self._do_attn(self.t2p_attn, query, x)
        params = params + query

        # pass thru FFN with residual
        res = params
        params = self.t2p_ffn(params)
        params = params + res

        params = torch.einsum("bnd,nd->bn", params, self._param_outputs)

        return params

    def penalty(self) -> torch.Tensor:
        return 0.0


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
        norm: Literal["layer", "rms"] = "layer",
        first_norm: bool = True,
    ):
        super().__init__()
        if first_norm:
            self.norm1 = (
                nn.LayerNorm(d_model) if norm == "layer" else nn.RMSNorm(d_model)
            )
        else:
            self.norm1 = nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if norm == "layer" else nn.RMSNorm(d_model)
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
        norm: Literal["layer", "rms"] = "layer",
        skip_first_norm: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DiTransformerBlock(
                    d_model,
                    conditioning_dim + 1,
                    num_heads,
                    d_ff,
                    dropout,
                    norm,
                    first_norm=False if i == 0 and skip_first_norm else True,
                )
                for i in range(num_layers)
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
