from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with projection-only parameters."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply masked self-attention; optionally return attention maps."""
        bsz, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {d_model}")

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        out = self.resid_dropout(self.o_proj(out))
        return out, (attn if return_attn else None)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: attention + MLP residual stack."""

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int, dropout: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one transformer block and optionally expose attention weights."""
        attn_out, attn = self.attn(self.ln_1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn


class TinyDecoderTransformer(nn.Module):
    """Minimal decoder-only transformer used in all tasks."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        seq_len: int = 4,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
        mlp_mult: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, mlp_mult, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, num_classes, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Predict class logits from the final sequence position."""
        _, seq_len = tokens.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len {self.seq_len}, got {seq_len}")

        positions = torch.arange(seq_len, device=tokens.device)
        h = self.token_emb(tokens) + self.pos_emb(positions)[None, :, :]
        h = self.dropout(h)

        attn_maps: list[torch.Tensor] | None = [] if return_attn else None
        for block in self.blocks:
            h, attn = block(h, return_attn=return_attn)
            if return_attn and attn is not None and attn_maps is not None:
                attn_maps.append(attn)

        h = self.ln_f(h)
        logits = self.readout(h[:, -1, :])
        return logits, attn_maps

