"""
model.py
────────
ConformerCTC  –  small ASR model, ≤ 5 M parameters.

Architecture:
  Conv2D subsampling  (4× time reduction)
  Sinusoidal positional encoding
  N × ConformerBlock  (dim=144, heads=4, ff=576, conv_kernel=31)
  Linear CTC head     (dim → vocab_size)

Parameter budget (dim=144, N=8):
  Subsampling  ≈   100 K
  8 × Conformer ≈ 3 900 K
  CTC head     ≈     5 K
  Total        ≈ 4 005 K  (well under 5 M)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ───────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float, max_len: int = 5_000):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10_000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return self.drop(x + self.pe[:, : x.size(1)])


# ── Sub-modules ───────────────────────────────────────────────────────────────

class ConvSubsampling(nn.Module):
    """2× Conv2D with stride 2 → 4× time reduction.

    Input : (B, T, n_mels)
    Output: (B, T//4, model_dim)
    """

    def __init__(self, n_mels: int, model_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # After 2 × stride-2 conv: freq_dim = ceil(n_mels / 4)
        freq_out = math.ceil(math.ceil(n_mels / 2) / 2)
        self.proj = nn.Linear(32 * freq_out, model_dim)

    @staticmethod
    def _subsample_lengths(lengths: torch.Tensor) -> torch.Tensor:
        # Each Conv2D(stride=2): L_out = floor((L_in + 2*pad - k) / stride) + 1
        # With pad=1, k=3, stride=2: L_out = floor((L + 1) / 2)
        l1 = torch.div(lengths + 1, 2, rounding_mode="floor")
        l2 = torch.div(l1 + 1, 2, rounding_mode="floor")
        return l2

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, F = x.shape
        x = x.unsqueeze(1)          # (B, 1, T, F)
        x = self.conv(x)            # (B, 32, T', F')
        B2, C, T2, F2 = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B2, T2, C * F2)
        x = self.proj(x)            # (B, T', model_dim)
        new_lengths = self._subsample_lengths(lengths)
        return x, new_lengths


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, ff_dim)
        self.fc2  = nn.Linear(ff_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.drop(F.silu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class ConvModule(nn.Module):
    """Conformer convolution sub-layer."""

    def __init__(self, dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm        = nn.LayerNorm(dim)
        self.pw1         = nn.Conv1d(dim, 2 * dim, 1)          # pointwise (GLU)
        self.dw          = nn.Conv1d(dim, dim, kernel_size,
                                     padding=kernel_size // 2, groups=dim)
        self.bn          = nn.BatchNorm1d(dim)
        self.pw2         = nn.Conv1d(dim, dim, 1)
        self.drop        = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.norm(x).transpose(1, 2)   # (B, D, T)
        x = F.glu(self.pw1(x), dim=1)      # (B, D, T)
        x = self.dw(x)
        x = F.silu(self.bn(x))
        x = self.drop(self.pw2(x))
        return x.transpose(1, 2)           # (B, T, D)


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int,
                 conv_kernel: int, dropout: float):
        super().__init__()
        self.ff1       = FeedForward(dim, ff_dim, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn      = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.attn_drop = nn.Dropout(dropout)
        self.conv      = ConvModule(dim, conv_kernel, dropout)
        self.ff2       = FeedForward(dim, ff_dim, dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(
            normed, normed, normed, key_padding_mask=key_padding_mask
        )
        x = x + self.attn_drop(attn_out)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


# ── Full model ────────────────────────────────────────────────────────────────

class ConformerCTC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_mels:     int   = 80,
        model_dim:  int   = 144,
        num_heads:  int   = 4,
        num_layers: int   = 8,
        ff_dim:     int   = 576,
        conv_kernel: int  = 31,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.subsampling = ConvSubsampling(n_mels, model_dim)
        self.pos_enc     = PositionalEncoding(model_dim, dropout)
        self.blocks      = nn.ModuleList([
            ConformerBlock(model_dim, num_heads, ff_dim, conv_kernel, dropout)
            for _ in range(num_layers)
        ])
        self.ctc_head = nn.Linear(model_dim, vocab_size)

    def forward(
        self,
        x:       torch.Tensor,   # (B, T, n_mels)
        lengths: torch.Tensor,   # (B,) – valid frame counts BEFORE subsampling
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_probs : (T', B, vocab_size)  – ready for nn.CTCLoss
            out_lens  : (B,)
        """
        x, out_lens = self.subsampling(x, lengths)   # (B, T', dim)
        x = self.pos_enc(x)

        B, T, _ = x.shape
        pad_mask = (
            torch.arange(T, device=x.device).unsqueeze(0) >= out_lens.unsqueeze(1)
        )  # (B, T) True where padded

        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)

        logits    = self.ctc_head(x)                       # (B, T', V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.permute(1, 0, 2), out_lens        # (T', B, V)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from text_utils import VOCAB_SIZE
    import config as cfg

    model = ConformerCTC(
        vocab_size  = VOCAB_SIZE,
        n_mels      = cfg.N_MELS,
        model_dim   = cfg.MODEL_DIM,
        num_heads   = cfg.NUM_HEADS,
        num_layers  = cfg.NUM_LAYERS,
        ff_dim      = cfg.FF_DIM,
        conv_kernel = cfg.CONV_KERNEL,
        dropout     = cfg.DROPOUT,
    )
    total = model.count_params()
    print(f"Parameters: {total:,}  ({total/1e6:.2f} M)")
    assert total <= 5_000_000, f"Model too large: {total:,} params"

    B, T = 4, 400
    x  = torch.randn(B, T, cfg.N_MELS)
    ln = torch.tensor([400, 380, 300, 250])
    lp, ol = model(x, ln)
    print(f"log_probs: {lp.shape}  out_lens: {ol}")
