from __future__ import annotations  

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class BackboneConfig:
    audio_group_dims: dict[str, int] = field(default_factory=dict)
    audio_pooled_group_dims: dict[str, int] = field(default_factory=dict)
    video_group_dims: dict[str, int] = field(default_factory=dict)

    d_adapter: int = 64
    d_model: int = 256
    tcn_layers: int = 4
    tcn_kernel_size: int = 3
    asp_alpha: float = 0.5
    asp_beta: float = 0.5
    dropout: float = 0.1

    n_sessions: int = 4
    d_session: int = 16
    d_shared: int = 256

class GroupAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(F.gelu(self.proj(self.norm(x))))

class ModalityFusion(nn.Module):
    def __init__(self, n_groups: int, d_adapter: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(n_groups * d_adapter, d_model)

    def forward(self, groups: list[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.cat(groups, dim=-1))


class DilatedResidualBlock(nn.Module):
    def __init__(
        self, d_model: int, kernel_size: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        T = x.size(1)

        h = self.norm1(x)
        h = h.transpose(1, 2)
        h = self.conv1(h)[:, :, :T]
        h = F.gelu(h)
        h = self.drop(h)

        # block 2
        h = h.transpose(1, 2)
        h = self.norm2(h)
        h = h.transpose(1, 2)
        h = self.conv2(h)[:, :, :T]
        h = self.drop(h)
        h = h.transpose(1, 2)

        out = h + residual
        out = out * mask.unsqueeze(-1).float()
        return out


class TCN(nn.Module):
    def __init__(
        self, d_model: int, n_layers: int, kernel_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(d_model, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x


class ASP(nn.Module):
    """Attentive Statistics Pooling with VAD and quality control signals."""

    def __init__(self, d_model: int, alpha: float = 0.5, beta: float = 0.5) -> None:
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        vad: torch.Tensor,
        qc: torch.Tensor,
    ) -> torch.Tensor:
        """
        x    : (B, T, D)
        mask : (B, T) bool
        vad  : (B, T) float
        qc   : (B, T) float
        Returns: (B, 2*D)
        """
        e = self.attn(x).squeeze(-1) 
        e = e + self.alpha * vad + self.beta * qc

        # mask invalid positions
        e = e.masked_fill(~mask, float("-inf"))
        w = F.softmax(e, dim=-1)
        w = w.masked_fill(~mask, 0.0)   # to avoid NaN in mean/std when all masked

        w_unsq = w.unsqueeze(-1)
        mean = (w_unsq * x).sum(dim=1)

        diff = x - mean.unsqueeze(1)
        var = (w_unsq * diff ** 2).sum(dim=1)
        std = torch.sqrt(var.clamp(min=1e-8))

        return torch.cat([mean, std], dim=-1)

class MTCNBackbone(nn.Module):
    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.audio_adapters = nn.ModuleDict({
            name: GroupAdapter(d_in, cfg.d_adapter, cfg.dropout)
            for name, d_in in cfg.audio_group_dims.items()
        })
        self.audio_pooled_adapters = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, cfg.d_adapter),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            for name, d_in in cfg.audio_pooled_group_dims.items()
        })
        self.video_adapters = nn.ModuleDict({
            name: GroupAdapter(d_in, cfg.d_adapter, cfg.dropout)
            for name, d_in in cfg.video_group_dims.items()
        })
        self.audio_group_names = sorted(cfg.audio_group_dims.keys())
        self.audio_pooled_group_names = sorted(cfg.audio_pooled_group_dims.keys())
        self.video_group_names = sorted(cfg.video_group_dims.keys())

        self.audio_fusion = ModalityFusion(
            len(self.audio_group_names), cfg.d_adapter, cfg.d_model
        )
        self.video_fusion = ModalityFusion(
            len(self.video_group_names), cfg.d_adapter, cfg.d_model
        )

        self.audio_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)
        self.video_tcn = TCN(cfg.d_model, cfg.tcn_layers, cfg.tcn_kernel_size, cfg.dropout)

        self.audio_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)
        self.video_asp = ASP(cfg.d_model, cfg.asp_alpha, cfg.asp_beta)

        fusion_in = 2 * cfg.d_model * 2  
        fusion_in += len(self.audio_pooled_group_names) * cfg.d_adapter
        fusion_in += cfg.d_session 

        self.session_embed = nn.Embedding(cfg.n_sessions, cfg.d_session)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, cfg.d_shared),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_shared, cfg.d_shared),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        audio_adapted = [
            self.audio_adapters[n](batch["audio_groups"][n])
            for n in self.audio_group_names
        ]
        video_adapted = [
            self.video_adapters[n](batch["video_groups"][n])
            for n in self.video_group_names
        ]

        a = self.audio_fusion(audio_adapted)
        v = self.video_fusion(video_adapted)

        mask_a = batch["mask_audio"]
        mask_v = batch["mask_video"]
        a = a * mask_a.unsqueeze(-1).float()
        v = v * mask_v.unsqueeze(-1).float()

        a = self.audio_tcn(a, mask_a)
        v = self.video_tcn(v, mask_v)

        vad = batch["vad_signal"]
        qc = batch["qc_quality"]
        z_a = self.audio_asp(a, mask_a, vad, qc)
        z_v = self.video_asp(v, mask_v, vad, qc)

        parts = [z_a, z_v]
        parts.extend(
            self.audio_pooled_adapters[name](batch["audio_pooled_groups"][name])
            for name in self.audio_pooled_group_names
        )
        parts.append(self.session_embed(batch["session_idx"]))

        z = torch.cat(parts, dim=-1)
        return self.fusion_mlp(z)
