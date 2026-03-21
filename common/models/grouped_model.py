from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mtcn_backbone import MTCNBackbone, BackboneConfig


class ParticipantAggregator(nn.Module):

    def __init__(self, d_in: int, d_out: int, method: str = "mlp", dropout: float = 0.2):
        super().__init__()
        self.method = method
        self.d_in = d_in
        self.d_out = d_out

        if method == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_out, d_out),
            )
        elif method == "attention":
            self.query = nn.Linear(d_in, 1)
            self.proj = nn.Linear(d_in, d_out)
        elif method == "mean":
            self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def forward(self, session_reprs: torch.Tensor, session_valid: torch.Tensor) -> torch.Tensor:
        mask = session_valid.float().unsqueeze(-1)  
        masked_reprs = session_reprs * mask

        if self.method == "mean":
            n_valid = mask.sum(dim=1).clamp(min=1)  
            pooled = masked_reprs.sum(dim=1) / n_valid  
            return self.proj(pooled)

        elif self.method == "mlp":
            n_valid = mask.sum(dim=1).clamp(min=1)
            pooled = masked_reprs.sum(dim=1) / n_valid
            return self.mlp(pooled)

        elif self.method == "attention":
            scores = self.query(session_reprs).squeeze(-1)  
            scores = scores.masked_fill(~session_valid, float("-inf"))
            weights = F.softmax(scores, dim=-1) 
            weights = weights.masked_fill(~session_valid, 0.0)
            pooled = (weights.unsqueeze(-1) * session_reprs).sum(dim=1)  
            return self.proj(pooled)


class SessionTypeClassifier(nn.Module):
    def __init__(self, d_in: int, n_classes: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class GroupedModel(nn.Module):

    def __init__(
        self,
        backbone: MTCNBackbone,
        d_shared: int,
        aggregator_method: str = "mlp",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = backbone
        self.aggregator = ParticipantAggregator(
            d_in=d_shared, d_out=d_shared,
            method=aggregator_method, dropout=dropout,
        )
        self.session_type_head = SessionTypeClassifier(d_in=d_shared)

    def forward(
        self,
        flat_batch: dict,
        n_participants: int,
        session_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        session_reprs = self.backbone(flat_batch) 

        B = n_participants
        session_grid = session_reprs.view(B, 4, -1)


        participant_repr = self.aggregator(session_grid, session_valid) 

        session_type_logits = self.session_type_head(session_reprs) 

        return {
            "session_reprs": session_reprs,
            "participant_repr": participant_repr,
            "session_type_logits": session_type_logits,
        }


class CORALHead(nn.Module):

    def __init__(self, d_in: int, n_items: int = 21, n_thresholds: int = 3):
        super().__init__()
        self.n_items = n_items
        self.n_thresholds = n_thresholds

        self.score_fc = nn.Linear(d_in, n_items)

        self.raw_thresholds = nn.Parameter(torch.zeros(n_items, n_thresholds))
        nn.init.constant_(self.raw_thresholds, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score_fc(x)

        spacings = F.softplus(self.raw_thresholds) 
        thresholds = torch.cumsum(spacings, dim=-1) 

        logits = scores.unsqueeze(-1) - thresholds.unsqueeze(0) 
        return logits

    @staticmethod
    def predict_int(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) > 0.5).long().sum(dim=-1)

    @staticmethod
    def predict_int_monotonic(logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        P0 = 1.0 - p1
        P1 = p1 - p2
        P2 = p2 - p3
        P3 = p3
        class_probs = torch.stack([P0, P1, P2, P3], dim=-1)
        return class_probs.argmax(dim=-1)

    @staticmethod
    def predict_expectation(logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        E = p1 + p2 + p3
        return E.round().long().clamp(0, 3)
