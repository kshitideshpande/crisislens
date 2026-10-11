"""One model class for every neural experiment.

mode="text"    BERTweet only
mode="image"   CLIP vision encoder only
mode="concat"  both, concatenated and mixed by an MLP
mode="gated"   both, mixed by a learned per-dimension gate (the deployed model)

All modes share the same three task heads, so differences in results come from
the inputs and the fusion method, not from anything else.

Gated fusion, per post:
    h_t = project(text vector)        h_i = project(image vector)
    g   = sigmoid(W [h_t ; h_i])      a vector in (0, 1), one value per dimension
    h   = g * h_t + (1 - g) * h_i
"image reliance" = mean(1 - g): how much this post's prediction leaned on the image.
"""

from __future__ import annotations

import torch
from torch import nn

from crisislens.models.encoders import hidden_size, image_features, text_features

MODES = ("text", "image", "concat", "gated")


def projector(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())


class FusionClassifier(nn.Module):
    def __init__(
        self,
        mode: str,
        task_classes: dict[str, int],
        text_encoder: nn.Module | None = None,
        image_encoder: nn.Module | None = None,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        modality_dropout: float = 0.0,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.uses_text = mode in ("text", "concat", "gated")
        self.uses_image = mode in ("image", "concat", "gated")
        if self.uses_text and text_encoder is None:
            raise ValueError(f"mode={mode} needs a text encoder")
        if self.uses_image and image_encoder is None:
            raise ValueError(f"mode={mode} needs an image encoder")

        self.text_encoder = text_encoder if self.uses_text else None
        self.image_encoder = image_encoder if self.uses_image else None
        self.modality_dropout = modality_dropout
        self.tasks = list(task_classes)

        if self.uses_text:
            self.text_proj = projector(hidden_size(text_encoder), hidden_dim, dropout)
            self.missing_text = nn.Parameter(torch.zeros(hidden_dim))  # learned "no text" vector
        if self.uses_image:
            self.image_proj = projector(hidden_size(image_encoder), hidden_dim, dropout)
            self.missing_image = nn.Parameter(torch.zeros(hidden_dim))
        if mode == "concat":
            self.mixer = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        if mode == "gated":
            self.gate = nn.Linear(2 * hidden_dim, hidden_dim)

        self.heads = nn.ModuleDict(
            {t: nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, n)) for t, n in task_classes.items()}
        )

    def _presence(self, batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Which posts have each modality, after random modality dropout in training."""
        n = batch["labels"].shape[0] if "labels" in batch else batch["pixel_values"].shape[0]
        text_ok = batch.get("text_present", torch.ones(n, dtype=torch.bool)).to(device).bool()
        image_ok = batch.get("image_present", torch.ones(n, dtype=torch.bool)).to(device).bool()
        if self.training and self.modality_dropout > 0 and self.mode in ("concat", "gated"):
            r = torch.rand(n, device=device)
            both = text_ok & image_ok  # only drop when the other modality is there
            text_ok = text_ok & ~(both & (r < self.modality_dropout / 2))
            image_ok = image_ok & ~(both & (r >= 1 - self.modality_dropout / 2))
        return text_ok, image_ok

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        text_ok, image_ok = self._presence(batch, device)

        h_t = h_i = None
        if self.uses_text:
            feats = text_features(self.text_encoder, batch["input_ids"].to(device), batch["attention_mask"].to(device))
            h_t = torch.where(text_ok[:, None], self.text_proj(feats), self.missing_text[None, :])
        if self.uses_image:
            feats = image_features(self.image_encoder, batch["pixel_values"].to(device))
            h_i = torch.where(image_ok[:, None], self.image_proj(feats), self.missing_image[None, :])

        out: dict[str, torch.Tensor] = {}
        if self.mode == "text":
            fused = h_t
        elif self.mode == "image":
            fused = h_i
        elif self.mode == "concat":
            fused = self.mixer(torch.cat([h_t, h_i], dim=-1))
        else:
            g = torch.sigmoid(self.gate(torch.cat([h_t, h_i], dim=-1)))
            # A missing modality gets zero weight, so the gate can't use garbage.
            g = torch.where(text_ok[:, None], g, torch.zeros_like(g))
            g = torch.where(image_ok[:, None], g, torch.ones_like(g))
            fused = g * h_t + (1 - g) * h_i
            out["image_reliance"] = (1 - g).mean(dim=-1)

        for task, head in self.heads.items():
            out[task] = head(fused)
        return out

    def param_groups(self, lr_text: float, lr_image: float, lr_head: float, weight_decay: float) -> list[dict]:
        """Small learning rates for pretrained encoders, a larger one for new layers."""
        groups, enc_ids = [], set()
        for enc, lr in ((self.text_encoder, lr_text), (self.image_encoder, lr_image)):
            if enc is None:
                continue
            params = [p for p in enc.parameters() if p.requires_grad]
            enc_ids |= {id(p) for p in enc.parameters()}
            if params:
                groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})
        head = [p for p in self.parameters() if id(p) not in enc_ids and p.requires_grad]
        groups.append({"params": head, "lr": lr_head, "weight_decay": weight_decay})
        return groups
