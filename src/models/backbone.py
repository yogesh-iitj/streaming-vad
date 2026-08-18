"""Frozen per-frame visual backbone.

Kept deliberately simple: the paper's contribution is the causal temporal
core (src/models/ssm.py) and its theoretical analysis, not the backbone.
Any frozen frame embedding works; two options are wired up:

  * resnet18   — torchvision, ImageNet-pretrained, 512-d. Small, reliable,
                 downloads ~45MB of weights on first use. Good default for
                 fast iteration on a laptop.
  * dinov2_vits14 — torch.hub, 384-d, generally stronger features for
                 anomaly detection. Downloads ~85MB on first use. Switch to
                 this for the numbers that go in the paper.

Both are frozen (no gradient, eval() mode) — only the SSM head is trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrozenBackbone(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.name = name

        if name == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=weights)
            self.embed_dim = net.fc.in_features  # 512
            net.fc = nn.Identity()
            self.net = net

        elif name == "dinov2_vits14":
            # requires internet access on first call (torch.hub download)
            self.net = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained
            )
            self.embed_dim = 384

        else:
            raise ValueError(f"Unknown backbone '{name}'")

        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> [B, embed_dim]"""
        was_training = self.net.training
        self.net.eval()
        if self.name == "dinov2_vits14":
            out = self.net(x)
        else:
            out = self.net(x)
        if was_training:
            self.net.train()
        return out

    def train(self, mode: bool = True):
        # Backbone is always frozen/eval regardless of the parent module's mode.
        return super().train(False)
