"""Streaming VAD model: causal SSM stack + a next-embedding prediction head.

Operates purely in (frozen) embedding space — the backbone lives outside
this module (see backbone.py), so the same detector code path is used
whether embeddings come from a live backbone forward pass or from a
precomputed feature cache (scripts/extract_features.py).

Self-supervision signal: predict e_{t+1} from information available up to
and including e_t (strictly causal). Anomaly score at time t is the
prediction error ||e_hat_t - e_{t+1}||_2 — high error means the recent
frame history did not anticipate what happened next, the standard
predictive-coding formulation used across the VAD literature (e.g. Liu et
al. 2018 "Future Frame Prediction for Anomaly Detection").
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ssm import CausalSSMStack


class PredictionHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, y_t: torch.Tensor) -> torch.Tensor:
        return self.net(y_t)


class StreamingVADModel(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        state_dim: int,
        num_layers: int = 2,
        gate_hidden_dim: int = 64,
        head_hidden_dim: int = 256,
        init_decay_min: float = 0.9,
        init_decay_max: float = 0.999,
        use_gate: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ssm = CausalSSMStack(
            embed_dim, state_dim, num_layers, gate_hidden_dim,
            init_decay_min, init_decay_max, use_gate,
        )
        self.head = PredictionHead(embed_dim, head_hidden_dim)

    def init_state(self, batch_size: int, device, dtype=torch.float32):
        return self.ssm.init_state(batch_size, device, dtype)

    def step(self, e_t: torch.Tensor, states):
        """Single-frame streaming update.

        e_t: [B, embed_dim] current frame embedding.
        Returns (e_hat_next [B, embed_dim] prediction for e_{t+1},
                 new_states, effective_decay [B, state_dim] of the last layer).
        """
        y_t, new_states, decays = self.ssm.step(e_t, states)
        e_hat_next = self.head(y_t)
        return e_hat_next, new_states, decays[-1]

    def forward(self, e_seq: torch.Tensor, states=None):
        """e_seq: [B, T, embed_dim].
        Returns predictions [B, T, embed_dim] (predictions[:, t] predicts
        e_seq[:, t+1]; the last position has no target and should be
        dropped by the caller), final_states, and the effective-decay
        sequence [B, T, state_dim] of the last layer for theory analysis.
        """
        y_seq, final_states, decay_seq = self.ssm(e_seq, states)
        predictions = self.head(y_seq)
        return predictions, final_states, decay_seq

    @staticmethod
    def prediction_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-timestep L2 prediction error. pred/target: [..., embed_dim] -> [...]"""
        return torch.linalg.vector_norm(pred - target, dim=-1)
