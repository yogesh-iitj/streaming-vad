"""Causal diagonal state-space core with an input/state-dependent
("event-boundary") decay gate.

This is the paper's central algorithmic + theoretical object. Design
choices, and why:

* Diagonal, real-valued recurrence (S4D/S5-style), not a full Mamba
  reimplementation: `mamba-ssm`'s selective-scan kernel is CUDA-only and
  will not run on Apple Silicon (MPS). A diagonal linear recurrence gives
  the same asymptotic O(1)-per-step behavior and is trivial to implement
  correctly on MPS/CPU.

* `step()` is the ONLY recurrence primitive. Training (`forward`) is
  implemented by literally calling `step()` once per timestep, rather than
  a separate parallel-scan formulation. This is deliberate: the paper's
  claim is strict causal streaming with O(1) state, and reviewers will
  (rightly) ask whether the "efficient" training-time formulation actually
  matches what runs at deployment. Using one code path for both removes
  that entire class of bug/criticism. For the clip lengths used here
  (<=64 frames) the sequential loop is not a bottleneck on a laptop GPU.

* The decay gate is what lets us later derive and validate a theorem
  relating decay spectrum -> detection delay -> minimum-detectable event
  duration (see paper/sections/theory.tex and scripts/analyze_theory.py).
  It is intentionally simple (an MLP over [x_t, s_{t-1}]) so the theory
  stays tractable; architectural cleverness is not the point here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EventBoundaryGate(nn.Module):
    """Produces a per-channel multiplier in (0, 1) that scales the layer's
    base decay rate at each timestep, conditioned on the current input and
    the previous state. Low gate value = fast forgetting / state reset
    (useful right at an anomaly onset); high gate value = slow, stable
    memory (useful during a steady normal or steady anomalous period)."""

    def __init__(self, embed_dim: int, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x_t: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([x_t, s_prev], dim=-1)))


class CausalDiagonalSSMLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        state_dim: int,
        gate_hidden_dim: int = 64,
        init_decay_min: float = 0.9,
        init_decay_max: float = 0.999,
        use_gate: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.use_gate = use_gate

        self.norm = nn.LayerNorm(embed_dim)
        self.w_in = nn.Linear(embed_dim, state_dim)
        self.w_out = nn.Linear(state_dim, embed_dim)
        # Instantiated even when use_gate=False so checkpoints keep a
        # consistent parameter set across the ablation; simply unused in
        # that mode (see step()).
        self.gate = EventBoundaryGate(embed_dim, state_dim, gate_hidden_dim)

        # base (time-invariant) decay per channel, constrained to
        # (init_decay_min, init_decay_max) via a sigmoid reparameterization,
        # following S4D-style initialization.
        self._decay_min = init_decay_min
        self._decay_max = init_decay_max
        init = torch.empty(state_dim).uniform_(-2.0, 2.0)
        self.raw_base_decay = nn.Parameter(init)

    @property
    def base_decay(self) -> torch.Tensor:
        span = self._decay_max - self._decay_min
        return self._decay_min + span * torch.sigmoid(self.raw_base_decay)

    def init_state(self, batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)

    def step(
        self, x_t: torch.Tensor, s_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One causal update.

        x_t:    [B, embed_dim]  current frame embedding (post input-norm)
        s_prev: [B, state_dim]  previous state
        returns (y_t [B, embed_dim], s_t [B, state_dim], effective_decay [B, state_dim])
        """
        x_n = self.norm(x_t)
        u_t = self.w_in(x_n)
        if self.use_gate:
            gate_t = self.gate(x_n, s_prev)
            effective_decay = self.base_decay.unsqueeze(0) * gate_t  # [B, state_dim] in (0,1)
        else:
            # ablation: fixed decay, no input/state-dependent modulation
            effective_decay = self.base_decay.unsqueeze(0).expand(x_t.shape[0], -1)

        s_t = effective_decay * s_prev + u_t
        y_t = self.w_out(s_t) + x_t  # residual back to embed space
        return y_t, s_t, effective_decay

    def forward(
        self, x_seq: torch.Tensor, s_prev: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x_seq: [B, T, embed_dim] -> y_seq: [B, T, embed_dim], final state,
        and effective_decay_seq: [B, T, state_dim] (kept for the decay/delay
        theory analysis)."""
        B, T, _ = x_seq.shape
        if s_prev is None:
            s_prev = self.init_state(B, x_seq.device, x_seq.dtype)

        ys, decays = [], []
        s_t = s_prev
        for t in range(T):
            y_t, s_t, eff_decay = self.step(x_seq[:, t], s_t)
            ys.append(y_t)
            decays.append(eff_decay)

        return torch.stack(ys, dim=1), s_t, torch.stack(decays, dim=1)


class CausalSSMStack(nn.Module):
    """Stack of `num_layers` CausalDiagonalSSMLayer with residual streams."""

    def __init__(
        self,
        embed_dim: int,
        state_dim: int,
        num_layers: int = 2,
        gate_hidden_dim: int = 64,
        init_decay_min: float = 0.9,
        init_decay_max: float = 0.999,
        use_gate: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CausalDiagonalSSMLayer(
                    embed_dim, state_dim, gate_hidden_dim, init_decay_min, init_decay_max, use_gate
                )
                for _ in range(num_layers)
            ]
        )

    def init_state(self, batch_size: int, device, dtype=torch.float32) -> list[torch.Tensor]:
        return [layer.init_state(batch_size, device, dtype) for layer in self.layers]

    def step(
        self, x_t: torch.Tensor, states: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        new_states, decays = [], []
        h = x_t
        for layer, s_prev in zip(self.layers, states):
            h, s_t, eff_decay = layer.step(h, s_prev)
            new_states.append(s_t)
            decays.append(eff_decay)
        return h, new_states, decays

    def forward(
        self, x_seq: torch.Tensor, states: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        B, T, _ = x_seq.shape
        if states is None:
            states = self.init_state(B, x_seq.device, x_seq.dtype)

        h = x_seq
        new_states = list(states)
        final_decays = None
        for i, (layer, s_prev) in enumerate(zip(self.layers, states)):
            h, s_t, decay_seq = layer(h, s_prev)
            new_states[i] = s_t
            final_decays = decay_seq  # decay spectrum of the last layer, used for theory analysis
        return h, new_states, final_decays
