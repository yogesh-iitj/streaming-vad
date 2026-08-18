"""Shared utilities: config loading (with single-level `extends`), device
selection, and seeding."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a single `extends: base.yaml` key
    relative to the configs/ directory."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    parent_name = cfg.pop("extends", None)
    if parent_name is not None:
        parent_path = path.parent / parent_name
        parent_cfg = load_config(parent_path)
        cfg = _deep_update(parent_cfg, cfg)

    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply CLI overrides of the form "a.b.c=value" (dotted key path,
    YAML-parsed value) to a loaded config, in place. Used for ablation
    sweeps (decay range, state size, gate on/off, ...) so they don't each
    need a hand-written near-duplicate YAML file."""
    for item in overrides or []:
        key, _, raw_value = item.partition("=")
        if not _:
            raise ValueError(f"--override expects key=value, got {item!r}")
        value = yaml.safe_load(raw_value)
        parts = key.strip().split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return cfg


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
