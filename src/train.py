#!/usr/bin/env python
"""Train the causal SSM head via next-embedding prediction on normal-only
training clips.

Two modes, chosen automatically based on whether cached features exist:
  * cached-feature mode (fast — recommended): run
    scripts/extract_features.py first, then this script only trains the
    small SSM head, no backbone forward pass needed.
  * raw-frame mode (fallback): loads images and runs the frozen backbone
    inline every step. Works, but much slower on a laptop GPU.

Usage:
    python -m src.train --config configs/ped2.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.datasets import VADClipWindowDataset
from src.models.backbone import FrozenBackbone
from src.models.detector import StreamingVADModel
from src.utils import apply_overrides, get_device, load_config, set_seed


def make_lr_lambda(warmup_epochs: int, total_epochs: int):
    def fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return fn


def build_model(cfg: dict) -> StreamingVADModel:
    return StreamingVADModel(
        embed_dim=cfg["backbone"]["embed_dim"],
        state_dim=cfg["ssm"]["state_dim"],
        num_layers=cfg["ssm"]["num_layers"],
        gate_hidden_dim=cfg["ssm"]["gate_hidden_dim"],
        head_hidden_dim=cfg["head"]["hidden_dim"],
        init_decay_min=cfg["ssm"]["init_decay_min"],
        init_decay_max=cfg["ssm"]["init_decay_max"],
        use_gate=cfg["ssm"].get("use_gate", True),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="+", default=[],
                         help='dotted-key overrides for ablations, e.g. '
                              '--override ssm.state_dim=32 ssm.use_gate=false')
    parser.add_argument("--tag", default=None,
                         help="suffix appended to checkpoint_dir/results_dir so ablation "
                              "runs do not overwrite the baseline (e.g. --tag nogate)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.override)
    if args.tag:
        cfg["paths"]["checkpoint_dir"] = f"{cfg['paths']['checkpoint_dir']}_{args.tag}"
        cfg["paths"]["results_dir"] = f"{cfg['paths']['results_dir']}_{args.tag}"
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["train"]["device"])
    print(f"[train] dataset={cfg['dataset']['name']} device={device}")

    feature_dir = Path(cfg["paths"]["feature_cache"])
    use_cached = (feature_dir / "train").exists()
    backbone = None
    if not use_cached:
        print("[train] no feature cache found, running backbone inline "
              "(slower — consider scripts/extract_features.py first)")
        backbone = FrozenBackbone(
            name=cfg["backbone"]["name"], pretrained=cfg["backbone"]["pretrained"]
        ).to(device)

    dataset = VADClipWindowDataset(
        root=cfg["dataset"]["root"],
        clip_len=cfg["train"]["clip_len"],
        image_size=cfg["backbone"]["image_size"],
        feature_dir=feature_dir if use_cached else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"] if use_cached is False else 0,
        drop_last=True,
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        make_lr_lambda(cfg["train"]["warmup_epochs"], cfg["train"]["epochs"]),
    )

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    history = []
    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()

        for batch in loader:
            batch = batch.to(device)  # [B, T, C, H, W] or [B, T, D]

            if not use_cached:
                B, T = batch.shape[:2]
                imgs = batch.view(B * T, *batch.shape[2:])
                with torch.no_grad():
                    x_seq = backbone(imgs).view(B, T, -1)
            else:
                x_seq = batch

            predictions, _, _ = model(x_seq)
            pred = predictions[:, :-1]
            target = x_seq[:, 1:].detach()
            loss = torch.nn.functional.mse_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1
            if step % cfg["train"]["log_every"] == 0:
                print(f"  epoch {epoch:03d} step {step:06d} loss {loss.item():.5f}")

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - t0
        print(f"[epoch {epoch:03d}] avg_loss={avg_loss:.5f} lr={scheduler.get_last_lr()[0]:.2e} "
              f"time={elapsed:.1f}s")
        history.append({"epoch": epoch, "loss": avg_loss, "time_s": elapsed})

        if (epoch + 1) % cfg["train"]["ckpt_every"] == 0 or epoch == cfg["train"]["epochs"] - 1:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, ckpt_path)
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, ckpt_dir / "latest.pt")
            print(f"  saved checkpoint -> {ckpt_path}")

    with open(results_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
