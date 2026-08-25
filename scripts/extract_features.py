#!/usr/bin/env python
"""Precompute frozen-backbone embeddings for every clip in a dataset and
cache them to disk as one .pt tensor per clip: [num_frames, embed_dim].

This is the main lever for fast iteration on a laptop: the backbone is
frozen, so there is no reason to re-run its forward pass every training
epoch. Run this once per dataset/backbone combination, then point
train.py / evaluate.py at the cached features via `paths.feature_cache`.

Usage:
    python scripts/extract_features.py --config configs/ped2.yaml
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets import list_clips
from src.data.transforms import get_transform
from src.models.backbone import FrozenBackbone
from src.utils import apply_overrides, get_device, load_config


@torch.no_grad()
def extract_split(backbone, transform, root, split, out_dir, device, batch_size=64):
    clips = list_clips(root, split)
    out_dir.mkdir(parents=True, exist_ok=True)

    for clip_id, frame_paths in tqdm(clips.items(), desc=f"{split}"):
        out_path = out_dir / f"{clip_id}.pt"
        if out_path.exists():
            continue

        feats = []
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i : i + batch_size]
            imgs = torch.stack(
                [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            ).to(device)
            feats.append(backbone(imgs).cpu())
        feats = torch.cat(feats, dim=0)  # [N, embed_dim]
        torch.save(feats, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--override", nargs="+", default=[],
                         help='e.g. --override backbone.name=dinov2_vits14 backbone.embed_dim=384')
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.override)
    device = get_device(cfg["train"]["device"])
    print(f"Using device: {device}")

    backbone = FrozenBackbone(
        name=cfg["backbone"]["name"], pretrained=cfg["backbone"]["pretrained"]
    ).to(device)
    transform = get_transform(cfg["backbone"]["image_size"])

    root = cfg["dataset"]["root"]
    out_root = Path(cfg["paths"]["feature_cache"])

    extract_split(backbone, transform, root, "train", out_root / "train", device, args.batch_size)
    extract_split(backbone, transform, root, "test", out_root / "test", device, args.batch_size)
    print(f"Cached features under {out_root}")


if __name__ == "__main__":
    main()
