#!/usr/bin/env python
"""Streaming evaluation: runs the model frame-by-frame (true streaming
inference, matching deployment) over every test clip, computes frame-level
AUC / EER, and measures per-frame latency on the current device — this is
the number that backs the "real-time on edge silicon" claim.

Usage:
    python -m src.evaluate --config configs/ped2.yaml --checkpoint checkpoints/ped2/latest.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import VADFullClipDataset, eval_collate
from src.metrics import equal_error_rate, frame_level_auc
from src.models.backbone import FrozenBackbone
from src.train import build_model
from src.utils import get_device, load_config


def sync(device: torch.device):
    if device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def run_streaming_clip(model, x_seq: torch.Tensor, device, measure_latency: bool = False):
    """x_seq: [N, embed_dim] single clip, in order.
    Returns per-frame anomaly scores [N] (score[0] = 0, undefined) and,
    if measure_latency, a list of per-step wall-clock latencies in ms —
    one per frame in this clip. Warmup (dropping the first few frames'
    latencies, where kernels are still being compiled/dispatched) is the
    caller's responsibility, not this function's, since it operates
    per-clip and a "warmup" defined in clips rather than frames breaks on
    datasets with few, long clips."""
    N = x_seq.shape[0]
    states = model.init_state(1, device)
    scores = np.zeros(N, dtype=np.float32)
    latencies = []

    prev_pred = None
    for t in range(N):
        e_t = x_seq[t : t + 1].to(device)

        if measure_latency:
            sync(device)
            t0 = time.perf_counter()

        pred_next, states, _ = model.step(e_t, states)

        if measure_latency:
            sync(device)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        if prev_pred is not None:
            scores[t] = model.prediction_error(prev_pred, e_t).item()
        prev_pred = pred_next

    return scores, latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                         help="used only to locate the dataset/paths; model architecture is "
                              "always taken from the checkpoint's own saved config, so this "
                              "stays correct even for ablation runs with --override")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--measure-latency", action="store_true", default=True)
    parser.add_argument("--latency-warmup", type=int, default=30,
                         help="number of leading FRAMES (not clips) across the whole eval run "
                              "to exclude from latency stats, to let MPS/CUDA kernels warm up")
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--tag", default=None,
                         help="must match the --tag used at training time, if any — "
                              "determines where results/scores are written")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # Architecture must match what was actually trained, which may differ
    # from --config if it was trained with --override; the checkpoint's
    # own saved config is the source of truth for model shape.
    model_cfg = ckpt["cfg"]
    if args.tag:
        cfg["paths"]["results_dir"] = f"{cfg['paths']['results_dir']}_{args.tag}"

    device = get_device(cfg["train"]["device"])
    print(f"[eval] dataset={cfg['dataset']['name']} device={device}")

    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    feature_dir = Path(cfg["paths"]["feature_cache"])
    use_cached = (feature_dir / "test").exists()
    backbone = None
    if not use_cached:
        backbone = FrozenBackbone(
            name=model_cfg["backbone"]["name"], pretrained=model_cfg["backbone"]["pretrained"]
        ).to(device)

    dataset = VADFullClipDataset(
        root=cfg["dataset"]["root"],
        image_size=model_cfg["backbone"]["image_size"],
        feature_dir=feature_dir if use_cached else None,
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=eval_collate)

    clip_scores, clip_labels = {}, {}
    all_latencies = []
    scores_dir = Path(cfg["paths"]["results_dir"]) / "scores"
    if args.save_scores:
        scores_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(loader):
        clip_id, data, gt = item["clip_id"], item["data"], item["gt"]

        if not use_cached:
            data = data.to(device)
            with torch.no_grad():
                x_seq = backbone(data).cpu()
        else:
            x_seq = data

        scores, latencies = run_streaming_clip(model, x_seq, device, args.measure_latency)
        clip_scores[clip_id] = scores
        if gt is not None:
            clip_labels[clip_id] = gt
        if args.measure_latency:
            all_latencies.extend(latencies)
        if args.save_scores:
            np.save(scores_dir / f"{clip_id}.npy", scores)

        print(f"  [{i+1}/{len(dataset)}] {clip_id}: {len(scores)} frames")

    missing_gt = [cid for cid in clip_scores if cid not in clip_labels]
    if missing_gt:
        print(f"[eval] WARNING: no ground truth for {len(missing_gt)} clip(s), "
              f"excluded from AUC/EER: {missing_gt[:5]}{'...' if len(missing_gt) > 5 else ''}")
        for cid in missing_gt:
            clip_scores.pop(cid)

    results = {
        "dataset": cfg["dataset"]["name"],
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "tag": args.tag,
        "ssm_state_dim": model_cfg["ssm"]["state_dim"],
        "ssm_use_gate": model_cfg["ssm"].get("use_gate", True),
        "ssm_init_decay_min": model_cfg["ssm"]["init_decay_min"],
        "ssm_init_decay_max": model_cfg["ssm"]["init_decay_max"],
        "num_test_clips": len(dataset),
        "num_scored_clips": len(clip_scores),
    }

    if clip_scores:
        results["frame_auc"] = frame_level_auc(clip_scores, clip_labels)
        results["eer"] = equal_error_rate(clip_scores, clip_labels)

    if all_latencies:
        lat = np.array(all_latencies)
        n_warmup = min(args.latency_warmup, max(0, len(lat) - 1))
        lat_steady = lat[n_warmup:]
        results["latency_warmup_frames_dropped"] = int(n_warmup)
        results["latency_ms_mean"] = float(lat_steady.mean())
        results["latency_ms_median"] = float(np.median(lat_steady))
        results["latency_ms_p95"] = float(np.percentile(lat_steady, 95))
        results["fps"] = float(1000.0 / lat_steady.mean())

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
