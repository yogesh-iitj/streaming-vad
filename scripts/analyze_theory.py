#!/usr/bin/env python
"""Validate the decay/detection-delay theoretical analysis: compares the
predicted settling-time bound (derived from a first-order linear system
view of the SSM's per-channel decay) against the empirically measured
detection delay on saved per-clip anomaly score curves.

Theory sketch (see paper/sections/theory.tex for the full derivation): for
a channel with effective decay `a` responding to a step change in its
driving input, the time to settle within a fraction `eps` of the new
steady state is

    delay_theory(a, eps) = ln(eps) / ln(a)

This script reports, per test clip with a labeled anomaly onset, the
empirical delay (frames between onset and score threshold-crossing, from
src.metrics.detection_delay) next to delay_theory computed from the
trained model's mean learned base decay, and writes results/<dataset>/
theory_analysis.json. Run once per dataset (it does NOT touch
paper/tables/ — that would make each dataset's run clobber the previous
one's row). Once you have theory_analysis.json for every dataset you want
in the paper, run scripts/make_theory_table.py against all of them
together to (re)generate paper/tables/theory_table.tex.

Requires evaluate.py to have been run with --save-scores first.

Usage:
    python scripts/analyze_theory.py --config configs/ped2.yaml \
        --checkpoint checkpoints/ped2/latest.pt --threshold 0.5
    python scripts/analyze_theory.py --config configs/avenue.yaml \
        --checkpoint checkpoints/avenue/latest.pt --threshold 0.5
    python scripts/make_theory_table.py --results \
        results/ped2/theory_analysis.json results/avenue/theory_analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets import load_gt
from src.metrics import detection_delay
from src.train import build_model
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=0.05,
                         help="settling fraction for the theoretical bound")
    parser.add_argument("--tag", default=None,
                         help="must match the --tag used at training/eval time, if any")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # architecture from the checkpoint's own config, not --config, so this
    # stays correct for ablation runs trained with --override (see
    # evaluate.py for the same reasoning)
    model = build_model(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.tag:
        cfg["paths"]["results_dir"] = f"{cfg['paths']['results_dir']}_{args.tag}"

    mean_decays = [layer.base_decay.mean().item() for layer in model.ssm.layers]
    delay_theory = {
        f"layer_{i}": math.log(args.eps) / math.log(a) for i, a in enumerate(mean_decays)
    }
    print(f"learned mean base decay per layer: {mean_decays}")
    print(f"theoretical settling delay (eps={args.eps}): {delay_theory}")

    scores_dir = Path(cfg["paths"]["results_dir"]) / "scores"
    if not scores_dir.exists():
        print(f"no saved score curves at {scores_dir}; re-run evaluate.py with --save-scores")
        return

    root = cfg["dataset"]["root"]
    rows = []
    for score_path in sorted(scores_dir.glob("*.npy")):
        clip_id = score_path.stem
        gt = load_gt(root, clip_id)
        if gt is None:
            continue
        scores = np.load(score_path)
        delay = detection_delay(scores, gt, args.threshold)
        if delay is not None:
            rows.append({"clip_id": clip_id, "empirical_delay_frames": delay})

    if rows:
        empirical_mean = float(np.mean([r["empirical_delay_frames"] for r in rows]))
    else:
        empirical_mean = None

    summary = {
        "dataset": cfg["dataset"]["name"],
        "threshold": args.threshold,
        "eps": args.eps,
        "mean_base_decay_per_layer": mean_decays,
        "theoretical_settling_delay_frames": delay_theory,
        "empirical_mean_delay_frames": empirical_mean,
        "num_clips_with_delay": len(rows),
        "per_clip": rows,
    }

    results_dir = Path(cfg["paths"]["results_dir"])
    out_json = results_dir / "theory_analysis.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out_json}")
    print("run scripts/make_theory_table.py against one or more theory_analysis.json "
          "files (one per dataset) to (re)generate paper/tables/theory_table.tex")


if __name__ == "__main__":
    main()
