#!/usr/bin/env python
"""Render a short video of one test clip: the raw frames alongside the
model's real (already computed) streaming anomaly score, with the
ground-truth anomalous region shaded for reference. Uses whatever is
already on disk, no re-inference, so what's shown is exactly what
evaluate.py measured, not a cherry-picked replay.

Requires evaluate.py --save-scores to have been run first for the given
dataset (produces results/<dataset>/scores/<clip_id>.npy) and ffmpeg to
be on PATH.

Usage:
    python scripts/make_qualitative_video.py --dataset ped2 --clip Test001
    python scripts/make_qualitative_video.py --dataset avenue --clip 19
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets import list_clips, load_gt
from src.metrics import normalize_scores
from src.utils import load_config

FFMPEG = shutil.which("ffmpeg") or str(Path.home() / "bin" / "ffmpeg")


def encode_mp4(png_glob: str, fps: float, out_path: Path):
    subprocess.run(
        [FFMPEG, "-y", "-framerate", str(fps), "-i", png_glob,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=640:-2",
         "-movflags", "+faststart", str(out_path)],
        check=True, capture_output=True,
    )


def encode_gif(png_glob: str, fps: float, out_path: Path, width: int = 360, gif_fps: float = 8):
    # GIFs render inline and autoplay on GitHub with no click needed, unlike
    # <video>; a two-pass palette gives much better quality per byte than a
    # single-pass GIF. Downsampled fps/width keep file size reasonable.
    with tempfile.TemporaryDirectory() as tmp:
        palette = Path(tmp) / "palette.png"
        vf = f"fps={gif_fps},scale={width}:-1:flags=lanczos"
        subprocess.run(
            [FFMPEG, "-y", "-framerate", str(fps), "-i", png_glob,
             "-vf", f"{vf},palettegen", str(palette)],
            check=True, capture_output=True,
        )
        subprocess.run(
            [FFMPEG, "-y", "-framerate", str(fps), "-i", png_glob, "-i", str(palette),
             "-filter_complex", f"{vf}[x];[x][1:v]paletteuse", str(out_path)],
            check=True, capture_output=True,
        )


def render_clip(dataset: str, clip_id: str, out_path: Path, fps: float, threshold: float,
                 fmt: str):
    cfg = load_config(f"configs/{dataset}.yaml")
    root = cfg["dataset"]["root"]

    clips = list_clips(root, "test")
    if clip_id not in clips:
        raise SystemExit(f"clip {clip_id!r} not found in {root}/test/frames; "
                          f"available: {sorted(clips)[:10]}...")
    frame_paths = clips[clip_id]

    score_path = Path(cfg["paths"]["results_dir"]) / "scores" / f"{clip_id}.npy"
    if not score_path.exists():
        raise SystemExit(f"{score_path} not found; run "
                          f"`python -m src.evaluate --config configs/{dataset}.yaml "
                          f"--checkpoint checkpoints/{dataset}/latest.pt --save-scores` first")
    scores = normalize_scores(np.load(score_path))
    gt = load_gt(root, clip_id)

    n = min(len(frame_paths), len(scores))
    frame_paths, scores = frame_paths[:n], scores[:n]
    gt = gt[:n] if gt is not None else np.zeros(n, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for t in range(n):
            fig, (ax_img, ax_score) = plt.subplots(
                2, 1, figsize=(5.5, 5.2), gridspec_kw={"height_ratios": [3, 1]}
            )

            img = Image.open(frame_paths[t]).convert("RGB")
            ax_img.imshow(img)
            ax_img.axis("off")
            border_color = "#d62728" if gt[t] else "#2ca02c"
            ax_img.add_patch(
                plt.Rectangle(
                    (0, 0), img.width - 1, img.height - 1, fill=False,
                    edgecolor=border_color, linewidth=6, transform=ax_img.transData,
                )
            )
            label = "ANOMALOUS (ground truth)" if gt[t] else "normal (ground truth)"
            ax_img.set_title(f"{dataset} / {clip_id}  frame {t+1}/{n}   {label}",
                              fontsize=9, color=border_color)

            anomalous_run = np.where(gt > 0)[0]
            if len(anomalous_run):
                ax_score.axvspan(anomalous_run.min(), anomalous_run.max(),
                                  color="#d62728", alpha=0.15, label="GT anomalous region")
            ax_score.plot(np.arange(t + 1), scores[: t + 1], color="#1f77b4", linewidth=1.2)
            ax_score.axhline(threshold, color="gray", linestyle="--", linewidth=0.8,
                              label=f"threshold={threshold}")
            ax_score.axvline(t, color="black", linewidth=0.8)
            ax_score.set_xlim(0, n)
            ax_score.set_ylim(-0.05, 1.05)
            ax_score.set_xlabel("frame", fontsize=8)
            ax_score.set_ylabel("anomaly score\n(normalized)", fontsize=8)
            ax_score.tick_params(labelsize=7)
            ax_score.legend(fontsize=6, loc="upper right")

            fig.tight_layout()
            fig.savefig(tmp_dir / f"frame_{t:05d}.png", dpi=110)
            plt.close(fig)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        png_glob = str(tmp_dir / "frame_%05d.png")
        written = []
        if fmt in ("mp4", "both"):
            mp4_path = out_path.with_suffix(".mp4")
            encode_mp4(png_glob, fps, mp4_path)
            written.append(mp4_path)
        if fmt in ("gif", "both"):
            gif_path = out_path.with_suffix(".gif")
            encode_gif(png_glob, fps, gif_path)
            written.append(gif_path)
    for p in written:
        print(f"wrote {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ped2", "avenue"])
    parser.add_argument("--clip", required=True)
    parser.add_argument("--fps", type=float, default=None,
                         help="defaults to the dataset's native capture fps from its config")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--format", choices=["mp4", "gif", "both"], default="both",
                         help="gif renders inline and autoplays on GitHub with no click; "
                              "mp4 is smaller and has playback controls")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(f"configs/{args.dataset}.yaml")
    fps = args.fps or cfg["dataset"]["fps"]
    out_path = Path(args.out or f"assets/qualitative/{args.dataset}_{args.clip}")

    render_clip(args.dataset, args.clip, out_path, fps, args.threshold, args.format)


if __name__ == "__main__":
    main()
