#!/usr/bin/env python
"""Normalize a raw downloaded VAD dataset into the layout the rest of this
codebase expects:

    <out>/train/frames/<clip_id>/000001.jpg ...
    <out>/test/frames/<clip_id>/000001.jpg ...
    <out>/test/gt/<clip_id>.npy        # 1D uint8, 1 = anomalous frame

CAVEAT: UCSD/Avenue/ShanghaiTech have been re-hosted by many different
mirrors over the years with slightly different internal layouts (frames as
.tif vs .jpg, ground truth as per-pixel masks vs frame-index text files vs
already-frame-level .npy, videos as .avi that need decoding vs already
extracted frames, etc). The converters below cover the most common layout
for each dataset as of 2025 mirrors, but you may need to tweak the
`--raw-*` glob patterns for whatever mirror you actually downloaded from —
run with --dry-run first and inspect the printed mapping before writing
anything.

Usage:
    python scripts/prepare_dataset.py --dataset ped2 \
        --raw-root /path/to/downloaded/UCSDped2 \
        --out data/UCSDped2

    python scripts/prepare_dataset.py --dataset avenue \
        --raw-root /path/to/downloaded/Avenue \
        --out data/Avenue

    python scripts/prepare_dataset.py --dataset shanghaitech \
        --raw-root /path/to/downloaded/ShanghaiTech \
        --out data/ShanghaiTech
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _save_frame(img: Image.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path, quality=95)


def _frame_label_from_mask(mask: np.ndarray) -> int:
    return int(np.any(mask > 0))


# ---------------------------------------------------------------- UCSD Ped2

def prepare_ped2(raw_root: Path, out_root: Path, dry_run: bool):
    train_dirs = sorted((raw_root / "Train").glob("Train[0-9][0-9][0-9]"))
    test_dirs = sorted((raw_root / "Test").glob("Test[0-9][0-9][0-9]"))
    test_dirs = [d for d in test_dirs if not d.name.endswith("_gt")]

    print(f"found {len(train_dirs)} train clips, {len(test_dirs)} test clips")
    if dry_run:
        return

    for clip_dir in tqdm(train_dirs, desc="ped2/train"):
        frames = sorted(clip_dir.glob("*.tif")) or sorted(clip_dir.glob("*.jpg"))
        for i, fp in enumerate(frames):
            _save_frame(Image.open(fp), out_root / "train" / "frames" / clip_dir.name / f"{i:06d}.jpg")

    for clip_dir in tqdm(test_dirs, desc="ped2/test"):
        frames = sorted(clip_dir.glob("*.tif")) or sorted(clip_dir.glob("*.jpg"))
        for i, fp in enumerate(frames):
            _save_frame(Image.open(fp), out_root / "test" / "frames" / clip_dir.name / f"{i:06d}.jpg")

        gt_dir = raw_root / "Test" / f"{clip_dir.name}_gt"
        if gt_dir.exists():
            masks = sorted(gt_dir.glob("*.bmp")) or sorted(gt_dir.glob("*.png"))
            labels = np.array(
                [_frame_label_from_mask(np.array(Image.open(m).convert("L"))) for m in masks],
                dtype=np.uint8,
            )
            gt_out = out_root / "test" / "gt" / f"{clip_dir.name}.npy"
            gt_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(gt_out, labels)
        else:
            print(f"  WARNING: no per-pixel gt dir for {clip_dir.name}, "
                  f"expected {gt_dir} — check your mirror's ground-truth format.")


# --------------------------------------------------------------- CUHK Avenue

def prepare_avenue(raw_root: Path, out_root: Path, dry_run: bool):
    import cv2
    from scipy.io import loadmat

    train_videos = sorted((raw_root / "training_videos").glob("*.avi"))
    test_videos = sorted((raw_root / "testing_videos").glob("*.avi"))
    gt_dir = raw_root / "ground_truth_demo" / "testing_label_mask"
    if not gt_dir.exists():
        gt_dir = raw_root / "testing_label_mask"

    print(f"found {len(train_videos)} train videos, {len(test_videos)} test videos, "
          f"gt_dir={'found' if gt_dir.exists() else 'MISSING: ' + str(gt_dir)}")
    if dry_run:
        return

    def extract(video_path: Path, out_dir: Path):
        cap = cv2.VideoCapture(str(video_path))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            _save_frame(img, out_dir / f"{i:06d}.jpg")
            i += 1
        cap.release()
        return i

    for vp in tqdm(train_videos, desc="avenue/train"):
        clip_id = vp.stem
        extract(vp, out_root / "train" / "frames" / clip_id)

    for vp in tqdm(test_videos, desc="avenue/test"):
        clip_id = vp.stem
        n_frames = extract(vp, out_root / "test" / "frames" / clip_id)

        # mask filenames are not necessarily zero-padded the same way as the
        # video filenames (e.g. video "01.avi" -> mask "1_label.mat" in the
        # official Avenue ground-truth release), so try both.
        candidates = [gt_dir / f"{clip_id}_label.mat"]
        if clip_id.isdigit():
            candidates.append(gt_dir / f"{int(clip_id)}_label.mat")
        mat_path = next((p for p in candidates if p.exists()), candidates[0])
        if mat_path.exists():
            mat = loadmat(mat_path)
            volLabel = mat["volLabel"][0]  # per-frame pixel masks, common Avenue gt format
            labels = np.array(
                [_frame_label_from_mask(volLabel[i]) for i in range(min(n_frames, len(volLabel)))],
                dtype=np.uint8,
            )
            gt_out = out_root / "test" / "gt" / f"{clip_id}.npy"
            gt_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(gt_out, labels)
        else:
            print(f"  WARNING: no gt .mat for {clip_id}, expected {mat_path} — "
                  f"check your mirror's ground-truth format.")


# ----------------------------------------------------------- ShanghaiTech

def prepare_shanghaitech(raw_root: Path, out_root: Path, dry_run: bool):
    train_src = raw_root / "training" / "frames"
    test_src = raw_root / "testing" / "frames"
    gt_src = raw_root / "testing" / "test_frame_mask"

    train_clips = sorted(train_src.iterdir()) if train_src.exists() else []
    test_clips = sorted(test_src.iterdir()) if test_src.exists() else []
    print(f"found {len(train_clips)} train clips, {len(test_clips)} test clips, "
          f"gt_dir={'found' if gt_src.exists() else 'MISSING: ' + str(gt_src)}")
    if dry_run:
        return

    for clip_dir in tqdm(train_clips, desc="shanghaitech/train"):
        dst = out_root / "train" / "frames" / clip_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for fp in sorted(clip_dir.glob("*")):
            shutil.copy2(fp, dst / fp.name)

    for clip_dir in tqdm(test_clips, desc="shanghaitech/test"):
        dst = out_root / "test" / "frames" / clip_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for fp in sorted(clip_dir.glob("*")):
            shutil.copy2(fp, dst / fp.name)

        gt_path = gt_src / f"{clip_dir.name}.npy"
        if gt_path.exists():
            labels = np.load(gt_path).astype(np.uint8)
            gt_out = out_root / "test" / "gt" / f"{clip_dir.name}.npy"
            gt_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(gt_out, labels)
        else:
            print(f"  WARNING: no gt npy for {clip_dir.name}, expected {gt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ped2", "avenue", "shanghaitech"])
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="only print what would be done")
    args = parser.parse_args()

    if args.dataset == "ped2":
        prepare_ped2(args.raw_root, args.out, args.dry_run)
    elif args.dataset == "avenue":
        prepare_avenue(args.raw_root, args.out, args.dry_run)
    elif args.dataset == "shanghaitech":
        prepare_shanghaitech(args.raw_root, args.out, args.dry_run)


if __name__ == "__main__":
    main()
