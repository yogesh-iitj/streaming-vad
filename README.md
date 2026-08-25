# Streaming Video Anomaly Detection with a Causal State-Space Core

A strictly-causal, O(1)-per-frame streaming video anomaly detector: a
frozen visual backbone feeds a lightweight causal diagonal state-space
(SSM) temporal core with a learned "event-boundary" decay gate, trained
self-supervised via next-frame-embedding prediction. Designed to run in
real time on commodity edge hardware (developed and benchmarked on an
Apple M3 Pro, MPS backend — no CUDA dependency).

## Why this exists / positioning

2025 saw several papers apply Mamba/SSMs to video anomaly detection
(VADMamba, ICME'25; Wave-MambaAD, ICCV'25; StreamVAD-style key-clip
methods) — so "SSM for VAD" alone is no longer novel. None of them,
however, provide:

1. A **formal definition of strict streaming** — O(1) state, zero
   lookahead, no clip/window buffering (the prior work still buffers
   clips internally).
2. Any **theoretical analysis** connecting the SSM's decay spectrum to
   detection delay / minimum-detectable event duration.
3. A **real edge-hardware deployment story** — measured latency, memory,
   and throughput on actual consumer silicon, not just GPU throughput
   claims.

This codebase is built around those three gaps: `src/train.py` /
`src/evaluate.py` produce the standard accuracy numbers, and
`scripts/analyze_theory.py` + the latency instrumentation in
`evaluate.py` produce the theory-validation and deployment evidence that
differentiate this from the existing 2025 SSM-VAD literature. The full
derivation (Proposition 1: a closed-form settling-delay bound from the
recurrence's decay spectrum) and positioning against prior work are
written up in an accompanying paper manuscript, kept separately from
this repo since it has its own review/revision lifecycle.

## Status

The pipeline has been run end-to-end on real data for UCSD Ped2 and CUHK
Avenue (download → prepare → extract features → train → evaluate →
theory validation). Current numbers, with default, untuned
hyperparameters (`configs/*.yaml` as committed, ResNet-18 backbone, no
sweep run yet):

| Dataset | Frame-AUC | EER | Latency (M3 Pro, mps) | FPS |
|---|---|---|---|---|
| UCSD Ped2 | 67.9% | 36.8% | 0.72 ms | ~1382 |
| CUHK Avenue | 70.2% | 34.9% | 0.77 ms | ~1291 |

**These AUCs are well below published SOTA on these benchmarks (~95–99%
on Ped2).** That's expected from an untuned first pass with vanilla
frozen ResNet-18 features (no motion signal, no fine-tuning) — it
demonstrates the pipeline is correct end-to-end, not that the method is
competitive yet.

The theory-validation numbers are a genuine, interesting finding as-is:
the fixed-decay bound (Eq. `\ref{eq:delay-bound}` in `theory.tex`)
predicts ~58 frames of settling delay from the learned base decay alone,
but empirical detection delay is far shorter — 1.6 frames on Ped2, 18.4
on Avenue.

**Ablations (decay range, state size, gate on/off) have also been run on
both datasets** (`scripts/make_ablation_table.py`) and turned up a real,
somewhat surprising finding: the event-boundary gate's effect on accuracy
**flips sign between datasets**. On Ped2 (16 training clips, 120-180
frames each) disabling the gate *improves* frame-AUC (76.6% vs. 67.9%
gated) and a smaller state also wins — pointing at overfitting on a
small training set. On Avenue (16 clips, but up to 1271 frames each —
much more total training data) disabling the gate *hurts* sharply (60.0%
vs. 70.2% gated), and a larger state helps slightly. Read together this
looks like a capacity/data-size interaction rather than the gate being
fundamentally unhelpful, but it's a two-dataset hypothesis. We also
tried combining each dataset's two individually-helpful changes
(gate-off + smaller state on Ped2, larger state + slower decay on
Avenue); neither combination beats its best individual component
(67.8% and 71.1% respectively, both roughly back to baseline or
single-factor levels), so the effects don't simply stack. A third,
larger dataset (ShanghaiTech, in progress) is the natural next check.

Ablation infra: `train.py`/`evaluate.py`/`analyze_theory.py` now accept
`--override key.path=value` (for sweeps, so ablations don't need a
hand-written YAML each) and `--tag` (keeps each ablation run's
checkpoints/results in a separate directory rather than overwriting the
baseline). `evaluate.py` and `analyze_theory.py` rebuild the model
architecture from the checkpoint's own saved config rather than
`--config`, specifically so this stays correct when `--config` and the
checkpoint's actual (overridden) architecture differ.

ShanghaiTech is not yet run (larger dataset, needs a manual OneDrive
grab — see `scripts/download_shanghaitech.sh`).

RBDC/TBDC in `src/metrics.py` remains an explicitly simplified stand-in
(see its docstring) — use the official evaluation toolkit for numbers
that go in the paper; frame-level AUC/EER here follow the standard
protocol and are fine to report directly.

## Setup

```bash
cd streaming-vad
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python >= 3.10. PyTorch will use the MPS backend automatically
on Apple Silicon (`src/utils.get_device` picks `mps` > `cuda` > `cpu`).

## Directory structure

```
streaming-vad/
├── configs/                  # base.yaml + one override per dataset
├── src/
│   ├── data/datasets.py       # VADClipWindowDataset (train), VADFullClipDataset (eval)
│   ├── data/transforms.py
│   ├── models/backbone.py     # frozen resnet18 / dinov2_vits14
│   ├── models/ssm.py          # causal diagonal SSM + event-boundary gate (the core contribution)
│   ├── models/detector.py     # SSM stack + next-embedding prediction head
│   ├── train.py
│   ├── evaluate.py            # streaming (frame-by-frame) eval + latency benchmarking
│   ├── metrics.py             # frame-AUC, EER, detection-delay, simplified RBDC
│   └── utils.py                # config loading, device/seed helpers
├── scripts/
│   ├── download_ped2.sh / download_avenue.sh / download_shanghaitech.sh
│   ├── prepare_dataset.py      # raw dataset -> normalized frame/gt layout
│   ├── extract_features.py     # precompute + cache frozen backbone features
│   ├── analyze_theory.py       # decay-spectrum vs detection-delay validation
│   ├── make_latex_table.py     # eval_results.json -> a LaTeX results-table fragment
│   ├── make_theory_table.py    # theory_analysis.json -> a LaTeX theory-table fragment
│   ├── make_ablation_table.py  # ablation eval_results.json's -> a LaTeX ablation-table fragment
│   └── make_qualitative_video.py  # frame + live score + ground truth -> a short mp4
├── assets/qualitative/         # short demo videos (see "Qualitative results" below)
├── data/                      # (gitignored) prepared datasets go here
├── features_cache/            # (gitignored) cached embeddings from extract_features.py
├── checkpoints/               # (gitignored) trained model weights
└── results/                   # (gitignored) eval_results.json, score curves, LaTeX-ready numbers
```

## Dataset preparation

Three standard unsupervised VAD benchmarks are wired up: **UCSD Ped2**,
**CUHK Avenue**, **ShanghaiTech**. All three get normalized into the same
layout:

```
data/<Dataset>/train/frames/<clip_id>/000001.jpg ...
data/<Dataset>/test/frames/<clip_id>/000001.jpg ...
data/<Dataset>/test/gt/<clip_id>.npy      # 1D uint8, 1 = anomalous frame
```

1. Download the raw dataset (see `scripts/download_*.sh` — each documents
   its official/canonical source and approximate size; run manually,
   these do **not** auto-execute):
   ```bash
   bash scripts/download_ped2.sh
   ```
2. Normalize it:
   ```bash
   python scripts/prepare_dataset.py --dataset ped2 \
       --raw-root data/raw/UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2 \
       --out data/UCSDped2 --dry-run   # inspect the mapping first
   # then drop --dry-run to actually write files
   ```
   Mirrors of these datasets vary in internal layout — `--dry-run` prints
   what it found before writing anything, and the script's docstring
   notes what to adjust if your mirror differs.

## Usage

```bash
# 1. Cache frozen backbone features (do this once per dataset+backbone;
#    makes every subsequent train/eval run much faster on a laptop GPU)
python scripts/extract_features.py --config configs/ped2.yaml

# 2. Train the SSM head (backbone stays frozen)
python -m src.train --config configs/ped2.yaml

# 3. Evaluate: streaming frame-by-frame inference, frame-AUC/EER,
#    and per-frame latency on the current device
python -m src.evaluate --config configs/ped2.yaml \
    --checkpoint checkpoints/ped2/latest.pt --save-scores

# 4. Validate the decay/detection-delay theory against the trained model
#    (writes results/<dataset>/theory_analysis.json — does NOT touch
#    paper/tables/, so running this for one dataset never clobbers
#    another dataset's numbers)
python scripts/analyze_theory.py --config configs/ped2.yaml \
    --checkpoint checkpoints/ped2/latest.pt --threshold 0.5
```

Repeat steps 1–4 for every dataset you want results for
(`configs/avenue.yaml`, `configs/shanghaitech.yaml`), then combine
everything into LaTeX table fragments (useful if you're writing this up
externally; default output goes to `paper/tables/*.tex`, a local,
gitignored path, override with `--out` if you want it elsewhere):

```bash
python scripts/make_latex_table.py --results \
    results/ped2/eval_results.json \
    results/avenue/eval_results.json \
    results/shanghaitech/eval_results.json

python scripts/make_theory_table.py --results \
    results/ped2/theory_analysis.json \
    results/avenue/theory_analysis.json \
    results/shanghaitech/theory_analysis.json
```

Re-run both combiner scripts (with whichever datasets are ready) any time
you re-run `evaluate.py` or `analyze_theory.py` for any one dataset.

## Reproducing the edge-latency numbers

`src/evaluate.py` measures per-frame wall-clock latency of `model.step()`
directly (with a warmup period, and `torch.mps.synchronize()` /
`torch.cuda.synchronize()` before each timing boundary so numbers aren't
just async-dispatch noise). To also report CoreML/ANE numbers (stronger
edge-deployment evidence than MPS alone), export the trained model with
`coremltools` and benchmark separately — not yet automated here.

## Key design decisions (and why)

- **No `mamba-ssm` dependency.** Its selective-scan kernel is CUDA-only
  and won't run on Apple Silicon. `src/models/ssm.py` implements a
  diagonal linear recurrence with the same O(1)-per-step behavior in
  plain PyTorch.
- **One code path for training and streaming inference.** `forward()` is
  literally `step()` called in a loop, not a separate parallel-scan
  formulation — this removes any risk of the "efficient training
  formulation" silently diverging from what actually runs at deployment,
  which is the whole point of the strict-streaming claim.
- **Backbone frozen, feature-cached.** The only thing trained is the SSM
  head — keeps the whole pipeline tractable on a laptop GPU without
  sacrificing the ability to swap in a stronger backbone (DINOv2) later.

## Qualitative results

Two short clips showing the model's real, unedited streaming output:
the raw test frame, a green/red border for the ground-truth label at
that frame, and the actual (already computed, not replayed after the
fact) normalized anomaly score scrolling underneath with the true
anomalous region shaded. These are exactly the runs behind the numbers
above, not cherry-picked to look better than the reported 67.9%/70.2%
AUC — the Ped2 clip in particular shows real false positives during the
labeled-normal region, which is honest given where the model currently
sits.

<video src="https://raw.githubusercontent.com/yogesh-iitj/streaming-vad/main/assets/qualitative/ped2_Test001.mp4" controls width="480"></video>

UCSD Ped2, `Test001` (180 frames @ 10 fps).

<video src="https://raw.githubusercontent.com/yogesh-iitj/streaming-vad/main/assets/qualitative/avenue_19.mp4" controls width="480"></video>

CUHK Avenue, clip `19` (248 frames @ 25 fps).

Regenerate these, or render a different clip, with:

```bash
python scripts/make_qualitative_video.py --dataset ped2 --clip Test001
python scripts/make_qualitative_video.py --dataset avenue --clip 19
```

Requires `evaluate.py --save-scores` to have been run for that dataset
first (see Usage above), and `ffmpeg` on `PATH`.
