"""Evaluation metrics for video anomaly detection.

frame_level_auc / eer follow the standard protocol used across the VAD
literature (Liu et al. 2018 and follow-ups): per-clip min-max normalize
the anomaly score curve to [0, 1] before pooling across all test clips and
computing one dataset-level ROC-AUC.

NOTE on RBDC/TBDC: the official Region-Based and Track-Based Detection
Criteria (Ramachandra & Jones, 2020) require pixel-level anomaly masks and
object tracks, matched via IoU thresholds across frames. The
implementation below is a simplified frame-overlap approximation for quick
iteration during development ONLY. For numbers that go in the paper, use
the official evaluation toolkit released by the datasets/prior work
instead of this stub — say so explicitly in the experiments section.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-8:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def frame_level_auc(
    clip_scores: dict[str, np.ndarray], clip_labels: dict[str, np.ndarray]
) -> float:
    """clip_scores/clip_labels: clip_id -> 1D array (same length per clip).
    Scores are min-max normalized per clip, then pooled for one AUC."""
    all_scores, all_labels = [], []
    for clip_id, scores in clip_scores.items():
        labels = clip_labels[clip_id]
        assert len(scores) == len(labels), f"length mismatch for clip {clip_id}"
        all_scores.append(normalize_scores(scores))
        all_labels.append(labels)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    if all_labels.min() == all_labels.max():
        raise ValueError("Ground truth contains only one class; AUC is undefined.")
    return float(roc_auc_score(all_labels, all_scores))


def equal_error_rate(
    clip_scores: dict[str, np.ndarray], clip_labels: dict[str, np.ndarray]
) -> float:
    all_scores, all_labels = [], []
    for clip_id, scores in clip_scores.items():
        all_scores.append(normalize_scores(scores))
        all_labels.append(clip_labels[clip_id])
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def detection_delay(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> int | None:
    """Frames between the true onset of an (the first) anomalous segment
    and the first frame the (normalized) score crosses `threshold`. Used
    to empirically validate the decay/delay theoretical analysis, not as a
    headline accuracy metric. Returns None if no anomaly segment or no
    crossing is found."""
    scores = normalize_scores(scores)
    onset_idxs = np.where(labels == 1)[0]
    if len(onset_idxs) == 0:
        return None
    onset = int(onset_idxs[0])
    crossing = np.where(scores[onset:] >= threshold)[0]
    if len(crossing) == 0:
        return None
    return int(crossing[0])


def simplified_region_overlap_score(
    pred_mask: np.ndarray, gt_mask: np.ndarray, iou_threshold: float = 0.1
) -> float:
    """Frame-level IoU between a thresholded prediction mask and the
    ground-truth anomaly mask, for a single frame. A crude stand-in for
    RBDC — see module docstring."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    iou = intersection / union
    return float(iou >= iou_threshold)
