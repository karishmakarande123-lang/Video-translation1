# eval/f0_metrics.py
# ---------------------------------------------------------------------------
# Frame-level F0 comparison — RMSE and Pearson correlation between two audio
# signals' F0 contours, after DTW alignment (the two signals generally
# differ in duration/timing). This is a genuine CONTOUR-level comparison —
# distinct from prosody/transfer.py's global mean/std statistics matching
# (see TECHNICAL_REPORT.md §6 for that distinction; that module rescales
# register, it doesn't measure contour similarity). Only frames voiced in
# BOTH signals after alignment are compared — unvoiced/silent regions have
# no meaningful pitch to compare.
#
# Alignment note: DTW runs directly on the two raw F0 sequences (including
# their 0-valued unvoiced frames), not on an MFCC-anchored path like
# eval/mcd.py uses. This is a simplified baseline choice — an MFCC-anchored
# alignment shared with mcd.py would be more robust, but keeps this module
# self-contained per the original eval/ file layout. Flag if consistency
# with eval/mcd.py's alignment matters for a specific paper claim.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from prosody.extract_f0 import extract_f0


@dataclass
class F0MetricsResult:
    f0_rmse_hz: float
    f0_correlation: float
    n_voiced_aligned_frames: int


def compute_f0_metrics(
    audio_a: np.ndarray,
    sr_a: int,
    audio_b: np.ndarray,
    sr_b: int,
    frame_period_ms: float = 5.0,
) -> F0MetricsResult:
    f0_a, _ = extract_f0(audio_a, sr_a, frame_period_ms)
    f0_b, _ = extract_f0(audio_b, sr_b, frame_period_ms)

    seq_a = f0_a.reshape(1, -1).astype(np.float64)
    seq_b = f0_b.reshape(1, -1).astype(np.float64)
    _, wp = librosa.sequence.dtw(X=seq_a, Y=seq_b, metric="euclidean")
    wp = wp[::-1]

    aligned_a = f0_a[wp[:, 0]]
    aligned_b = f0_b[wp[:, 1]]

    voiced = (aligned_a > 0) & (aligned_b > 0)
    n_voiced = int(voiced.sum())
    if n_voiced < 2:
        return F0MetricsResult(f0_rmse_hz=float("nan"), f0_correlation=float("nan"), n_voiced_aligned_frames=n_voiced)

    va, vb = aligned_a[voiced], aligned_b[voiced]
    rmse = float(np.sqrt(np.mean((va - vb) ** 2)))
    # np.corrcoef on a constant sequence (zero variance) produces NaN —
    # only realistic on pathologically short/flat clips, but guard anyway.
    corr = float(np.corrcoef(va, vb)[0, 1]) if np.std(va) > 0 and np.std(vb) > 0 else float("nan")

    return F0MetricsResult(f0_rmse_hz=rmse, f0_correlation=corr, n_voiced_aligned_frames=n_voiced)
