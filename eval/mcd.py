# eval/mcd.py
# ---------------------------------------------------------------------------
# Mel-Cepstral Distortion (MCD, Kubichek 1993) between two audio signals —
# standard objective TTS quality metric measuring spectral-envelope
# distance. Lower is better; MCD is unbounded above but values roughly
# under ~6-8dB are typical for "good" synthesis vs. a real reference,
# depending on domain/corpus.
#
# Uses pyworld (spectral envelope, already a project dependency — see
# prosody/extract_f0.py) + pysptk (envelope -> mel-cepstral coefficients,
# the standard SPTK-convention analysis MCD's formula assumes) — NOT
# librosa.feature.mfcc(), which computes dB-scale DCT coefficients in a
# completely different feature space. Confirmed empirically: naively
# plugging librosa MFCCs into the standard MCD formula produced values in
# the 300-500dB range (nonsense — real MCD is single/low-double digits).
# pyworld+pysptk mel-cepstral coefficients come out at the expected O(1)
# magnitude, matching literature MCD ranges.
#
# DTW-aligns the two signals' mel-cepstral sequences first, since they
# generally differ in duration/timing (different speakers, different
# synthesis paths) — computing MCD on frame-index-aligned audio without
# DTW would be meaningless whenever the two clips aren't the same length.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import pysptk
import pyworld as pw


@dataclass
class MCDResult:
    mcd_db: float
    n_aligned_frames: int


def _mel_cepstrum(audio: np.ndarray, sr: int, order: int = 24) -> np.ndarray:
    """Returns (n_frames, order+1) mel-cepstral coefficients via WORLD's
    spectral envelope (cheaptrick) + pysptk's mel-cepstral analysis."""
    audio64 = np.ascontiguousarray(audio, dtype=np.float64)
    f0, t = pw.harvest(audio64, sr)
    sp = pw.cheaptrick(audio64, f0, t, sr)
    alpha = pysptk.util.mcepalpha(sr)
    return pysptk.sp2mc(sp, order=order, alpha=alpha)


def compute_mcd(
    audio_a: np.ndarray,
    sr_a: int,
    audio_b: np.ndarray,
    sr_b: int,
    order: int = 24,
    target_sr: int = 16000,
) -> MCDResult:
    """
    MCD between audio_a and audio_b. Resamples both to `target_sr` first
    for a consistent analysis rate — inputs often start at different
    sample rates (e.g. TTS output at 24kHz, dataset recordings at 48kHz).
    """
    if sr_a != target_sr:
        audio_a = librosa.resample(audio_a.astype(np.float32), orig_sr=sr_a, target_sr=target_sr)
    if sr_b != target_sr:
        audio_b = librosa.resample(audio_b.astype(np.float32), orig_sr=sr_b, target_sr=target_sr)

    mc_a = _mel_cepstrum(audio_a, target_sr, order)
    mc_b = _mel_cepstrum(audio_b, target_sr, order)

    # Drop the 0th coefficient (overall gain/energy) per MCD convention —
    # it reflects loudness, not spectral envelope shape. librosa.sequence.dtw
    # wants feature-major arrays: (n_features, n_frames).
    feat_a = mc_a[:, 1:].T
    feat_b = mc_b[:, 1:].T

    _, wp = librosa.sequence.dtw(X=feat_a, Y=feat_b, metric="euclidean")
    wp = wp[::-1]  # librosa returns the warping path end-to-start

    diffs = feat_a[:, wp[:, 0]] - feat_b[:, wp[:, 1]]  # (order, n_aligned_frames)
    per_frame_dist = np.sqrt(np.sum(diffs**2, axis=0))
    mcd = (10.0 / np.log(10)) * np.sqrt(2.0) * float(np.mean(per_frame_dist))

    return MCDResult(mcd_db=mcd, n_aligned_frames=int(wp.shape[0]))
