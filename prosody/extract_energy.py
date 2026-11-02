# prosody/extract_energy.py
# ---------------------------------------------------------------------------
# RMS energy extraction — the second prosody feature (alongside F0) used by
# the DSP transfer baseline in prosody/transfer.py. CPU-only.
# ---------------------------------------------------------------------------

from __future__ import annotations

import librosa
import numpy as np


def extract_energy(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Frame-wise RMS energy contour."""
    return librosa.feature.rms(y=audio.astype(np.float32), frame_length=frame_length, hop_length=hop_length)[0]


def energy_stats(audio: np.ndarray) -> dict:
    """Whole-signal RMS (single scalar) — used for the simple loudness-matching baseline."""
    return {"rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))}
