# prosody/extract_f0.py
# ---------------------------------------------------------------------------
# F0 (fundamental frequency / pitch contour) extraction via WORLD's Harvest
# algorithm (pyworld). CPU-only, no GPU/training involved — this is real
# signal-processing analysis, usable standalone for offline feature caching
# (per the original project design: prosody features computed OFFLINE, never
# inside a training loop) or inline for the DSP prosody-transfer baseline in
# prosody/transfer.py.
# ---------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pyworld as pw


def extract_f0(audio: np.ndarray, sr: int, frame_period_ms: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts an F0 contour from `audio` (mono, float).
    Returns (f0, time_axis) — f0 is 0.0 at unvoiced frames (WORLD convention).
    """
    audio64 = np.ascontiguousarray(audio, dtype=np.float64)
    f0, t = pw.harvest(audio64, sr, frame_period=frame_period_ms)
    return f0, t


def f0_stats(f0: np.ndarray) -> dict:
    """Summary statistics over voiced frames only (f0 > 0)."""
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return {"voiced_frames": 0, "mean_hz": 0.0, "std_hz": 0.0, "min_hz": 0.0, "max_hz": 0.0}
    return {
        "voiced_frames": int(voiced.size),
        "mean_hz": float(np.mean(voiced)),
        "std_hz": float(np.std(voiced)),
        "min_hz": float(np.min(voiced)),
        "max_hz": float(np.max(voiced)),
    }
