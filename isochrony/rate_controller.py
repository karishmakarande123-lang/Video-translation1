# isochrony/rate_controller.py
# ---------------------------------------------------------------------------
# Post-hoc isochrony correction: time-stretches synthesized audio so its
# duration falls within tolerance of a target duration (normally the
# source speech's duration, from duration_predictor.py).
#
# This is a DSP baseline (phase-vocoder time-stretch via librosa), applied
# AFTER TTS synthesis — not a duration-aware decoder that controls
# generation itself. It works on any audio without retraining, at the cost
# of some artifacts at extreme stretch ratios (librosa's phase vocoder
# degrades gracefully up to roughly +/-30-40%; beyond that, pitch/timbre
# artifacts become audible). See TECHNICAL_REPORT.md §6 for the tradeoff
# vs. a learned duration-constrained decoder.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class IsochronyResult:
    audio: np.ndarray
    original_duration_sec: float
    target_duration_sec: float
    duration_ratio_before: float  # original_duration / target_duration
    duration_ratio_after: float  # actual output duration / target_duration, after any correction
    stretch_applied: bool
    compliant_before: bool
    compliant_after: bool


def apply_isochrony_correction(
    audio: np.ndarray,
    sr: int,
    target_duration_sec: float,
    tolerance: float = 0.15,
) -> IsochronyResult:
    """
    Time-stretches `audio` toward `target_duration_sec` if it falls outside
    `tolerance` (fractional, e.g. 0.15 = +/-15%). Returns the (possibly
    unchanged) audio plus before/after compliance metrics.
    """
    original_duration = len(audio) / sr
    ratio_before = original_duration / target_duration_sec
    compliant_before = abs(ratio_before - 1.0) <= tolerance

    if compliant_before:
        return IsochronyResult(
            audio=audio,
            original_duration_sec=original_duration,
            target_duration_sec=target_duration_sec,
            duration_ratio_before=ratio_before,
            duration_ratio_after=ratio_before,
            stretch_applied=False,
            compliant_before=True,
            compliant_after=True,
        )

    # librosa's rate > 1 speeds up (shortens) audio, rate < 1 slows down
    # (lengthens) it. To go from original_duration to target_duration:
    # rate = original_duration / target_duration == ratio_before.
    stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=ratio_before)

    new_duration = len(stretched) / sr
    ratio_after = new_duration / target_duration_sec
    compliant_after = abs(ratio_after - 1.0) <= tolerance

    return IsochronyResult(
        audio=stretched,
        original_duration_sec=original_duration,
        target_duration_sec=target_duration_sec,
        duration_ratio_before=ratio_before,
        duration_ratio_after=ratio_after,
        stretch_applied=True,
        compliant_before=False,
        compliant_after=compliant_after,
    )
