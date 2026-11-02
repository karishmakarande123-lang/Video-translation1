# prosody/transfer.py
# ---------------------------------------------------------------------------
# DSP-baseline cross-lingual prosody transfer: rescales the synthesized
# (Marathi) audio's F0 contour to match the source (English) audio's F0
# statistics (mean + spread, in log domain), and matches overall loudness
# (RMS). Resynthesis uses WORLD (pyworld): analyze the target audio's own
# spectral envelope + aperiodicity (preserving its phonetic content/timbre),
# swap in the rescaled F0 contour, resynthesize.
#
# IMPORTANT — what this is NOT: the original project spec calls for a
# trained <30M-parameter PyTorch network (prosody/mapping_model.py) learning
# a genuine cross-lingual prosody mapping from a parallel EN-MR corpus. No
# such training data or trained model exists in this project. This module
# is a global-statistics DSP baseline — it matches pitch RANGE/REGISTER, not
# the actual pitch contour SHAPE (intonation pattern), since source and
# target utterances differ in duration, phoneme sequence, and word order,
# so frame-level contour transfer isn't meaningful without forced alignment.
# Treat results from this module as a naive baseline to compare a future
# learned model against, not as a "prosody transfer" quality result on its
# own. See TECHNICAL_REPORT.md §6/§7.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyworld as pw

from prosody.extract_f0 import extract_f0, f0_stats


@dataclass
class ProsodyTransferResult:
    audio: np.ndarray
    source_f0_stats: dict
    target_f0_stats_before: dict
    target_f0_stats_after: dict
    source_rms: float
    target_rms_before: float
    target_rms_after: float


def _rescale_f0(target_f0: np.ndarray, source_stats: dict, target_stats: dict) -> np.ndarray:
    if target_stats["voiced_frames"] == 0 or source_stats["voiced_frames"] == 0:
        return target_f0  # nothing usable to rescale toward/from

    target_std_log = np.log(1.0 + target_stats["std_hz"] / max(target_stats["mean_hz"], 1e-6))
    source_std_log = np.log(1.0 + source_stats["std_hz"] / max(source_stats["mean_hz"], 1e-6))
    target_std_log = max(target_std_log, 1e-6)

    voiced = target_f0 > 0
    log_f0 = np.zeros_like(target_f0)
    log_f0[voiced] = np.log(target_f0[voiced])

    target_mean_log = np.log(max(target_stats["mean_hz"], 1e-6))
    source_mean_log = np.log(max(source_stats["mean_hz"], 1e-6))

    rescaled_log_f0 = np.zeros_like(target_f0)
    rescaled_log_f0[voiced] = (
        (log_f0[voiced] - target_mean_log) / target_std_log * source_std_log + source_mean_log
    )

    new_f0 = np.zeros_like(target_f0)
    new_f0[voiced] = np.exp(rescaled_log_f0[voiced])
    return new_f0


def transfer_prosody(
    source_audio: np.ndarray,
    source_sr: int,
    target_audio: np.ndarray,
    target_sr: int,
    frame_period_ms: float = 5.0,
) -> ProsodyTransferResult:
    """
    Rescales `target_audio`'s pitch register to match `source_audio`'s F0
    statistics, and matches overall loudness. Returns the modified audio at
    `target_sr` (unchanged sample rate/duration — only pitch/loudness
    change, not timing; isochrony/rate_controller.py handles duration
    separately).
    """
    source_f0, _ = extract_f0(source_audio, source_sr, frame_period_ms)
    source_stats = f0_stats(source_f0)

    target_audio64 = np.ascontiguousarray(target_audio, dtype=np.float64)
    target_f0, t = pw.harvest(target_audio64, target_sr, frame_period=frame_period_ms)
    target_stats_before = f0_stats(target_f0)

    sp = pw.cheaptrick(target_audio64, target_f0, t, target_sr)
    ap = pw.d4c(target_audio64, target_f0, t, target_sr)

    new_f0 = _rescale_f0(target_f0, source_stats, target_stats_before)
    target_stats_after = f0_stats(new_f0)

    resynthesized = pw.synthesize(new_f0, sp, ap, target_sr, frame_period_ms)
    resynthesized = resynthesized.astype(np.float32)

    # Loudness match (simple scalar RMS gain, not a full envelope warp —
    # cheap and avoids re-introducing clipping; guard against silence).
    source_rms = float(np.sqrt(np.mean(source_audio.astype(np.float64) ** 2)))
    target_rms_before = float(np.sqrt(np.mean(resynthesized.astype(np.float64) ** 2)))
    if target_rms_before > 1e-8:
        gain = source_rms / target_rms_before
        # Cap the gain so a very quiet/loud source can't blow up the output;
        # the peak-normalization safeguard in baseline/tts.py-style code
        # would need to run again downstream regardless.
        gain = float(np.clip(gain, 0.3, 3.0))
        resynthesized = resynthesized * gain
    target_rms_after = float(np.sqrt(np.mean(resynthesized.astype(np.float64) ** 2)))

    return ProsodyTransferResult(
        audio=resynthesized,
        source_f0_stats=source_stats,
        target_f0_stats_before=target_stats_before,
        target_f0_stats_after=target_stats_after,
        source_rms=source_rms,
        target_rms_before=target_rms_before,
        target_rms_after=target_rms_after,
    )
