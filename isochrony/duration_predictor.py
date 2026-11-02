# isochrony/duration_predictor.py
# ---------------------------------------------------------------------------
# Predicts the isochrony TARGET duration for synthesized TTS output.
#
# This is a DSP baseline, not a trained model: the target is simply the
# source (English) speech's own duration — the isochrony goal is
# output_duration ~= source_duration, so dubbed speech stays lip/turn-sync
# with the source. A learned duration predictor (estimating target duration
# from source text/phoneme count without needing the source audio at
# inference time) is future work — see TECHNICAL_REPORT.md §6/§7.
# ---------------------------------------------------------------------------

from __future__ import annotations

import soundfile as sf


def predict_target_duration_sec(source_audio_path: str) -> float:
    """Returns the source audio's duration in seconds — the isochrony target."""
    info = sf.info(source_audio_path)
    return info.frames / info.samplerate
