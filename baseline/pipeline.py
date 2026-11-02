# baseline/pipeline.py
# ---------------------------------------------------------------------------
# Orchestrates the cascaded EN->MR S2ST baseline: ASR -> MT -> TTS.
#
# This file defines the STAGE CONTRACT that asr.py / mt.py / tts.py implement
# (each stage exposes .load(), a single inference method, and .unload()), and
# enforces the sequential GPU-loading discipline required on a 6GB card:
# only one stage's model is ever resident on GPU at a time. The pipeline
# instantiates a stage, loads it, runs inference, unloads it, empties the
# CUDA cache, then moves to the next stage.
#
# `status_callback` is the hook the GUI (gui/app.py, gui/state.py) will use
# later to show "ASR running -> MT running -> TTS running" progress without
# this module knowing anything about Gradio.
#
# The `mode` argument ("baseline" vs "baseline+research") and the unused
# prosody/isochrony/speaker slots on PipelineResult are placeholders so that
# later modules (prosody/, isochrony/, speaker/) can plug into this same
# pipeline without a rewrite; they are not implemented in this baseline file.
# ---------------------------------------------------------------------------

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
import yaml

logger = logging.getLogger(__name__)

StageName = Literal["asr", "mt", "tts", "research"]
StageStatus = Literal["pending", "running", "done", "error"]
StatusCallback = Callable[[StageName, StageStatus, str], None]

PipelineMode = Literal["baseline", "baseline+research"]


# s2st_en_mr/ project root — baseline/ is one level below it. Used to
# anchor relative paths (config file location, cache dirs) so behavior
# doesn't depend on the process's current working directory. This matters
# in practice: launching via a terminal `cd`'d into the project root vs.
# an IDE's "Run" button (which may use the file's own directory as cwd)
# can resolve the same relative path to two different locations, causing
# a stage to write output under one path and a later reader to look for it
# under another — confirmed happening with the GUI writing TTS output
# under gui/.cache/... while callers expected .cache/... at the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_hardware_config(config_path: str | Path = "configs/hardware.yaml") -> dict:
    """Loads the single source of truth for device/dtype/model settings."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key, value in config.get("paths", {}).items():
        p = Path(value)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        config["paths"][key] = str(p)

    return config


@dataclass
class PipelineResult:
    # Baseline outputs (always populated on success). In "baseline+research"
    # mode, tts_audio_path is the FINAL (prosody+isochrony-enhanced) audio;
    # baseline_tts_audio_path holds the pre-enhancement TTS output so the
    # two can be compared directly.
    asr_text: str = ""
    mt_text: str = ""
    tts_audio_path: str = ""
    tts_sample_rate: int = 0
    baseline_tts_audio_path: Optional[str] = None

    # Timing, surfaced in the GUI status panel
    stage_timings_sec: dict = field(default_factory=dict)

    # Populated only in "baseline+research" mode. prosody_metrics/
    # duration_ratio come from the DSP baseline in prosody/ and isochrony/
    # (NOT a trained model — see those modules' docstrings and
    # TECHNICAL_REPORT.md §6/7). speaker_similarity awaits speaker/, still
    # unimplemented.
    prosody_metrics: Optional[dict] = None
    duration_ratio: Optional[float] = None
    speaker_similarity: Optional[float] = None

    error: Optional[str] = None


def _empty_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class S2STPipeline:
    """
    Cascaded ASR -> MT -> TTS pipeline with strict sequential GPU stage
    loading. Stage classes are constructed lazily inside `run()` (not in
    `__init__`) so that simply instantiating S2STPipeline never touches
    the GPU.
    """

    def __init__(self, config_path: str | Path = "configs/hardware.yaml"):
        self.config = load_hardware_config(config_path)
        if not self.config.get("sequential_stage_loading", True):
            raise ValueError(
                "sequential_stage_loading=false is not supported on this "
                "pipeline's 6GB VRAM target; refusing to hold multiple "
                "stages resident on GPU at once."
            )

    def _report(
        self,
        callback: Optional[StatusCallback],
        stage: StageName,
        status: StageStatus,
        message: str = "",
    ) -> None:
        logger.info("[%s] %s %s", stage, status, message)
        if callback is not None:
            callback(stage, status, message)

    def run(
        self,
        source_audio_path: str,
        reference_speaker_audio_path: str,
        reference_text: Optional[str] = None,
        mode: PipelineMode = "baseline",
        status_callback: Optional[StatusCallback] = None,
    ) -> PipelineResult:
        """
        Runs the full EN audio -> MR audio pipeline.

        source_audio_path: path to the English input speech (wav/mp3).
        reference_speaker_audio_path: short Marathi reference clip used by
            IndicF5 to clone the target voice/timbre for the TTS stage.
        reference_text: transcript of reference_speaker_audio_path. IndicF5
            is both audio- and text-prompted; if omitted, TTSStage falls
            back to auto-transcribing the reference clip (lower fidelity).
        mode: "baseline" runs ASR->MT->TTS only. "baseline+research" additionally
            applies the DSP prosody-transfer + isochrony-correction baseline
            from prosody/ and isochrony/ (NOT a trained model — see those
            modules' docstrings) to the TTS output.
        """
        result = PipelineResult()

        # Local imports: keeps `import baseline.pipeline` cheap and avoids
        # pulling in torch/faster-whisper/transformers just to build this
        # orchestrator object.
        from baseline.asr import ASRStage
        from baseline.mt import MTStage
        from baseline.tts import TTSStage

        try:
            # ---- ASR stage --------------------------------------------------
            self._report(status_callback, "asr", "running", "loading faster-whisper")
            t0 = time.perf_counter()
            asr = ASRStage(self.config["asr"], cache_dir=self.config["paths"]["cache_dir"])
            asr.load()
            asr_text = asr.transcribe(source_audio_path)
            asr.unload()
            del asr
            _empty_gpu_cache()
            result.asr_text = asr_text
            result.stage_timings_sec["asr"] = time.perf_counter() - t0
            self._report(status_callback, "asr", "done", asr_text)

            # ---- MT stage -----------------------------------------------------
            self._report(status_callback, "mt", "running", "loading IndicTrans2")
            t0 = time.perf_counter()
            mt = MTStage(self.config["mt"], cache_dir=self.config["paths"]["cache_dir"])
            mt.load()
            mt_text = mt.translate(asr_text)
            mt.unload()
            del mt
            _empty_gpu_cache()
            result.mt_text = mt_text
            result.stage_timings_sec["mt"] = time.perf_counter() - t0
            self._report(status_callback, "mt", "done", mt_text)

            # ---- TTS stage ----------------------------------------------------
            self._report(status_callback, "tts", "running", "loading IndicF5")
            t0 = time.perf_counter()
            tts = TTSStage(self.config["tts"], cache_dir=self.config["paths"]["cache_dir"])
            tts.load()
            audio_path, sample_rate = tts.synthesize(
                text=mt_text,
                reference_audio_path=reference_speaker_audio_path,
                reference_text=reference_text,
            )
            tts.unload()
            del tts
            _empty_gpu_cache()
            result.tts_audio_path = audio_path
            result.tts_sample_rate = sample_rate
            result.stage_timings_sec["tts"] = time.perf_counter() - t0
            self._report(status_callback, "tts", "done", audio_path)

            # ---- Research stage: DSP prosody transfer + isochrony -----------------
            # CPU-only (pyworld, librosa) — no GPU involved, so this never
            # competes with ASR/MT/TTS for VRAM regardless of ordering.
            if mode == "baseline+research":
                self._report(status_callback, "research", "running", "prosody transfer + isochrony")
                t0 = time.perf_counter()
                enhanced_path, prosody_metrics, duration_ratio = self._apply_research_enhancements(
                    source_audio_path=source_audio_path,
                    baseline_audio_path=audio_path,
                    sample_rate=sample_rate,
                )
                result.baseline_tts_audio_path = audio_path
                result.tts_audio_path = enhanced_path
                result.prosody_metrics = prosody_metrics
                result.duration_ratio = duration_ratio
                result.stage_timings_sec["research"] = time.perf_counter() - t0
                self._report(status_callback, "research", "done", enhanced_path)

        except Exception as exc:  # surfaced to GUI rather than crashing the app
            logger.exception("Pipeline failed")
            result.error = str(exc)
            # Best-effort cleanup so a mid-pipeline failure doesn't leave a
            # stage's model resident on a 6GB card for the next attempt.
            _empty_gpu_cache()

        return result

    def _apply_research_enhancements(
        self,
        source_audio_path: str,
        baseline_audio_path: str,
        sample_rate: int,
    ) -> tuple[str, dict, float]:
        """
        Applies the DSP prosody-transfer + isochrony-correction baseline to
        `baseline_audio_path`, using `source_audio_path` as the prosody/
        duration reference. Returns (enhanced_audio_path, prosody_metrics,
        duration_ratio). CPU-only — see prosody/transfer.py and
        isochrony/rate_controller.py for what this is and isn't.
        """
        import uuid

        import numpy as np
        import soundfile as sf

        from isochrony.duration_predictor import predict_target_duration_sec
        from isochrony.rate_controller import apply_isochrony_correction
        from prosody.transfer import transfer_prosody

        source_audio, source_sr = sf.read(source_audio_path)
        if source_audio.ndim > 1:
            source_audio = source_audio.mean(axis=1)  # downmix to mono
        source_audio = source_audio.astype(np.float32)

        target_audio, target_sr = sf.read(baseline_audio_path)
        target_audio = target_audio.astype(np.float32)

        prosody_result = transfer_prosody(source_audio, source_sr, target_audio, target_sr)

        target_duration = predict_target_duration_sec(source_audio_path)
        iso_result = apply_isochrony_correction(prosody_result.audio, target_sr, target_duration)

        # Guard against clipping post-enhancement, same lesson as
        # baseline/tts.py: peak-normalize only if needed, preserve dynamics
        # otherwise.
        final_audio = iso_result.audio
        peak = float(np.max(np.abs(final_audio))) if final_audio.size else 0.0
        if peak > 1.0:
            final_audio = final_audio / peak * 0.98

        cache_dir = Path(self.config["paths"]["cache_dir"])
        enhanced_path = cache_dir / f"research_{uuid.uuid4().hex}.wav"
        sf.write(str(enhanced_path), final_audio, samplerate=sample_rate)

        prosody_metrics = {
            "source_f0_mean_hz": prosody_result.source_f0_stats["mean_hz"],
            "source_f0_std_hz": prosody_result.source_f0_stats["std_hz"],
            "target_f0_mean_hz_before": prosody_result.target_f0_stats_before["mean_hz"],
            "target_f0_mean_hz_after": prosody_result.target_f0_stats_after["mean_hz"],
            "target_rms_before": prosody_result.target_rms_before,
            "target_rms_after": prosody_result.target_rms_after,
            "isochrony_stretch_applied": iso_result.stretch_applied,
            "isochrony_ratio_before": iso_result.duration_ratio_before,
            "isochrony_ratio_after": iso_result.duration_ratio_after,
            "isochrony_compliant_before": iso_result.compliant_before,
            "isochrony_compliant_after": iso_result.compliant_after,
        }

        return str(enhanced_path), prosody_metrics, iso_result.duration_ratio_after
