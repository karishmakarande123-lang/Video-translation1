# baseline/asr.py
# ---------------------------------------------------------------------------
# ASR stage: English speech -> English text, via faster-whisper (CTranslate2
# backend, int8_float16 on GPU). Loaded/unloaded on demand by
# baseline/pipeline.py so it never shares GPU memory with the MT or TTS
# stage (sequential loading discipline, see configs/hardware.yaml).
# ---------------------------------------------------------------------------

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class ASRStage:
    """Wraps faster-whisper for the ASR stage of the pipeline."""

    def __init__(self, config: dict, cache_dir: str | Path):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return  # already loaded

        from faster_whisper import WhisperModel

        device = self.config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        compute_type = self.config.get("compute_type", "int8_float16")
        if device == "cpu" and compute_type == "int8_float16":
            # int8_float16 is a GPU-only compute type; fall back sanely on CPU.
            compute_type = "int8"

        logger.info(
            "Loading faster-whisper '%s' on %s (%s)",
            self.config.get("model_size", "small"),
            device,
            compute_type,
        )
        self._model = WhisperModel(
            self.config.get("model_size", "small"),
            device=device,
            device_index=self.config.get("device_index", 0),
            compute_type=compute_type,
            download_root=str(self.cache_dir / "faster_whisper"),
        )

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribes `audio_path` and returns the concatenated text."""
        if self._model is None:
            raise RuntimeError("ASRStage.load() must be called before transcribe().")

        segments, info = self._model.transcribe(
            audio_path,
            language=language or self.config.get("language", "en"),
            beam_size=self.config.get("beam_size", 5),
            vad_filter=self.config.get("vad_filter", True),
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        logger.info(
            "ASR done: detected_language=%s duration=%.2fs -> %d chars",
            info.language,
            info.duration,
            len(text),
        )
        return text

    def unload(self) -> None:
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
