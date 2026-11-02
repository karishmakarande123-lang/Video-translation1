# baseline/tts.py
# ---------------------------------------------------------------------------
# TTS stage: Marathi text + a reference speaker clip -> cloned Marathi
# speech, via ai4bharat/IndicF5 (flow-matching, reference-audio prompted).
# Loaded/unloaded on demand by baseline/pipeline.py; never resident on GPU
# at the same time as the ASR or MT stage.
#
# IndicF5 is prompted by BOTH a reference audio clip AND that clip's own
# transcript (needed for alignment). If the GUI/caller doesn't supply
# `reference_text`, this stage auto-transcribes the reference clip with a
# small CPU-only faster-whisper pass (kept off the GPU on purpose, since the
# TTS model itself is resident on GPU at that point) — pass reference_text
# explicitly whenever possible for better cloning fidelity.
# ---------------------------------------------------------------------------

from __future__ import annotations

import gc
import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TTSStage:
    """Wraps ai4bharat/IndicF5 for the TTS stage of the pipeline."""

    def __init__(self, config: dict, cache_dir: str | Path):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._device = None

    def load(self) -> None:
        if self._model is not None:
            return  # already loaded

        from transformers import AutoModel

        self._device = self.config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        model_name = self.config.get("model_name", "ai4bharat/IndicF5")

        logger.info("Loading IndicF5 '%s' on %s", model_name, self._device)
        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir / "indicf5"),
        )
        self._patch_undertrained_weights(model_name)
        self._model = self._model.to(self._device)
        # NOT casting to fp16 here even if config asks for it: IndicF5's own
        # inference pipeline (f5_tts.infer.utils_infer.infer_process ->
        # vocoder.decode) computes mel-spectrogram features in fp32 without
        # autocast, so a naive .half() cast leaves the vocoder's conv
        # weights in fp16 while its input stays fp32 -> RuntimeError:
        # Input type (float) and bias type (struct c10::Half) should be the
        # same. Confirmed empirically; fp32 fits fine in 6GB VRAM anyway
        # since this stage is never GPU-resident alongside ASR/MT.
        self._model.eval()

    def _patch_undertrained_weights(self, model_name: str) -> None:
        """
        Works around a confirmed bug in ai4bharat/IndicF5's trust_remote_code
        model.py: because it wraps submodules in torch.compile() inside
        __init__ (self.ema_model = torch.compile(load_model(...)), likewise
        for self.vocoder), transformers' from_pretrained() key-matching
        fails to bind ~16 parameters (the GRN scale/shift in the DiT text
        embedding, and the ConvNeXt layer-scale gamma in the vocoder) even
        though model.safetensors has correct trained values under the exact
        same key names. Left unpatched, those parameters silently stay at
        their zero/default __init__ value and the model produces incoherent
        noise, NOT degraded-but-recognizable speech — confirmed by feeding
        the output back through faster-whisper and getting hallucinated
        garbage tokens instead of any real transcription. A second, direct
        state_dict load repairs it completely (0 missing, 0 unexpected).
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        ckpt_path = hf_hub_download(
            model_name, filename="model.safetensors", cache_dir=str(self.cache_dir / "indicf5")
        )
        raw_state_dict = load_file(ckpt_path)
        missing, unexpected = self._model.load_state_dict(raw_state_dict, strict=False)
        if missing or unexpected:
            logger.warning(
                "IndicF5 weight patch: %d still missing, %d unexpected after reload "
                "(expected 0/0 — the upstream bug this works around may have changed)",
                len(missing),
                len(unexpected),
            )

    def synthesize(
        self,
        text: str,
        reference_audio_path: str,
        reference_text: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Synthesizes Marathi speech for `text`, cloning the voice from
        `reference_audio_path`. Returns (output_wav_path, sample_rate).
        """
        if self._model is None:
            raise RuntimeError("TTSStage.load() must be called before synthesize().")

        if not reference_text:
            reference_text = self._auto_transcribe_reference(reference_audio_path)

        with torch.no_grad():
            audio = self._model(
                text,
                ref_audio_path=reference_audio_path,
                ref_text=reference_text,
            )

        audio = np.asarray(audio, dtype=np.float32)
        sample_rate = self.config.get("sample_rate", 24000)

        # IndicF5's raw output routinely exceeds +/-1.0 (confirmed
        # empirically: peaks around 1.2-1.4). sf.write() defaults to 16-bit
        # PCM for .wav, which HARD-CLIPS (not scales) anything outside
        # [-1, 1] — measured 94-97% of samples clipped on unpatched output,
        # which is what "very noisy" audio actually was. Peak-normalize
        # only when needed, leaving a little headroom, so already-safe
        # audio keeps its natural dynamics.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak * 0.98

        out_path = self.cache_dir / f"tts_{uuid.uuid4().hex}.wav"
        import soundfile as sf

        sf.write(str(out_path), audio, samplerate=sample_rate)
        logger.info("TTS done: %s (%.2fs audio)", out_path, len(audio) / sample_rate)
        return str(out_path), sample_rate

    def _auto_transcribe_reference(self, reference_audio_path: str) -> str:
        """
        Fallback when no reference transcript is supplied. Deliberately runs
        on CPU with a tiny model so it never competes for VRAM with the
        IndicF5 model already loaded on GPU.
        """
        from faster_whisper import WhisperModel

        logger.warning(
            "No reference_text supplied for voice cloning; auto-"
            "transcribing the reference clip on CPU with faster-whisper "
            "'small'. For best cloning fidelity, pass the reference clip's "
            "exact transcript instead (e.g. a GUI text box)."
        )
        whisper = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _ = whisper.transcribe(reference_audio_path, language="mr")
        text = " ".join(segment.text.strip() for segment in segments).strip()
        del whisper
        gc.collect()
        return text

    def unload(self) -> None:
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
