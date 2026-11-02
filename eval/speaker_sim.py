# eval/speaker_sim.py
# ---------------------------------------------------------------------------
# ECAPA-TDNN speaker similarity: cosine similarity between two audio clips'
# speaker embeddings (speechbrain/spkrec-ecapa-voxceleb, pretrained on
# VoxCeleb — NOT fine-tuned on Marathi/Indic speakers, a domain-mismatch
# caveat worth stating in the paper). This is the standard objective proxy
# for "does the cloned voice sound like the reference speaker" in TTS/
# voice-cloning literature. Range: -1..1, higher = more similar identity;
# in practice same-speaker pairs typically score well above 0.5-0.6 with
# this model, but there's no universal threshold — report the number, not
# a pass/fail call, unless it's calibrated against this project's own data.
#
# Loaded/unloaded like the other GPU-capable stages (sequential discipline)
# even though ECAPA itself is small — kept consistent with the project's
# stage-loading pattern rather than a special case.
# ---------------------------------------------------------------------------

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch

ECAPA_SAMPLE_RATE = 16000  # VoxCeleb / speechbrain's pretrained ECAPA expects 16kHz input


@dataclass
class SpeakerSimResult:
    cosine_similarity: float


class SpeakerSimStage:
    def __init__(self, cache_dir: str | Path, device: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return

        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(self.cache_dir / "ecapa"),
            run_opts={"device": self.device},
            # Windows without Developer Mode can't create symlinks (confirmed
            # empirically: OSError: [WinError 1314] A required privilege is
            # not held by the client). speechbrain's own fetch() doesn't
            # fall back gracefully the way huggingface_hub does elsewhere in
            # this project — force plain file copies instead.
            local_strategy=LocalStrategy.COPY,
        )

    def _embed(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        if sr != ECAPA_SAMPLE_RATE:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=ECAPA_SAMPLE_RATE)
        wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = self._model.encode_batch(wav)
        return emb.squeeze()

    def similarity(self, audio_a: np.ndarray, sr_a: int, audio_b: np.ndarray, sr_b: int) -> SpeakerSimResult:
        if self._model is None:
            raise RuntimeError("SpeakerSimStage.load() must be called before similarity().")
        emb_a = self._embed(audio_a, sr_a)
        emb_b = self._embed(audio_b, sr_b)
        cos_sim = torch.nn.functional.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        return SpeakerSimResult(cosine_similarity=float(cos_sim))

    def unload(self) -> None:
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
