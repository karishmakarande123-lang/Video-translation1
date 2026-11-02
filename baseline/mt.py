# baseline/mt.py
# ---------------------------------------------------------------------------
# MT stage: English text -> Marathi text, via IndicTrans2 (distilled 200M,
# ai4bharat/indictrans2-en-indic-dist-200M). Loaded/unloaded on demand by
# baseline/pipeline.py; never resident on GPU at the same time as the ASR
# or TTS stage.
#
# IndicTrans2 requires IndicTransToolkit for pre/post-processing (language
# tags, script/number normalization, punctuation handling) — this is NOT
# optional, raw tokenizer in/out will produce broken translations.
# ---------------------------------------------------------------------------

from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class MTStage:
    """Wraps ai4bharat/indictrans2-en-indic-dist-200M for the MT stage."""

    def __init__(self, config: dict, cache_dir: str | Path):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = None
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        if self._model is not None:
            return  # already loaded

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError as exc:
            raise ImportError(
                "IndicTransToolkit is required for IndicTrans2 pre/post-"
                "processing. Install with:\n"
                "  pip install IndicTransToolkit\n"
                "or, if that package isn't on PyPI in your environment:\n"
                "  pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
            ) from exc

        model_name = self.config.get("model_name", "ai4bharat/indictrans2-en-indic-dist-200M")
        self._device = self.config.get("device", "cuda") if torch.cuda.is_available() else "cpu"

        logger.info("Loading IndicTrans2 '%s' on %s", model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir / "indictrans2"),
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir / "indictrans2"),
        )
        self._model = self._model.to(self._device)
        if self.config.get("dtype", "fp16") == "fp16" and self._device == "cuda":
            self._model = self._model.half()
        self._model.eval()
        self._processor = IndicProcessor(inference=True)

    def translate(self, text: str) -> str:
        """Translates English `text` to Marathi. Empty input -> empty output."""
        if self._model is None:
            raise RuntimeError("MTStage.load() must be called before translate().")
        if not text.strip():
            return ""

        src_lang = self.config.get("src_lang", "eng_Latn")
        tgt_lang = self.config.get("tgt_lang", "mar_Deva")

        batch = self._processor.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self._tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            generated_tokens = self._model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                num_beams=5,
                use_cache=True,
            )

        decoded = self._tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(), skip_special_tokens=True
        )
        translations = self._processor.postprocess_batch(decoded, lang=tgt_lang)
        translation = translations[0] if translations else ""
        logger.info("MT done: %d chars -> %d chars", len(text), len(translation))
        return translation

    def unload(self) -> None:
        self._tokenizer = None
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
