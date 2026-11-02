# eval/chrf.py
# ---------------------------------------------------------------------------
# chrF++ (character n-gram F-score with word bigrams added, Popović 2017)
# between an ASR round-trip transcription and a reference text. Generally
# preferred over BLEU for morphologically rich languages like Marathi —
# BLEU's word-level n-gram matching harshly penalizes minor inflectional
# variation that chrF++'s character-level matching tolerates better, which
# matters a lot given the ASR round-trip noise this project's transcriptions
# already carry (see TECHNICAL_REPORT.md §4).
#
# Same reference-choice caveat as eval/asr_bleu.py applies here.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import sacrebleu

# word_order=2 is what makes this chrF++ rather than plain chrF (word_order=0).
_CHRF_PLUS_PLUS = sacrebleu.CHRF(word_order=2)


@dataclass
class ChrfResult:
    score: float  # 0-100
    hypothesis: str
    reference: str


def chrf_from_texts(hypothesis: str, reference: str) -> ChrfResult:
    result = _CHRF_PLUS_PLUS.sentence_score(hypothesis, [reference])
    return ChrfResult(score=result.score, hypothesis=hypothesis, reference=reference)


def chrf_from_audio(audio_path: str, reference_text: str, asr_stage, language: str = "mr") -> ChrfResult:
    """Transcribes `audio_path` via an already-loaded ASRStage, then scores against `reference_text`."""
    hypothesis = asr_stage.transcribe(audio_path, language=language)
    return chrf_from_texts(hypothesis, reference_text)


def corpus_chrf(hypotheses: list[str], references: list[str]) -> float:
    result = _CHRF_PLUS_PLUS.corpus_score(hypotheses, [references])
    return result.score
