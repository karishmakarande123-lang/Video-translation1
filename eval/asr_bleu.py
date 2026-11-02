# eval/asr_bleu.py
# ---------------------------------------------------------------------------
# ASR-BLEU: the standard S2ST evaluation metric when no direct text-level
# alignment exists between source and target speech — synthesize target
# speech, transcribe it back via ASR, compute BLEU between the transcription
# and a reference text (via sacrebleu).
#
# IMPORTANT — what "reference" means here, since this project has no
# human-authored parallel EN-MR corpus (e.g. FLORES, CVSS): two distinct
# reference choices are supported and MUST be reported separately, never
# conflated as "translation quality vs. human reference":
#   - Against KNOWN-CORRECT target text (e.g. samples/tts_validation/'s
#     real IISc dataset sentences): measures TTS+ASR round-trip content
#     fidelity, decoupled from MT quality — this is what
#     scripts/run_full_evaluation.py uses.
#   - Against the MT stage's own output text (a real end-to-end pipeline
#     run): measures round-trip fidelity to whatever MT produced, NOT
#     translation adequacy — MT could be wrong and this metric wouldn't
#     know.
# See TECHNICAL_REPORT.md for how this is framed in the write-up.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import sacrebleu


@dataclass
class BleuResult:
    score: float  # 0-100, sacrebleu convention
    hypothesis: str
    reference: str


def asr_bleu_from_texts(hypothesis: str, reference: str) -> BleuResult:
    """Sentence-level BLEU between an ASR transcription (hypothesis) and a reference text."""
    result = sacrebleu.sentence_bleu(hypothesis, [reference])
    return BleuResult(score=result.score, hypothesis=hypothesis, reference=reference)


def asr_bleu_from_audio(audio_path: str, reference_text: str, asr_stage, language: str = "mr") -> BleuResult:
    """Transcribes `audio_path` via an already-loaded ASRStage, then scores against `reference_text`."""
    hypothesis = asr_stage.transcribe(audio_path, language=language)
    return asr_bleu_from_texts(hypothesis, reference_text)


def corpus_asr_bleu(hypotheses: list[str], references: list[str]) -> float:
    """
    Corpus-level BLEU (sacrebleu.corpus_bleu) across many samples — the
    standard way to report BLEU for a dataset; more statistically stable
    than averaging per-sentence BLEU scores (which sacrebleu's own docs
    warn against — see https://github.com/mjpost/sacrebleu#chrf--chrf).
    """
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result.score
