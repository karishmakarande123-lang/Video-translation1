# scripts/generate_tts_validation_samples.py
# ---------------------------------------------------------------------------
# Synthesizes 10 real, professionally-transcribed Marathi sentences (drawn
# from the IISc SYSPIN dataset, one per topic domain, 6-18 words each) with
# a single cloned reference voice, then validates each output via the same
# round-trip ASR sanity check used elsewhere in this project (see
# TECHNICAL_REPORT.md §4): feed the synthesized audio back through
# faster-whisper and compare against the intended text. This is a
# diagnostic/qualitative check, NOT a formal ASR-BLEU/WER evaluation
# (eval/ is unimplemented) — word_overlap_pct below is a rough proxy only.
#
# Respects the sequential GPU-loading discipline: TTS is fully loaded, runs
# all 10 syntheses, and is unloaded BEFORE ASR loads for the round-trip
# checks — never both resident on GPU at once.
#
# Usage (from the s2st_en_mr/ project root):
#   python -m scripts.generate_tts_validation_samples
# Output: samples/tts_validation/*.wav, samples/tts_validation/results.csv,
#         samples/tts_validation/results.md
# ---------------------------------------------------------------------------

from __future__ import annotations

import csv
import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from baseline.pipeline import load_hardware_config

OUT_DIR = Path("samples/tts_validation")

# One consistent reference speaker for all 10 syntheses (isolates the effect
# of target-text content; reference voice is held constant).
REF_AUDIO = (
    # Included in this repository (CC-BY-4.0, see samples/syspin_utterances/README.md)
    "samples/syspin_utterances/IISc_SYSPINProject_mr_f_AGRI_00004.wav"
)
REF_TEXT = (
    "ऑक्टोबर महिन्यात भाज्या एकवीस पॉइंट शहात्तर टक्क्यांनी, कांदा तेवीस पॉइंट "
    "अठ्ठावन्न टक्क्यांनी, फळांच्या किंमती अकरा पॉइंट शहाण्णव टक्क्यांनी, दूध "
    "अकरा पॉइंट बारा टक्क्यांनी तर अंडी, मांस व मासे बारा पॉइंट पस्तीस "
    "टक्क्यांनी महागले."
)

NUM_ANNOTATION_RE = re.compile(r"\{[^{}]+\}\{([^{}]+)\}")


def clean_transcript(text: str) -> str:
    """
    IISc transcripts encode numbers as {digit-form}{word-form}, e.g.
    "{3}{तीन}", for training flexibility. Keep only the spoken word-form —
    feeding raw braces to the TTS text frontend is out-of-distribution
    input, not natural Marathi text.
    """
    text = NUM_ANNOTATION_RE.sub(r"\1", text)
    return text.strip()


def audio_stats(data: np.ndarray, sr: int) -> dict:
    return {
        "duration_sec": round(len(data) / sr, 2),
        "peak": round(float(np.max(np.abs(data))), 3) if data.size else 0.0,
        "rms": round(float(np.sqrt(np.mean(data.astype(np.float64) ** 2))), 3) if data.size else 0.0,
        "clipped_pct": round(float(np.mean(np.abs(data) >= 0.999) * 100), 1) if data.size else 0.0,
    }


def word_overlap_pct(intended: str, transcribed: str) -> float:
    """Rough proxy only: fraction of intended words also present in the
    round-trip transcription, order-independent. Not a substitute for a
    real WER/ASR-BLEU computation."""
    intended_words = set(intended.replace(",", "").replace(".", "").split())
    transcribed_words = set(transcribed.replace(",", "").replace(".", "").split())
    if not intended_words:
        return 0.0
    return round(100 * len(intended_words & transcribed_words) / len(intended_words), 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "input_sentences.json", "r", encoding="utf-8") as f:
        picked = json.load(f)

    config = load_hardware_config("configs/hardware.yaml")
    results = []

    # ---- Phase 1: TTS (loaded once, 10 syntheses, unloaded) -------------------
    from baseline.tts import TTSStage

    tts = TTSStage(config["tts"], cache_dir=config["paths"]["cache_dir"])
    print("Loading TTS...")
    tts.load()

    for i, (domain, sent_id, raw_text) in enumerate(picked, start=1):
        text = clean_transcript(raw_text)
        print(f"[{i}/10] ({domain}) synthesizing: {text[:40]}...")
        t0 = time.perf_counter()
        audio_path, sr = tts.synthesize(text=text, reference_audio_path=REF_AUDIO, reference_text=REF_TEXT)
        elapsed = time.perf_counter() - t0

        data, sr = sf.read(audio_path)
        stats = audio_stats(np.asarray(data, dtype=np.float32), sr)

        final_path = OUT_DIR / f"sample_{i:02d}_{domain.lower()}.wav"
        Path(audio_path).replace(final_path)

        results.append(
            {
                "sample": i,
                "domain": domain,
                "sentence_id": sent_id,
                "input_text": text,
                "synth_time_sec": round(elapsed, 1),
                "output_path": str(final_path),
                **stats,
            }
        )

    tts.unload()
    del tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("TTS unloaded.\n")

    # ---- Phase 2: ASR round-trip validation (loaded once, 10 transcriptions) --
    from baseline.asr import ASRStage

    asr = ASRStage(config["asr"], cache_dir=config["paths"]["cache_dir"])
    print("Loading ASR for round-trip validation...")
    asr.load()

    for r in results:
        print(f"[{r['sample']}/10] round-trip ASR...")
        transcription = asr.transcribe(r["output_path"], language="mr")
        r["roundtrip_asr"] = transcription
        r["word_overlap_pct"] = word_overlap_pct(r["input_text"], transcription)

    asr.unload()
    del asr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("ASR unloaded.\n")

    # ---- Write results ------------------------------------------------------------
    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    md_path = OUT_DIR / "results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# TTS Validation: 10-Sample Comparison\n\n")
        f.write(
            "One cloned reference voice, 10 real Marathi sentences (one per "
            "topic domain, IISc SYSPIN dataset). Round-trip ASR is a "
            "qualitative sanity check, not a formal WER/ASR-BLEU metric.\n\n"
        )
        f.write(
            "| # | Domain | Input text | Duration (s) | Peak | RMS | Clipped % | "
            "Word overlap % | Round-trip ASR |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(
                f"| {r['sample']} | {r['domain']} | {r['input_text']} | "
                f"{r['duration_sec']} | {r['peak']} | {r['rms']} | "
                f"{r['clipped_pct']} | {r['word_overlap_pct']} | {r['roundtrip_asr']} |\n"
            )

        avg_overlap = sum(r["word_overlap_pct"] for r in results) / len(results)
        avg_clip = sum(r["clipped_pct"] for r in results) / len(results)
        f.write(f"\n**Average word overlap: {avg_overlap:.1f}%. Average clipped: {avg_clip:.1f}%.**\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
