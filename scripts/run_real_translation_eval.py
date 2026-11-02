# scripts/run_real_translation_eval.py
# ---------------------------------------------------------------------------
# Real end-to-end English->Marathi translation evaluation, fixing the
# reviewer finding that the paper's 10-sample tables never ran ASR/MT on
# real English audio (samples/tts_validation/ feeds Marathi text straight
# to TTS) and reused one unrelated 12s clip (srk_test_clip_12s.wav) as the
# prosody/duration target for all 10 unrelated samples
# (samples/research_comparison/).
#
# For each of the 10 real FLEURS English samples (fetched by
# scripts.fetch_real_translation_samples.py, with real source/speaker/
# duration metadata in samples/real_translation_eval/metadata.csv):
#   1. Runs the FULL S2STPipeline (ASR -> MT -> TTS -> DSP prosody/isochrony)
#      on the actual English audio, mode="baseline+research". The isochrony
#      target is THIS sample's own English source duration (not a shared
#      stress-test target).
#   2. Scores the MT output text against the REAL human Marathi reference
#      translation (FLEURS mr_in, same FLoRes-101 sentence id) with
#      BLEU/chrF++ — genuine translation-adequacy evaluation, which
#      eval/asr_bleu.py's docstring explicitly notes this project lacked
#      until now.
#   3. Scores baseline/enhanced synthesized audio against the REAL FLEURS
#      Marathi reference recording (MCD, F0-RMSE/correlation, ECAPA speaker
#      similarity) — NOTE: that reference is a different, independent human
#      speaker from the cloned target voice (FLEURS has no speaker overlap
#      across languages), so speaker-similarity here is bounded well below
#      what same-speaker ground truth (as in samples/full_evaluation/)
#      would give; reported for completeness, not as a voice-cloning-
#      fidelity number (see §6.6 note added alongside this table).
#   4. Computes duration compliance against the TRUE source English
#      duration.
#
# Usage (from the s2st_en_mr/ project root):
#   python -m scripts.run_real_translation_eval
# Output: samples/real_translation_eval/results.csv, results.md,
#         summary.json, baseline_NN.wav, enhanced_NN.wav
# ---------------------------------------------------------------------------

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from baseline.pipeline import S2STPipeline
from eval.asr_bleu import asr_bleu_from_texts, corpus_asr_bleu
from eval.chrf import chrf_from_texts, corpus_chrf
from eval.f0_metrics import compute_f0_metrics
from eval.length_compliance import compute_duration_compliance, summarize_length_compliance
from eval.mcd import compute_mcd
from eval.speaker_sim import SpeakerSimStage

IN_DIR = Path("samples/real_translation_eval")
OUT_DIR = IN_DIR

# Same held-constant Marathi cloning reference used throughout this project
# (samples/tts_validation/, samples/research_comparison/) — isolates the
# effect of the input content, not the cloned voice. SYSPIN utterance id:
# IISc_SYSPINProject_mr_f_AGRI_00004.
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


def load_mono(path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def main() -> None:
    with open(IN_DIR / "metadata.csv", newline="", encoding="utf-8") as f:
        meta_rows = list(csv.DictReader(f))

    pipeline = S2STPipeline(config_path="configs/hardware.yaml")

    rows = []
    audio_cache: dict[int, dict] = {}
    for m in meta_rows:
        i = int(m["sample"])
        print(f"[{i}/{len(meta_rows)}] fleurs_id={m['fleurs_id']} running full ASR->MT->TTS+DSP pipeline...")

        def status(stage, state, msg):
            print(f"    [{stage:4s}] {state:8s} {msg[:80]}")

        result = pipeline.run(
            source_audio_path=m["en_source_path"],
            reference_speaker_audio_path=REF_AUDIO,
            reference_text=REF_TEXT,
            mode="baseline+research",
            status_callback=status,
        )
        if result.error:
            print(f"    PIPELINE FAILED: {result.error}")
            continue

        baseline_path = Path(result.baseline_tts_audio_path)
        enhanced_path = Path(result.tts_audio_path)
        saved_baseline = OUT_DIR / f"baseline_{i:02d}.wav"
        saved_enhanced = OUT_DIR / f"enhanced_{i:02d}.wav"
        baseline_audio, baseline_sr = load_mono(baseline_path)
        enhanced_audio, enhanced_sr = load_mono(enhanced_path)
        sf.write(saved_baseline, baseline_audio, baseline_sr)
        sf.write(saved_enhanced, enhanced_audio, enhanced_sr)

        mr_ref_audio, mr_ref_sr = load_mono(m["mr_reference_path"])
        audio_cache[i] = {
            "baseline": (baseline_audio, baseline_sr),
            "enhanced": (enhanced_audio, enhanced_sr),
            "mr_ref": (mr_ref_audio, mr_ref_sr),
        }

        row = {
            "sample": i,
            "fleurs_id": m["fleurs_id"],
            "en_source_corpus": m["en_source_corpus"],
            "en_speaker_gender": m["en_speaker_gender"],
            "en_duration_sec": float(m["en_duration_sec"]),
            "en_text": m["en_text"],
            "asr_text": result.asr_text,
            "mt_text": result.mt_text,
            "mr_reference_text": m["mr_reference_text"],
            "baseline_path": str(saved_baseline),
            "enhanced_path": str(saved_enhanced),
            "baseline_duration_sec": round(len(baseline_audio) / baseline_sr, 2),
            "enhanced_duration_sec": round(len(enhanced_audio) / enhanced_sr, 2),
        }

        # --- Translation adequacy: MT output text vs. REAL human Marathi reference ---
        bleu = asr_bleu_from_texts(result.mt_text, m["mr_reference_text"])
        chrf = chrf_from_texts(result.mt_text, m["mr_reference_text"])
        row["mt_bleu_vs_human_ref"] = round(bleu.score, 2)
        row["mt_chrf_vs_human_ref"] = round(chrf.score, 2)

        # --- Duration compliance vs. the TRUE English source duration ---
        dc_base = compute_duration_compliance(row["baseline_duration_sec"], row["en_duration_sec"])
        dc_enh = compute_duration_compliance(row["enhanced_duration_sec"], row["en_duration_sec"])
        row["duration_ratio_baseline"] = round(dc_base.duration_ratio, 3)
        row["duration_ratio_enhanced"] = round(dc_enh.duration_ratio, 3)
        row["compliant_baseline"] = dc_base.compliant
        row["compliant_enhanced"] = dc_enh.compliant

        # --- MCD / F0 vs. REAL FLEURS Marathi reference recording (independent speaker) ---
        mcd_base = compute_mcd(baseline_audio, baseline_sr, mr_ref_audio, mr_ref_sr)
        mcd_enh = compute_mcd(enhanced_audio, enhanced_sr, mr_ref_audio, mr_ref_sr)
        row["mcd_baseline_db"] = round(mcd_base.mcd_db, 2)
        row["mcd_enhanced_db"] = round(mcd_enh.mcd_db, 2)

        f0_base = compute_f0_metrics(baseline_audio, baseline_sr, mr_ref_audio, mr_ref_sr)
        f0_enh = compute_f0_metrics(enhanced_audio, enhanced_sr, mr_ref_audio, mr_ref_sr)
        row["f0_rmse_baseline_hz"] = round(f0_base.f0_rmse_hz, 1)
        row["f0_rmse_enhanced_hz"] = round(f0_enh.f0_rmse_hz, 1)
        row["f0_corr_baseline"] = round(f0_base.f0_correlation, 3)
        row["f0_corr_enhanced"] = round(f0_enh.f0_correlation, 3)

        rows.append(row)

    # ---- Round-trip ASR intelligibility check on both outputs (sequential GPU, after TTS/MT unloaded) ----
    from baseline.asr import ASRStage

    config = pipeline.config
    asr = ASRStage(config["asr"], cache_dir=config["paths"]["cache_dir"])
    print("Loading ASR for round-trip intelligibility checks...")
    asr.load()
    for row in rows:
        row["baseline_roundtrip_asr"] = asr.transcribe(row["baseline_path"], language="mr")
        row["enhanced_roundtrip_asr"] = asr.transcribe(row["enhanced_path"], language="mr")
    asr.unload()
    del asr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- ECAPA speaker similarity vs. REAL FLEURS Marathi reference (independent speaker) ----
    speaker_stage = SpeakerSimStage(cache_dir=config["paths"]["cache_dir"])
    print("Loading ECAPA-TDNN...")
    speaker_stage.load()
    for row in rows:
        i = row["sample"]
        cached = audio_cache[i]
        mr_ref_audio, mr_ref_sr = cached["mr_ref"]
        baseline_audio, baseline_sr = cached["baseline"]
        enhanced_audio, enhanced_sr = cached["enhanced"]
        spk_base = speaker_stage.similarity(baseline_audio, baseline_sr, mr_ref_audio, mr_ref_sr)
        spk_enh = speaker_stage.similarity(enhanced_audio, enhanced_sr, mr_ref_audio, mr_ref_sr)
        row["speaker_sim_baseline"] = round(spk_base.cosine_similarity, 3)
        row["speaker_sim_enhanced"] = round(spk_enh.cosine_similarity, 3)
    speaker_stage.unload()
    del speaker_stage
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No samples completed successfully — nothing to write.")

    # ---- Corpus-level aggregates ------------------------------------------------------
    corpus_bleu_score = corpus_asr_bleu([r["mt_text"] for r in rows], [r["mr_reference_text"] for r in rows])
    corpus_chrf_score = corpus_chrf([r["mt_text"] for r in rows], [r["mr_reference_text"] for r in rows])
    length_summary_base = summarize_length_compliance(
        [compute_duration_compliance(r["baseline_duration_sec"], r["en_duration_sec"]) for r in rows]
    )
    length_summary_enh = summarize_length_compliance(
        [compute_duration_compliance(r["enhanced_duration_sec"], r["en_duration_sec"]) for r in rows]
    )

    summary = {
        "n_samples": len(rows),
        "corpus_mt_bleu_vs_human_ref": corpus_bleu_score,
        "corpus_mt_chrf_vs_human_ref": corpus_chrf_score,
        "mean_mcd_db_baseline": float(np.mean([r["mcd_baseline_db"] for r in rows])),
        "mean_mcd_db_enhanced": float(np.mean([r["mcd_enhanced_db"] for r in rows])),
        "mean_f0_rmse_hz_baseline": float(np.nanmean([r["f0_rmse_baseline_hz"] for r in rows])),
        "mean_f0_rmse_hz_enhanced": float(np.nanmean([r["f0_rmse_enhanced_hz"] for r in rows])),
        "mean_f0_corr_baseline": float(np.nanmean([r["f0_corr_baseline"] for r in rows])),
        "mean_f0_corr_enhanced": float(np.nanmean([r["f0_corr_enhanced"] for r in rows])),
        "mean_speaker_sim_baseline": float(np.mean([r["speaker_sim_baseline"] for r in rows])),
        "mean_speaker_sim_enhanced": float(np.mean([r["speaker_sim_enhanced"] for r in rows])),
        "duration_compliance_rate_baseline": length_summary_base.compliance_rate,
        "duration_compliance_rate_enhanced": length_summary_enh.compliance_rate,
        "mean_duration_ratio_baseline": length_summary_base.mean_duration_ratio,
        "mean_duration_ratio_enhanced": length_summary_enh.mean_duration_ratio,
    }

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = OUT_DIR / "results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Real English->Marathi Translation Evaluation (FLEURS, 10 samples)\n\n")
        f.write(
            "Source: Google FLEURS (Conneau et al., 2022; CC-BY-4.0), `en_us` test split "
            "for English input, `mr_in` test split (same FLoRes-101 sentence id) as the "
            "REAL human Marathi reference translation + recording, read by an independent "
            "speaker. Each English clip was run through the full ASR->MT->TTS pipeline "
            "(mode=baseline+research), not read as pre-known Marathi text. Isochrony target "
            "is each sample's own English source duration. Cloning reference voice held "
            "constant across all 10 (SYSPIN IISc_SYSPINProject_mr_f_AGRI_00004).\n\n"
            "Caveat: MCD/F0/speaker-similarity below compare synthesized audio to the FLEURS "
            "Marathi reference recording, which is a DIFFERENT speaker from the cloned voice "
            "(FLEURS speakers are not shared across language configs) — these measure "
            "content/prosody proximity to a correct human rendition, not voice-cloning "
            "fidelity (see samples/full_evaluation/ for a same-speaker-ground-truth "
            "comparison on a separate, non-translation TTS task).\n\n"
        )
        f.write("## Corpus-level summary\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Corpus BLEU (MT text vs. human reference translation) | {summary['corpus_mt_bleu_vs_human_ref']:.2f} |\n")
        f.write(f"| Corpus chrF++ (MT text vs. human reference translation) | {summary['corpus_mt_chrf_vs_human_ref']:.2f} |\n")
        f.write(f"| Mean MCD dB, lower=better (B/E) | {summary['mean_mcd_db_baseline']:.2f} / {summary['mean_mcd_db_enhanced']:.2f} |\n")
        f.write(f"| Mean F0-RMSE Hz, lower=better (B/E) | {summary['mean_f0_rmse_hz_baseline']:.1f} / {summary['mean_f0_rmse_hz_enhanced']:.1f} |\n")
        f.write(f"| Mean F0 correlation (B/E) | {summary['mean_f0_corr_baseline']:.3f} / {summary['mean_f0_corr_enhanced']:.3f} |\n")
        f.write(f"| Mean ECAPA speaker similarity vs. FLEURS ref (B/E) | {summary['mean_speaker_sim_baseline']:.3f} / {summary['mean_speaker_sim_enhanced']:.3f} |\n")
        f.write(f"| Duration compliance rate vs. true source (B/E) | {length_summary_base.compliance_rate:.0%} / {length_summary_enh.compliance_rate:.0%} |\n")
        f.write(f"| Mean duration ratio vs. true source (B/E) | {summary['mean_duration_ratio_baseline']:.3f} / {summary['mean_duration_ratio_enhanced']:.3f} |\n")

        f.write("\n## Per-sample results\n\n")
        f.write(
            "| # | FLEURS id | EN speaker | EN dur (s) | MT BLEU | MT chrF++ | "
            "MCD dB (B/E) | F0-RMSE Hz (B/E) | F0-corr (B/E) | Spk-sim (B/E) | Dur ratio (B/E) |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['sample']} | {r['fleurs_id']} | {r['en_speaker_gender']} | {r['en_duration_sec']:.2f} | "
                f"{r['mt_bleu_vs_human_ref']} | {r['mt_chrf_vs_human_ref']} | "
                f"{r['mcd_baseline_db']}/{r['mcd_enhanced_db']} | "
                f"{r['f0_rmse_baseline_hz']}/{r['f0_rmse_enhanced_hz']} | "
                f"{r['f0_corr_baseline']}/{r['f0_corr_enhanced']} | "
                f"{r['speaker_sim_baseline']}/{r['speaker_sim_enhanced']} | "
                f"{r['duration_ratio_baseline']:.2f}/{r['duration_ratio_enhanced']:.2f} |\n"
            )

        f.write("\n## ASR / MT text per sample\n\n")
        for r in rows:
            f.write(f"**Sample {r['sample']}** (FLEURS id {r['fleurs_id']})\n\n")
            f.write(f"- EN source text (FLEURS transcript): {r['en_text']}\n")
            f.write(f"- ASR transcription: {r['asr_text']}\n")
            f.write(f"- MT output (Marathi): {r['mt_text']}\n")
            f.write(f"- Human reference translation (FLEURS mr_in): {r['mr_reference_text']}\n\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
