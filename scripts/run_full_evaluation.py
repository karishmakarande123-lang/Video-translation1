# scripts/run_full_evaluation.py
# ---------------------------------------------------------------------------
# Full formal evaluation (eval/) of Baseline vs. Baseline+Prosody+Isochrony,
# across the same 10 real Marathi sentences used throughout this project
# (samples/tts_validation/, samples/research_comparison/).
#
# Uses REAL ground-truth recordings: the 10 target sentences are drawn from
# the IISc SYSPIN dataset, and the exact same speaker's real recording of
# each sentence exists in that dataset (confirmed: same sentence IDs are
# present in wav/). This lets MCD, F0-RMSE/correlation, and ECAPA speaker
# similarity compare synthesized speech against genuine human speech of the
# same speaker+text — the standard objective-TTS-eval setup — rather than
# only an internal baseline-vs-enhanced ablation.
#
# BLEU/chrF++ reuse the round-trip ASR transcriptions already computed by
# generate_tts_validation_samples.py / generate_research_comparison_
# 10samples.py (no redundant GPU work). MCD/F0 are CPU-only (librosa,
# pyworld). ECAPA is a small GPU-capable model, loaded/unloaded once for
# all embeddings (sequential discipline, see baseline/pipeline.py).
#
# Usage (from the s2st_en_mr/ project root):
#   python -m scripts.run_full_evaluation
# Output: samples/full_evaluation/results.csv, results.md
# ---------------------------------------------------------------------------

from __future__ import annotations

import csv
import gc
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from eval.asr_bleu import asr_bleu_from_texts, corpus_asr_bleu
from eval.chrf import chrf_from_texts, corpus_chrf
from eval.f0_metrics import compute_f0_metrics
from eval.length_compliance import compute_duration_compliance, summarize_length_compliance
from eval.mcd import compute_mcd
from eval.speaker_sim import SpeakerSimStage

TTS_VALIDATION_DIR = Path("samples/tts_validation")
RESEARCH_DIR = Path("samples/research_comparison")
# Same-speaker ground-truth recordings, included in this repository
# (CC-BY-4.0, see samples/syspin_utterances/README.md).
GROUND_TRUTH_DIR = Path("samples/syspin_utterances")
OUT_DIR = Path("samples/full_evaluation")

NUM_ANNOTATION_RE = re.compile(r"\{[^{}]+\}\{([^{}]+)\}")


def clean_transcript(text: str) -> str:
    """Same cleanup as scripts/generate_tts_validation_samples.py: keep
    only the spoken word-form of IISc's {digit}{word} number annotations."""
    return NUM_ANNOTATION_RE.sub(r"\1", text).strip()


def load_mono(path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(TTS_VALIDATION_DIR / "input_sentences.json", "r", encoding="utf-8") as f:
        sentences = json.load(f)

    with open(RESEARCH_DIR / "10samples_results.csv", newline="", encoding="utf-8") as f:
        research_rows = {row["sample"]: row for row in csv.DictReader(f)}

    # ---- Phase 1: CPU-only metrics (BLEU/chrF++/MCD/F0) ---------------------------
    # Deliberately runs to completion BEFORE ECAPA/speechbrain ever loads: calling
    # librosa.resample() for the first time while speechbrain frames are on the
    # call stack crashes with an unrelated ImportError (confirmed empirically —
    # librosa's lazy_loader walks inspect.stack() during its first-use import,
    # which trips speechbrain's own lazy-module __getattr__ for an optional,
    # uninstalled submodule it has nothing to do with). Finishing all librosa
    # work first, so its lazy imports resolve with a clean call stack, avoids it.
    rows = []
    audio_cache: dict[int, dict] = {}
    for i, (domain, sentence_id, raw_text) in enumerate(sentences, start=1):
        print(f"[{i}/10] ({domain}) CPU metrics (BLEU/chrF++/MCD/F0)...")
        reference_text = clean_transcript(raw_text)
        r = research_rows[str(i)]

        gt_path = GROUND_TRUTH_DIR / f"{sentence_id}.wav"
        gt_audio, gt_sr = load_mono(gt_path)
        baseline_audio, baseline_sr = load_mono(r["baseline_path"])
        enhanced_audio, enhanced_sr = load_mono(r["enhanced_path"])
        audio_cache[i] = {
            "gt": (gt_audio, gt_sr),
            "baseline": (baseline_audio, baseline_sr),
            "enhanced": (enhanced_audio, enhanced_sr),
        }

        row = {"sample": i, "domain": domain, "sentence_id": sentence_id}

        # --- BLEU / chrF++ (reusing already-computed round-trip ASR text) ------
        bleu_base = asr_bleu_from_texts(r["baseline_asr"], reference_text)
        bleu_enh = asr_bleu_from_texts(r["enhanced_asr"], reference_text)
        chrf_base = chrf_from_texts(r["baseline_asr"], reference_text)
        chrf_enh = chrf_from_texts(r["enhanced_asr"], reference_text)
        row["bleu_baseline"] = round(bleu_base.score, 2)
        row["bleu_enhanced"] = round(bleu_enh.score, 2)
        row["chrf_baseline"] = round(chrf_base.score, 2)
        row["chrf_enhanced"] = round(chrf_enh.score, 2)

        # --- MCD vs. real ground-truth recording --------------------------------
        mcd_base = compute_mcd(baseline_audio, baseline_sr, gt_audio, gt_sr)
        mcd_enh = compute_mcd(enhanced_audio, enhanced_sr, gt_audio, gt_sr)
        row["mcd_baseline_db"] = round(mcd_base.mcd_db, 2)
        row["mcd_enhanced_db"] = round(mcd_enh.mcd_db, 2)

        # --- F0-RMSE + correlation vs. real ground-truth recording ------------------
        f0_base = compute_f0_metrics(baseline_audio, baseline_sr, gt_audio, gt_sr)
        f0_enh = compute_f0_metrics(enhanced_audio, enhanced_sr, gt_audio, gt_sr)
        row["f0_rmse_baseline_hz"] = round(f0_base.f0_rmse_hz, 1)
        row["f0_rmse_enhanced_hz"] = round(f0_enh.f0_rmse_hz, 1)
        row["f0_corr_baseline"] = round(f0_base.f0_correlation, 3)
        row["f0_corr_enhanced"] = round(f0_enh.f0_correlation, 3)

        # --- Duration compliance (reuse isochrony ratios already computed) ---------
        row["duration_ratio_baseline"] = float(r["duration_ratio_before"])
        row["duration_ratio_enhanced"] = float(r["duration_ratio_after"])
        row["compliant_baseline"] = r["compliant_before"] == "True"
        row["compliant_enhanced"] = r["compliant_after"] == "True"

        rows.append(row)

    # ---- Phase 2: ECAPA speaker similarity (loaded once, all embeddings) --------
    speaker_stage = SpeakerSimStage(cache_dir=".cache/s2st_en_mr")
    print("Loading ECAPA-TDNN...")
    speaker_stage.load()

    for row in rows:
        i = row["sample"]
        print(f"[{i}/10] ({row['domain']}) speaker similarity...")
        cached = audio_cache[i]
        gt_audio, gt_sr = cached["gt"]
        baseline_audio, baseline_sr = cached["baseline"]
        enhanced_audio, enhanced_sr = cached["enhanced"]

        spk_base = speaker_stage.similarity(baseline_audio, baseline_sr, gt_audio, gt_sr)
        spk_enh = speaker_stage.similarity(enhanced_audio, enhanced_sr, gt_audio, gt_sr)
        row["speaker_sim_baseline"] = round(spk_base.cosine_similarity, 3)
        row["speaker_sim_enhanced"] = round(spk_enh.cosine_similarity, 3)

    speaker_stage.unload()
    del speaker_stage
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Corpus-level aggregates ----------------------------------------------------
    corpus_bleu_base = corpus_asr_bleu([r["baseline_asr"] for r in research_rows.values()],
                                        [clean_transcript(s[2]) for s in sentences])
    corpus_bleu_enh = corpus_asr_bleu([r["enhanced_asr"] for r in research_rows.values()],
                                       [clean_transcript(s[2]) for s in sentences])
    corpus_chrf_base = corpus_chrf([r["baseline_asr"] for r in research_rows.values()],
                                    [clean_transcript(s[2]) for s in sentences])
    corpus_chrf_enh = corpus_chrf([r["enhanced_asr"] for r in research_rows.values()],
                                   [clean_transcript(s[2]) for s in sentences])

    length_summary_base = summarize_length_compliance(
        [compute_duration_compliance(row["duration_ratio_baseline"], 1.0) for row in rows]
    )
    length_summary_enh = summarize_length_compliance(
        [compute_duration_compliance(row["duration_ratio_enhanced"], 1.0) for row in rows]
    )

    # ---- Write CSV --------------------------------------------------------------------
    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- Write corpus-level summary JSON -----------------------------------------------
    # Separate from results.csv's per-sample rows deliberately: corpus BLEU/chrF++ are
    # NOT the mean of per-sentence scores (sacrebleu explicitly warns against that — see
    # eval/asr_bleu.py's corpus_asr_bleu docstring) — a chart/report reading this file
    # instead of naively averaging results.csv's per-sample bleu_* columns avoids
    # reporting a different, incorrect "BLEU" number than this file's correct one.
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "corpus_bleu_baseline": corpus_bleu_base,
                "corpus_bleu_enhanced": corpus_bleu_enh,
                "corpus_chrf_baseline": corpus_chrf_base,
                "corpus_chrf_enhanced": corpus_chrf_enh,
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
            },
            f,
            indent=2,
        )

    # ---- Write Markdown summary ---------------------------------------------------------
    md_path = OUT_DIR / "results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Full Formal Evaluation: Baseline vs. Baseline+Prosody+Isochrony\n\n")
        f.write(
            "10 real Marathi sentences (IISc SYSPIN dataset), evaluated against the "
            "REAL ground-truth recording of the same speaker saying the same sentence "
            "(MCD, F0-RMSE/correlation, ECAPA speaker similarity), and against the "
            "known-correct target text (BLEU/chrF++, via the already-computed round-trip "
            "ASR transcriptions). Duration compliance reuses the isochrony ratios from "
            "the stress test in samples/research_comparison/ — see TECHNICAL_REPORT.md "
            "§6.5 for why those ratios are more extreme than a typical per-utterance run.\n\n"
        )
        f.write("## Corpus-level summary\n\n")
        f.write("| Metric | Baseline | Enhanced |\n|---|---|---|\n")
        f.write(f"| Corpus BLEU | {corpus_bleu_base:.2f} | {corpus_bleu_enh:.2f} |\n")
        f.write(f"| Corpus chrF++ | {corpus_chrf_base:.2f} | {corpus_chrf_enh:.2f} |\n")
        f.write(f"| Mean MCD (dB, lower=better) | {np.mean([r['mcd_baseline_db'] for r in rows]):.2f} | {np.mean([r['mcd_enhanced_db'] for r in rows]):.2f} |\n")
        f.write(f"| Mean F0-RMSE (Hz, lower=better) | {np.nanmean([r['f0_rmse_baseline_hz'] for r in rows]):.1f} | {np.nanmean([r['f0_rmse_enhanced_hz'] for r in rows]):.1f} |\n")
        f.write(f"| Mean F0 correlation | {np.nanmean([r['f0_corr_baseline'] for r in rows]):.3f} | {np.nanmean([r['f0_corr_enhanced'] for r in rows]):.3f} |\n")
        f.write(f"| Mean ECAPA speaker similarity | {np.mean([r['speaker_sim_baseline'] for r in rows]):.3f} | {np.mean([r['speaker_sim_enhanced'] for r in rows]):.3f} |\n")
        f.write(f"| Duration compliance rate | {length_summary_base.compliance_rate:.0%} ({length_summary_base.n_compliant}/{length_summary_base.n_samples}) | {length_summary_enh.compliance_rate:.0%} ({length_summary_enh.n_compliant}/{length_summary_enh.n_samples}) |\n")

        f.write("\n## Per-sample results\n\n")
        f.write(
            "| # | Domain | BLEU (B/E) | chrF++ (B/E) | MCD dB (B/E) | "
            "F0-RMSE Hz (B/E) | F0-corr (B/E) | Speaker sim (B/E) | Dur ratio (B/E) |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['sample']} | {r['domain']} | "
                f"{r['bleu_baseline']}/{r['bleu_enhanced']} | "
                f"{r['chrf_baseline']}/{r['chrf_enhanced']} | "
                f"{r['mcd_baseline_db']}/{r['mcd_enhanced_db']} | "
                f"{r['f0_rmse_baseline_hz']}/{r['f0_rmse_enhanced_hz']} | "
                f"{r['f0_corr_baseline']}/{r['f0_corr_enhanced']} | "
                f"{r['speaker_sim_baseline']}/{r['speaker_sim_enhanced']} | "
                f"{r['duration_ratio_baseline']:.2f}/{r['duration_ratio_enhanced']:.2f} |\n"
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

    _make_chart(summary_path)


def _make_chart(summary_path: Path) -> None:
    """
    Reads corpus-level values from summary.json (NOT results.csv's per-sample
    columns) — corpus BLEU/chrF++ are not the mean of per-sentence scores
    (see eval/asr_bleu.py's corpus_asr_bleu docstring); reading the wrong
    source here would silently show a different, incorrect "BLEU" number
    than results.md's correct one.
    """
    import matplotlib.pyplot as plt

    INK = "#0b0b0b"
    MUTED = "#898781"
    GRID = "#e1e0d9"
    SURFACE = "#fcfcfb"
    BASELINE_COLOR = "#eb6834"
    ENHANCED_COLOR = "#0ca30c"

    with open(summary_path, "r", encoding="utf-8") as f:
        s = json.load(f)

    metrics = [
        ("Corpus BLEU\n(higher better)", s["corpus_bleu_baseline"], s["corpus_bleu_enhanced"], False),
        ("Corpus chrF++\n(higher better)", s["corpus_chrf_baseline"], s["corpus_chrf_enhanced"], False),
        ("Mean MCD (dB)\n(lower better)", s["mean_mcd_db_baseline"], s["mean_mcd_db_enhanced"], True),
        ("Mean F0-RMSE (Hz)\n(lower better)", s["mean_f0_rmse_hz_baseline"], s["mean_f0_rmse_hz_enhanced"], True),
        ("Mean F0 correlation\n(higher better)", s["mean_f0_corr_baseline"], s["mean_f0_corr_enhanced"], False),
        ("Mean ECAPA speaker sim.\n(higher better)", s["mean_speaker_sim_baseline"], s["mean_speaker_sim_enhanced"], False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    fig.patch.set_facecolor(SURFACE)
    axes = axes.flatten()

    for ax, (title, base_val, enh_val, lower_better) in zip(axes, metrics):
        ax.set_facecolor(SURFACE)
        bars = ax.bar(["Baseline", "Enhanced"], [base_val, enh_val], color=[BASELINE_COLOR, ENHANCED_COLOR], width=0.5)
        for bar, val in zip(bars, [base_val, enh_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9, color=INK)

        worse = enh_val > base_val if lower_better else enh_val < base_val
        ax.set_title(title, color=INK, fontsize=10)
        if worse:
            ax.text(0.5, 0.92, "enhanced worse", transform=ax.transAxes, ha="center", fontsize=8, color="#d03b3b", style="italic")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.5)

    fig.suptitle(
        "Full formal evaluation, 10-sample corpus/mean: Baseline vs. Baseline+Prosody+Isochrony\n"
        "(vs. real ground-truth recordings, stress-test duration target — see TECHNICAL_REPORT.md §6.6)",
        color=INK, fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    out_path = figures_dir / "full_evaluation_10samples.png"
    fig.savefig(out_path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
