# scripts/generate_research_comparison_10samples.py
# ---------------------------------------------------------------------------
# Extends the single-example baseline-vs-research comparison in
# TECHNICAL_REPORT.md §6.3 to all 10 samples from
# samples/tts_validation/ (real Marathi sentences, one per topic domain).
# For each: applies the DSP prosody transfer + isochrony correction
# (prosody/transfer.py, isochrony/rate_controller.py) using the same
# validated English source clip as the prosody/duration reference, then
# round-trip ASR-checks both baseline and enhanced audio.
#
# Produces ONE summary comparison chart (F0 mean + duration ratio,
# baseline vs enhanced, across all 10 samples) plus a results table.
#
# Usage (from the s2st_en_mr/ project root):
#   python -m scripts.generate_research_comparison_10samples
# Output: samples/research_comparison/10samples_results.csv/.md,
#         samples/research_comparison/enhanced_NN_<domain>.wav,
#         figures/research_comparison_10samples.png
# ---------------------------------------------------------------------------

from __future__ import annotations

import csv
import gc
import os
from pathlib import Path

import numpy as np
import soundfile as sf

import torch

from baseline.pipeline import load_hardware_config
from isochrony.duration_predictor import predict_target_duration_sec
from isochrony.rate_controller import apply_isochrony_correction
from prosody.transfer import transfer_prosody

TTS_SAMPLES_DIR = Path("samples/tts_validation")
# Fixed 12-s English source clip used as the prosody/duration target for the
# SYSPIN DSP stress test (paper Table 6, Fig. 4). This clip is NOT
# redistributed in this repository because its licence could not be
# documented; to re-run the stress test, supply any English speech clip via
# the S2ST_STRESS_SOURCE environment variable (results will differ).
SOURCE_AUDIO = Path(os.environ.get("S2ST_STRESS_SOURCE", "samples/source_audio/srk_test_clip_12s.wav"))
OUT_DIR = Path("samples/research_comparison")


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_hardware_config("configs/hardware.yaml")

    sample_paths = sorted(TTS_SAMPLES_DIR.glob("sample_*.wav"))
    if not sample_paths:
        raise FileNotFoundError(
            f"No samples found in {TTS_SAMPLES_DIR} — run "
            "scripts.generate_tts_validation_samples first."
        )

    source_audio, source_sr = load_mono(SOURCE_AUDIO)
    target_duration = predict_target_duration_sec(str(SOURCE_AUDIO))

    results = []
    for i, baseline_path in enumerate(sample_paths, start=1):
        domain = baseline_path.stem.split("_", 2)[-1]
        print(f"[{i}/{len(sample_paths)}] ({domain}) applying prosody + isochrony...")

        baseline_audio, baseline_sr = load_mono(baseline_path)

        prosody_result = transfer_prosody(source_audio, source_sr, baseline_audio, baseline_sr)
        iso_result = apply_isochrony_correction(prosody_result.audio, baseline_sr, target_duration)

        final_audio = iso_result.audio
        peak = float(np.max(np.abs(final_audio))) if final_audio.size else 0.0
        if peak > 1.0:
            final_audio = final_audio / peak * 0.98

        enhanced_path = OUT_DIR / f"enhanced_{i:02d}_{domain}.wav"
        sf.write(str(enhanced_path), final_audio, samplerate=baseline_sr)

        results.append(
            {
                "sample": i,
                "domain": domain.upper(),
                "baseline_path": str(baseline_path),
                "enhanced_path": str(enhanced_path),
                "baseline_f0_mean_hz": round(prosody_result.target_f0_stats_before["mean_hz"], 1),
                "enhanced_f0_mean_hz": round(prosody_result.target_f0_stats_after["mean_hz"], 1),
                "source_f0_mean_hz": round(prosody_result.source_f0_stats["mean_hz"], 1),
                "baseline_duration_sec": round(len(baseline_audio) / baseline_sr, 2),
                "enhanced_duration_sec": round(len(final_audio) / baseline_sr, 2),
                "duration_ratio_before": round(iso_result.duration_ratio_before, 3),
                "duration_ratio_after": round(iso_result.duration_ratio_after, 3),
                "compliant_before": iso_result.compliant_before,
                "compliant_after": iso_result.compliant_after,
                "stretch_applied": iso_result.stretch_applied,
            }
        )

    # ---- Round-trip ASR check on both baseline and enhanced (sequential GPU) ----
    from baseline.asr import ASRStage

    asr = ASRStage(config["asr"], cache_dir=config["paths"]["cache_dir"])
    print("Loading ASR for round-trip checks...")
    asr.load()
    for r in results:
        r["baseline_asr"] = asr.transcribe(r["baseline_path"], language="mr")
        r["enhanced_asr"] = asr.transcribe(r["enhanced_path"], language="mr")
        print(f"[{r['sample']}] round-trip ASR done ({r['domain']})")
    asr.unload()
    del asr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Write results table ------------------------------------------------------
    csv_path = OUT_DIR / "10samples_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    md_path = OUT_DIR / "10samples_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Baseline vs. Baseline+Research: 10-Sample Comparison\n\n")
        f.write(
            f"Source reference for prosody/duration: `{SOURCE_AUDIO}` "
            f"(F0 mean {results[0]['source_f0_mean_hz']}Hz, duration {target_duration:.2f}s). "
            "Same 10 TTS samples as samples/tts_validation/, each run through "
            "the DSP prosody+isochrony baseline (TECHNICAL_REPORT.md §6).\n\n"
        )
        f.write(
            "| # | Domain | Baseline F0 (Hz) | Enhanced F0 (Hz) | Baseline dur (s) | "
            "Enhanced dur (s) | Ratio before | Ratio after | Compliant after | Stretched |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(
                f"| {r['sample']} | {r['domain']} | {r['baseline_f0_mean_hz']} | "
                f"{r['enhanced_f0_mean_hz']} | {r['baseline_duration_sec']} | "
                f"{r['enhanced_duration_sec']} | {r['duration_ratio_before']} | "
                f"{r['duration_ratio_after']} | {r['compliant_after']} | {r['stretch_applied']} |\n"
            )
        f.write("\n## Round-trip ASR (intelligibility check)\n\n")
        f.write("| # | Domain | Baseline round-trip | Enhanced round-trip |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['sample']} | {r['domain']} | {r['baseline_asr']} | {r['enhanced_asr']} |\n")

        n_compliant_before = sum(r["compliant_before"] for r in results)
        n_compliant_after = sum(r["compliant_after"] for r in results)
        f.write(
            f"\n**Isochrony compliance: {n_compliant_before}/10 before -> "
            f"{n_compliant_after}/10 after.**\n"
        )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

    # ---- Summary chart --------------------------------------------------------------
    _make_chart(results, target_duration)


def _make_chart(results: list[dict], target_duration: float) -> None:
    import matplotlib.pyplot as plt

    INK = "#0b0b0b"
    MUTED = "#898781"
    GRID = "#e1e0d9"
    SURFACE = "#fcfcfb"
    BASELINE_COLOR = "#eb6834"  # orange
    ENHANCED_COLOR = "#0ca30c"  # good green
    SOURCE_COLOR = "#4a3aa7"  # violet

    domains = [r["domain"][:4].title() for r in results]
    x = np.arange(len(domains))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.bar(x - width / 2, [r["baseline_f0_mean_hz"] for r in results], width, color=BASELINE_COLOR, label="Baseline")
    ax.bar(x + width / 2, [r["enhanced_f0_mean_hz"] for r in results], width, color=ENHANCED_COLOR, label="Enhanced")
    ax.axhline(results[0]["source_f0_mean_hz"], color=SOURCE_COLOR, linewidth=1.5, linestyle="--", label="Source (English)")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=40, ha="right", color=INK, fontsize=8)
    ax.set_ylabel("F0 mean (Hz)", color=INK, fontsize=10)
    ax.set_title("Pitch register: baseline vs. enhanced", color=INK, fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.5)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    ax.bar(x - width / 2, [r["duration_ratio_before"] for r in results], width, color=BASELINE_COLOR, label="Baseline")
    ax.bar(x + width / 2, [r["duration_ratio_after"] for r in results], width, color=ENHANCED_COLOR, label="Enhanced")
    ax.axhline(1.0, color=SOURCE_COLOR, linewidth=1.5, linestyle="--", label="Perfect isochrony")
    ax.axhspan(0.85, 1.15, color=SOURCE_COLOR, alpha=0.08)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=40, ha="right", color=INK, fontsize=8)
    ax.set_ylabel("Duration ratio (output / source)", color=INK, fontsize=10)
    ax.set_title("Isochrony: baseline vs. enhanced (shaded = compliant band)", color=INK, fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.5)

    fig.suptitle(
        "Baseline vs. Baseline+Prosody+Isochrony: 10-sample comparison",
        color=INK, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    Path("figures").mkdir(exist_ok=True)
    out_path = Path("figures/research_comparison_10samples.png")
    fig.savefig(out_path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
