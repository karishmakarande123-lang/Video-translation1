# scripts/generate_report_figures.py
# ---------------------------------------------------------------------------
# Generates paper-ready figures documenting the TTS clipping bug found and
# fixed in baseline/tts.py (see TECHNICAL_REPORT.md §3.4c). Uses two real,
# paired synthesis outputs already produced during debugging — identical
# text and reference audio/transcript, differing only in whether the
# peak-normalization fix was applied — so the comparison isolates the fix's
# effect rather than any other variable.
#
# Usage (from the s2st_en_mr/ project root):
#   python -m scripts.generate_report_figures
# Output: figures/clipping_waveform_comparison.png,
#         figures/clipping_spectrogram_comparison.png
# ---------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Palette (validated categorical/status colors — see dataviz skill palette.md)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
CRITICAL = "#d03b3b"  # before fix (clipped)
GOOD = "#0ca30c"  # after fix (normalized)

BEFORE_PATH = ".cache/s2st_en_mr/tts_4c8b2930391e428fac72f51b1083b282.wav"
AFTER_PATH = ".cache/s2st_en_mr/tts_3af3407d37de479ca95d9762bf160ae3.wav"
OUT_DIR = Path("figures")


def audio_stats(data: np.ndarray) -> dict:
    return {
        "peak": float(np.max(np.abs(data))),
        "rms": float(np.sqrt(np.mean(data.astype(np.float64) ** 2))),
        "clipped_pct": float(np.mean(np.abs(data) >= 0.999) * 100),
    }


def plot_waveform_comparison(before, after, sr, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    t_before = np.arange(len(before)) / sr
    t_after = np.arange(len(after)) / sr

    for ax, data, t, color, label, stats in [
        (axes[0], before, t_before, CRITICAL, "Before fix (clipped)", audio_stats(before)),
        (axes[1], after, t_after, GOOD, "After fix (peak-normalized)", audio_stats(after)),
    ]:
        ax.set_facecolor(SURFACE)
        ax.plot(t, data, color=color, linewidth=0.5)
        ax.axhline(1.0, color=MUTED, linewidth=0.75, linestyle="--")
        ax.axhline(-1.0, color=MUTED, linewidth=0.75, linestyle="--")
        ax.set_ylim(-1.15, 1.15)
        ax.set_ylabel("Amplitude", color=INK, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(
            f"{label}  —  peak={stats['peak']:.2f}, RMS={stats['rms']:.2f}, "
            f"clipped={stats['clipped_pct']:.1f}%",
            color=INK,
            fontsize=10,
            loc="left",
        )

    axes[1].set_xlabel("Time (s)", color=INK, fontsize=10)
    fig.suptitle(
        "IndicF5 TTS output: waveform before/after clipping fix",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_spectrogram_comparison(before, after, sr, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, data, label in [(axes[0], before, "Before fix (clipped)"), (axes[1], after, "After fix (normalized)")]:
        mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=80, fmax=sr // 2)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(
            mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma", fmax=sr // 2
        )
        ax.set_facecolor(SURFACE)
        ax.set_title(label, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Time (s)", color=INK, fontsize=10)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    axes[0].set_ylabel("Mel frequency (Hz)", color=INK, fontsize=10)
    axes[1].set_ylabel("")
    cbar = fig.colorbar(img, ax=axes, format="%+2.0f dB", fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle(
        "Mel-spectrogram: clipping shows as broadband harmonic distortion",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    before, sr_b = sf.read(BEFORE_PATH)
    after, sr_a = sf.read(AFTER_PATH)
    assert sr_b == sr_a, "sample rates must match for a fair comparison"

    before = before.astype(np.float32)
    after = after.astype(np.float32)

    plot_waveform_comparison(before, after, sr_b, OUT_DIR / "clipping_waveform_comparison.png")
    plot_spectrogram_comparison(before, after, sr_b, OUT_DIR / "clipping_spectrogram_comparison.png")


if __name__ == "__main__":
    main()
