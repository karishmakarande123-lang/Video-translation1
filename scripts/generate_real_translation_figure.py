# scripts/generate_real_translation_figure.py
# ---------------------------------------------------------------------------
# Figure for the real English->Marathi translation evaluation
# (samples/real_translation_eval/summary.json), replacing the old figure
# built from samples/research_comparison/'s single-clip-for-all-ten stress
# test (which never involved ASR/MT at all).
#
# Usage: python -m scripts.generate_real_translation_figure
# Output: figures/real_translation_eval_10samples.png
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BASELINE_COLOR = "#eb6834"
ENHANCED_COLOR = "#0ca30c"

with open("samples/real_translation_eval/summary.json", encoding="utf-8") as f:
    s = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
fig.patch.set_facecolor(SURFACE)
axes = axes.flatten()

# Panel 1: translation adequacy (single value each — MT text doesn't change between modes)
ax = axes[0]
ax.set_facecolor(SURFACE)
vals = [s["corpus_mt_bleu_vs_human_ref"], s["corpus_mt_chrf_vs_human_ref"]]
bars = ax.bar(["BLEU", "chrF++"], vals, color=["#4a3aa7", "#1f8fd6"], width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_title("MT text vs. human reference\n(real FLEURS translation)", color=INK, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(axis="y", color=GRID, linewidth=0.5)

metrics = [
    ("Mean MCD (dB)\n(lower better)", s["mean_mcd_db_baseline"], s["mean_mcd_db_enhanced"], True),
    ("Mean F0-RMSE (Hz)\n(lower better)", s["mean_f0_rmse_hz_baseline"], s["mean_f0_rmse_hz_enhanced"], True),
    ("Mean F0 correlation\n(higher better)", s["mean_f0_corr_baseline"], s["mean_f0_corr_enhanced"], False),
    ("Mean ECAPA speaker sim.\nvs. FLEURS ref (diff. speaker)", s["mean_speaker_sim_baseline"], s["mean_speaker_sim_enhanced"], False),
    ("Mean duration ratio\nvs. true source (target=1.0)", s["mean_duration_ratio_baseline"], s["mean_duration_ratio_enhanced"], None),
]

for ax, (title, base_val, enh_val, lower_better) in zip(axes[1:], metrics):
    ax.set_facecolor(SURFACE)
    bars = ax.bar(["Baseline", "Enhanced"], [base_val, enh_val], color=[BASELINE_COLOR, ENHANCED_COLOR], width=0.5)
    for bar, val in zip(bars, [base_val, enh_val]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    if lower_better is not None:
        worse = enh_val > base_val if lower_better else enh_val < base_val
        if worse:
            ax.text(0.5, 0.92, "enhanced worse", transform=ax.transAxes, ha="center", fontsize=8, color="#d03b3b", style="italic")
    else:
        ax.axhline(1.0, color="#4a3aa7", linewidth=1.2, linestyle="--")
    ax.set_title(title, color=INK, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.5)

fig.suptitle(
    "Real English→Marathi translation evaluation (FLEURS, 10 samples)\n"
    "Full ASR→MT→TTS pipeline on real English audio with documented source/speaker/duration",
    color=INK, fontsize=12, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.90])

out_path = Path("figures/real_translation_eval_10samples.png")
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=300, facecolor=SURFACE)
plt.close(fig)
print(f"Saved {out_path}")
