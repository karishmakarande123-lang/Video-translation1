# scripts/fetch_real_translation_samples.py
# ---------------------------------------------------------------------------
# Pulls 10 REAL English speech samples + their REAL human-translated Marathi
# counterparts from Google FLEURS (Conneau et al., 2022; CC-BY-4.0), to fix
# the reviewer finding that Sections 5.?/5.? evaluated TTS reading Marathi
# text directly (no ASR/MT involved) rather than an actual English->Marathi
# translation.
#
# FLEURS is built from FLoRes-101 sentences: the same `id` field refers to
# the same underlying sentence across every language config, so en_us/{id}
# and mr_in/{id} are genuine translations of each other, each read aloud by
# a real (independent) speaker in that language, with full provenance
# (dataset, split, speaker gender, duration) available from the dataset
# itself — unlike the project's existing Shah Rukh Khan clip, which has no
# documented source/speaker/license.
#
# Usage (from the s2st_en_mr/ project root):
#   python -m scripts.fetch_real_translation_samples
# Output: samples/real_translation_eval/english_source/*.wav,
#         samples/real_translation_eval/marathi_reference/*.wav,
#         samples/real_translation_eval/metadata.csv
# ---------------------------------------------------------------------------

from __future__ import annotations

import csv
import io
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

OUT_DIR = Path("samples/real_translation_eval")
EN_DIR = OUT_DIR / "english_source"
MR_DIR = OUT_DIR / "marathi_reference"

N_SAMPLES = 10
MIN_DUR, MAX_DUR = 4.0, 13.0  # seconds — keeps clips comparable to the project's existing ~5-12s samples
FLEURS_SR = 16000


def _gender_label(g) -> str:
    # FLEURS feature schema: ClassLabel with names ["male", "female", "other"]
    try:
        return {0: "male", 1: "female", 2: "other"}.get(int(g), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def main() -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    MR_DIR.mkdir(parents=True, exist_ok=True)

    print("Streaming FLEURS en_us (test split)...")
    en_ds = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
    en_ds = en_ds.cast_column("audio", Audio(decode=False))

    candidates = []
    for ex in en_ds:
        dur = ex["num_samples"] / FLEURS_SR
        if MIN_DUR <= dur <= MAX_DUR:
            candidates.append(ex)
        if len(candidates) >= N_SAMPLES * 3:  # overfetch so we can also require an mr_in match
            break

    print(f"Collected {len(candidates)} candidate EN clips in duration range; need matching mr_in ids...")

    print("Streaming FLEURS mr_in (test split) to find matching sentence ids...")
    mr_ds = load_dataset("google/fleurs", "mr_in", split="test", streaming=True)
    mr_ds = mr_ds.cast_column("audio", Audio(decode=False))

    want_ids = {ex["id"] for ex in candidates}
    mr_by_id = {}
    for ex in mr_ds:
        if ex["id"] in want_ids:
            mr_by_id[ex["id"]] = ex
        if len(mr_by_id) >= len(want_ids):
            break

    matched = [ex for ex in candidates if ex["id"] in mr_by_id][:N_SAMPLES]
    if len(matched) < N_SAMPLES:
        raise RuntimeError(f"Only found {len(matched)} EN/MR matched pairs, need {N_SAMPLES}.")

    rows = []
    for i, en_ex in enumerate(matched, start=1):
        fid = en_ex["id"]
        mr_ex = mr_by_id[fid]

        en_audio, en_sr = sf.read(io.BytesIO(en_ex["audio"]["bytes"]))
        mr_audio, mr_sr = sf.read(io.BytesIO(mr_ex["audio"]["bytes"]))

        en_path = EN_DIR / f"{i:02d}_fleurs_en_us_{fid}.wav"
        mr_path = MR_DIR / f"{i:02d}_fleurs_mr_in_{fid}.wav"
        sf.write(en_path, en_audio, en_sr)
        sf.write(mr_path, mr_audio, mr_sr)

        en_dur = len(en_audio) / en_sr
        mr_dur = len(mr_audio) / mr_sr

        rows.append(
            {
                "sample": i,
                "fleurs_id": fid,
                "en_source_path": str(en_path),
                "en_source_corpus": "Google FLEURS (en_us, test split)",
                "en_speaker_gender": _gender_label(en_ex.get("gender")),
                "en_duration_sec": round(en_dur, 2),
                "en_text": en_ex["raw_transcription"],
                "mr_reference_path": str(mr_path),
                "mr_reference_corpus": "Google FLEURS (mr_in, test split)",
                "mr_speaker_gender": _gender_label(mr_ex.get("gender")),
                "mr_duration_sec": round(mr_dur, 2),
                "mr_reference_text": mr_ex["raw_transcription"],
            }
        )
        print(f"[{i}/{N_SAMPLES}] id={fid} en_dur={en_dur:.2f}s mr_dur={mr_dur:.2f}s")

    csv_path = OUT_DIR / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
