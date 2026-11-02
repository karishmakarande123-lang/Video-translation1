# Prosody-Aware English-to-Marathi Speech-to-Speech Translation with Duration Control

Code, data identifiers and evaluation outputs accompanying the paper
*"Prosody-Aware English-to-Marathi Speech-to-Speech Translation with Duration
Control"* (K. Karande, M. Tayal).

The system is a cascaded English→Marathi S2ST pipeline:

| Stage | Model | Code |
|---|---|---|
| ASR | Faster-Whisper `small` (int8_float16) | `baseline/asr.py` |
| MT | IndicTrans2 `ai4bharat/indictrans2-en-indic-dist-200M` | `baseline/mt.py` |
| TTS | IndicF5 `ai4bharat/IndicF5` (reference-audio voice cloning) | `baseline/tts.py` |
| DSP pitch-register mapping (WORLD) | — | `prosody/transfer.py` |
| DSP isochrony controller (phase-vocoder time stretch, ±15 % tolerance) | — | `isochrony/` |
| Evaluation (BLEU, chrF++, MCD, F0-RMSE/corr, ECAPA-TDNN, duration compliance) | — | `eval/` |

Models are loaded **sequentially** (load → infer → unload → `torch.cuda.empty_cache()`)
so the whole pipeline runs on a 6 GB GPU (tested on an RTX 3060 Laptop GPU).

**What is and is not implemented.** The prosody and duration modules are a
DSP baseline: they match the source speaker's pitch register (log-F0 mean and
spread) and overall duration. The learned cross-lingual prosody-transfer
network, learned duration model, and automatic speaker-drift correction
described as future work in the paper are **not** part of this release.

## 1. Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux

# torch and torchaudio must be installed together, same version, same CUDA index
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`ai4bharat/indictrans2-en-indic-dist-200M` and `ai4bharat/IndicF5` are gated
on Hugging Face: request access on both model pages, create a read token, and
run `hf auth login`. Then pre-fetch the models:

```bash
python -m scripts.download_models
```

Model weights are downloaded from Hugging Face and are not stored in this
repository.

## 2. Reproducing the paper

All commands are run from the repository root.

| Paper item | Command | Outputs |
|---|---|---|
| Table 2, Table 5 — real EN→MR translation, 10 FLEURS samples | `python -m scripts.run_real_translation_eval` (the inputs are already in the repo; `scripts.fetch_real_translation_samples` re-downloads them) | `samples/real_translation_eval/` (`results.md`, `results.csv`, `summary.json`, `baseline_NN.wav`, `enhanced_NN.wav`) |
| Fig. 5 | `python -m scripts.generate_real_translation_figure` | `figures/real_translation_eval_10samples.png` |
| Table 6 — same-speaker (SYSPIN) re-evaluation | `python -m scripts.generate_tts_validation_samples` → `python -m scripts.generate_research_comparison_10samples` → `python -m scripts.run_full_evaluation` | `samples/tts_validation/`, `samples/research_comparison/`, `samples/full_evaluation/` |
| Table 3, Table 4, Figs. 2–3 — dependency fixes and clipping correction | see `requirements.txt` comments and `baseline/tts.py`; figures from `python -m scripts.generate_report_figures` | `figures/clipping_*.png` |
| Fig. 4 (F0 contours), Fig. 6 (TTS validation) | pre-generated from the stress-test and TTS-validation outputs | `figures/prosody_transfer_f0_comparison.png`, `figures/tts_validation_10samples.png` |

The pre-computed outputs used in the paper are included, so the tables can be
checked without re-running anything.

**Note on the SYSPIN stress test (Table 6, Fig. 4).** The enhanced outputs in
`samples/research_comparison/` were produced with a single fixed 12-s English
speech clip as the pitch and duration target. That clip is **not**
redistributed here because its licence could not be documented. To re-run
the stress test, provide any English clip through the `S2ST_STRESS_SOURCE`
environment variable; the numbers will differ from the paper. The FLEURS
evaluation (Table 5) uses each sample's own English source and is fully
reproducible from this repository.

## 3. Data

| File | Contents |
|---|---|
| `data/fleurs_ids.csv` | The 10 FLEURS sample IDs (`en_us` / `mr_in`, test split), speaker gender, duration, file paths |
| `data/syspin_ids.csv` | The 11 IISc SYSPIN utterance IDs and their role |
| `samples/real_translation_eval/english_source/` | The 10 English source recordings (FLEURS `en_us`) |
| `samples/real_translation_eval/marathi_reference/` | The matching human Marathi references (FLEURS `mr_in`) |
| `samples/real_translation_eval/metadata.csv` | Transcripts, reference translations and durations |
| `samples/syspin_utterances/` | 10 SYSPIN target sentences + the voice-cloning reference `IISc_SYSPINProject_mr_f_AGRI_00004.wav` |

### Licences

- **Google FLEURS** — CC-BY-4.0. A. Conneau et al., "FLEURS: Few-shot Learning
  Evaluation of Universal Representations of Speech," IEEE SLT 2022.
- **IISc SYSPIN Marathi Female Spk001** — CC-BY-4.0, © Indian Institute of
  Science, Bengaluru (https://spiredatasets.iisc.ac.in/syspinCorpus). See
  `samples/syspin_utterances/README.md`.
- **Code** — MIT (see `LICENSE`).

## 4. Hardware and environment notes

- Tested on Intel i7-12700H, 16 GB RAM, RTX 3060 Laptop GPU (6 GB), Windows + WSL2.
- `transformers` is pinned to `4.44.2`: newer versions break IndicTrans2's
  custom model code (see comments in `requirements.txt`).
- Install IndicF5 from the AI4Bharat GitHub fork, not the PyPI `f5-tts`
  package (`load_model()` signature mismatch).
- `baseline/tts.py` re-loads ~16 IndicF5 parameters that `from_pretrained()`
  silently skips (`_patch_undertrained_weights`), and peak-normalises the
  output to 0.98 only when it exceeds ±1.0 (prevents 16-bit PCM clipping).
- `TECHNICAL_REPORT.md` contains the full development notes.

## 5. Optional GUI

```bash
python -m gui.app
```

Opens a local Flask interface at http://127.0.0.1:7860 that runs the
pipeline on an uploaded English clip and Marathi reference voice, with live
stage progress (Server-Sent Events). Only one run is allowed at a time.
