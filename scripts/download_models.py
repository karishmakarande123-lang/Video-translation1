# scripts/download_models.py
# ---------------------------------------------------------------------------
# One-time pre-download of every baseline model into the same cache_dir the
# pipeline stages use (configs/hardware.yaml -> paths.cache_dir). Running
# this separately from scripts/run_baseline.py surfaces network/auth issues
# up front, with download progress bars, instead of mid-pipeline.
#
# Usage (from the s2st_en_mr/ project root):
#   python -m scripts.download_models
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
from pathlib import Path

from baseline.pipeline import load_hardware_config


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # load_hardware_config resolves paths.* to absolute, anchored at the
    # project root — not the process's cwd, which otherwise differs between
    # a terminal `cd`'d into the project vs. an IDE "Run" button and causes
    # files to be written/read from two different locations.
    config = load_hardware_config("configs/hardware.yaml")

    cache_dir = Path(config["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- ASR: faster-whisper ---------------------------------------------------
    from faster_whisper import WhisperModel

    print(f"Downloading faster-whisper '{config['asr']['model_size']}' ...")
    WhisperModel(
        config["asr"]["model_size"],
        device="cpu",  # download-only pass, no need to touch the GPU here
        compute_type="int8",
        download_root=str(cache_dir / "faster_whisper"),
    )

    # --- MT: IndicTrans2 ---------------------------------------------------------
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    mt_name = config["mt"]["model_name"]
    print(f"Downloading MT model '{mt_name}' ...")
    AutoTokenizer.from_pretrained(
        mt_name, trust_remote_code=True, cache_dir=str(cache_dir / "indictrans2")
    )
    AutoModelForSeq2SeqLM.from_pretrained(
        mt_name, trust_remote_code=True, cache_dir=str(cache_dir / "indictrans2")
    )

    # --- TTS: IndicF5 -------------------------------------------------------------
    from transformers import AutoModel

    tts_name = config["tts"]["model_name"]
    print(f"Downloading TTS model '{tts_name}' ...")
    AutoModel.from_pretrained(
        tts_name, trust_remote_code=True, cache_dir=str(cache_dir / "indicf5")
    )

    print(f"\nAll models cached under: {cache_dir.resolve()}")


if __name__ == "__main__":
    main()
