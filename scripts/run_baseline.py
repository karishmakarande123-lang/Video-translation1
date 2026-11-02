# scripts/run_baseline.py
# ---------------------------------------------------------------------------
# CLI entry point to run the baseline ASR->MT->TTS pipeline end to end from
# the command line (no GUI). Useful for smoke-testing the pipeline and for
# batch/debug runs before wiring up gui/app.py.
#
# Usage (run from the s2st_en_mr/ project root):
#   python -m scripts.run_baseline --audio path/to/english.wav ^
#       --ref-audio path/to/marathi_reference.wav ^
#       --ref-text "मराठी संदर्भ मजकूर" --verbose
#
# --ref-text is optional but recommended: without it, TTSStage auto-
# transcribes the reference clip on CPU, which is lower fidelity.
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import logging
import sys

from baseline.pipeline import S2STPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the EN->MR S2ST baseline pipeline.")
    parser.add_argument("--audio", required=True, help="Path to source English speech (wav/mp3).")
    parser.add_argument(
        "--ref-audio", required=True, help="Path to Marathi reference speaker clip for voice cloning."
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Transcript of --ref-audio (recommended; auto-transcribed if omitted).",
    )
    parser.add_argument("--config", default="configs/hardware.yaml", help="Path to hardware.yaml.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    def status_callback(stage: str, status: str, message: str) -> None:
        print(f"[{stage.upper():4s}] {status:8s} {message}")

    pipeline = S2STPipeline(config_path=args.config)
    result = pipeline.run(
        source_audio_path=args.audio,
        reference_speaker_audio_path=args.ref_audio,
        reference_text=args.ref_text,
        status_callback=status_callback,
    )

    if result.error:
        print(f"Pipeline failed: {result.error}", file=sys.stderr)
        return 1

    print("\n--- Results ---")
    print(f"ASR (English):  {result.asr_text}")
    print(f"MT  (Marathi):  {result.mt_text}")
    print(f"TTS audio:      {result.tts_audio_path} ({result.tts_sample_rate} Hz)")
    print(f"Stage timings:  {result.stage_timings_sec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
