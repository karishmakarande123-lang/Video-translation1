# gui/components.py
# ---------------------------------------------------------------------------
# Small, stateless helpers used by gui/app.py: validating/saving uploaded
# audio, formatting Server-Sent Events, and shaping a PipelineResult into
# the JSON payload the browser renders. Kept separate from app.py so the
# Flask route bodies stay thin.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import uuid
from pathlib import Path

from werkzeug.datastructures import FileStorage

from baseline.pipeline import PipelineResult

# .webm/.ogg included for browser-recorded audio (MediaRecorder default
# output), not just uploaded files.
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB is generous for a short demo clip


class UploadError(ValueError):
    pass


def save_upload(file: FileStorage, upload_dir: Path) -> str:
    """Validates and saves an uploaded/recorded audio file, returns its saved path."""
    if file is None or file.filename == "":
        raise UploadError("No audio file was provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise UploadError(
            f"Unsupported audio format '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        )

    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    file.save(dest)

    size = dest.stat().st_size
    if size == 0:
        dest.unlink(missing_ok=True)
        raise UploadError("Uploaded audio file is empty.")
    if size > MAX_UPLOAD_BYTES:
        dest.unlink(missing_ok=True)
        raise UploadError("Uploaded audio file exceeds the 50MB limit.")

    return str(dest)


def sse_event(event: str, data: dict) -> str:
    """Formats a single named Server-Sent Event message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def result_to_payload(result: PipelineResult) -> dict:
    """Builds the JSON-serializable payload sent to the browser on completion."""
    metrics: dict = {}
    if result.duration_ratio is not None:
        metrics["duration_ratio"] = result.duration_ratio
    if result.speaker_similarity is not None:
        metrics["speaker_similarity"] = result.speaker_similarity
    if result.prosody_metrics:
        metrics.update(result.prosody_metrics)

    return {
        "asr_text": result.asr_text,
        "mt_text": result.mt_text,
        "tts_sample_rate": result.tts_sample_rate,
        "stage_timings_sec": result.stage_timings_sec,
        # Empty until eval/ modules exist — pipeline.py leaves these fields
        # None in baseline mode, see baseline/pipeline.py PipelineResult.
        "metrics": metrics,
        "has_baseline_audio": bool(result.baseline_tts_audio_path),
        "error": result.error,
    }
