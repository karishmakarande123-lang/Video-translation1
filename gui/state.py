# gui/state.py
# ---------------------------------------------------------------------------
# In-memory state for the Flask GUI: tracks which pipeline stage is
# currently loaded on the GPU (for the status indicator) and holds the
# per-job registries used to stream progress over Server-Sent Events.
# Everything here is process-local, in-memory state — fine for a single-
# user local demo on one machine, not meant to survive a restart, scale to
# multiple workers, or persist jobs long-term.
#
# Because the GPU only has 6GB of VRAM, at most one pipeline run may be in
# flight at a time; `run_lock` enforces that at the app level, on top of
# the sequential per-stage loading baseline/pipeline.py already does within
# a single run.
# ---------------------------------------------------------------------------

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

from baseline.pipeline import PipelineResult

run_lock = threading.Lock()

_current_stage = "idle"  # "idle" | "asr" | "mt" | "tts"
_current_status = ""
_state_lock = threading.Lock()


@dataclass
class Job:
    job_id: str
    events: "queue.Queue" = field(default_factory=queue.Queue)
    result: Optional[PipelineResult] = None


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def create_job() -> Job:
    job = Job(job_id=uuid.uuid4().hex)
    with _jobs_lock:
        _jobs[job.job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(job_id)


def set_current_stage(stage: str, status: str) -> None:
    global _current_stage, _current_status
    with _state_lock:
        _current_stage = stage
        _current_status = status


def get_current_stage() -> dict:
    with _state_lock:
        return {"stage": _current_stage, "status": _current_status}
