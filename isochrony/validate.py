# isochrony/validate.py
# ---------------------------------------------------------------------------
# Computes the isochrony compliance metric used in PipelineResult.duration_ratio
# and reported in the GUI metrics panel / eval reports.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DurationCompliance:
    output_duration_sec: float
    source_duration_sec: float
    duration_ratio: float  # output / source; 1.0 == perfect isochrony
    tolerance: float
    compliant: bool


def compute_duration_compliance(
    output_duration_sec: float,
    source_duration_sec: float,
    tolerance: float = 0.15,
) -> DurationCompliance:
    ratio = output_duration_sec / source_duration_sec
    return DurationCompliance(
        output_duration_sec=output_duration_sec,
        source_duration_sec=source_duration_sec,
        duration_ratio=ratio,
        tolerance=tolerance,
        compliant=abs(ratio - 1.0) <= tolerance,
    )
