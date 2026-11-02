# eval/length_compliance.py
# ---------------------------------------------------------------------------
# Thin batch-eval wrapper around isochrony/validate.py's per-sample
# duration-compliance check — kept here (rather than only inline in
# isochrony/validate.py) to match the original eval/ file layout and to add
# corpus-level aggregation (compliance rate, mean/std ratio) for a results
# table, the way asr_bleu.py/chrf.py add corpus_* aggregation over
# sentence-level sacrebleu scores.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isochrony.validate import DurationCompliance, compute_duration_compliance

__all__ = ["DurationCompliance", "compute_duration_compliance", "LengthComplianceSummary", "summarize_length_compliance"]


@dataclass
class LengthComplianceSummary:
    n_samples: int
    n_compliant: int
    compliance_rate: float
    mean_duration_ratio: float
    std_duration_ratio: float


def summarize_length_compliance(results: list[DurationCompliance]) -> LengthComplianceSummary:
    if not results:
        return LengthComplianceSummary(0, 0, 0.0, 0.0, 0.0)

    ratios = np.array([r.duration_ratio for r in results])
    n_compliant = sum(r.compliant for r in results)
    return LengthComplianceSummary(
        n_samples=len(results),
        n_compliant=n_compliant,
        compliance_rate=n_compliant / len(results),
        mean_duration_ratio=float(np.mean(ratios)),
        std_duration_ratio=float(np.std(ratios)),
    )
