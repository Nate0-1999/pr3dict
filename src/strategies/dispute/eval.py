"""Evaluation helpers for resolution-advantage backtests and paper runs."""
from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple


@dataclass
class CalibrationSummary:
    sample_size: int
    brier_score: float
    log_loss: float


def brier_score(predicted: float, observed: int) -> float:
    """Binary Brier score for one observation."""
    return (predicted - observed) ** 2


def log_loss(predicted: float, observed: int, eps: float = 1e-12) -> float:
    """Binary log loss for one observation."""
    p = min(max(predicted, eps), 1.0 - eps)
    if observed not in (0, 1):
        raise ValueError("observed must be 0 or 1")
    return -(observed * math.log(p) + (1 - observed) * math.log(1 - p))


def summarize_calibration(observations: Sequence[Tuple[float, int]]) -> CalibrationSummary:
    """Aggregate Brier score and log loss across observations."""
    if not observations:
        return CalibrationSummary(sample_size=0, brier_score=0.0, log_loss=0.0)

    brier_values = [brier_score(p, y) for p, y in observations]
    log_values = [log_loss(p, y) for p, y in observations]
    n = len(observations)
    return CalibrationSummary(
        sample_size=n,
        brier_score=sum(brier_values) / n,
        log_loss=sum(log_values) / n,
    )


def bucket_by_confidence(observations: Iterable[Tuple[float, int]], bucket_size: float = 0.1) -> List[Tuple[str, CalibrationSummary]]:
    """
    Group observations by probability bucket for calibration inspection.

    Returns tuples of (bucket_label, summary).
    """
    if bucket_size <= 0 or bucket_size > 1:
        raise ValueError("bucket_size must be in (0,1]")

    buckets: dict[str, list[Tuple[float, int]]] = {}
    for predicted, observed in observations:
        idx = min(int(predicted / bucket_size), int(1.0 / bucket_size) - 1)
        low = idx * bucket_size
        high = low + bucket_size
        label = f"{low:.1f}-{high:.1f}"
        buckets.setdefault(label, []).append((predicted, observed))

    return [(label, summarize_calibration(values)) for label, values in sorted(buckets.items())]
