import pytest

from src.strategies.dispute.eval import (
    brier_score,
    bucket_by_confidence,
    log_loss,
    summarize_calibration,
)


def test_single_point_metrics():
    assert brier_score(0.8, 1) == pytest.approx(0.04)
    assert log_loss(0.8, 1) < 0.25


def test_summary_non_empty():
    obs = [(0.8, 1), (0.2, 0), (0.6, 1), (0.3, 0)]
    summary = summarize_calibration(obs)
    assert summary.sample_size == 4
    assert summary.brier_score >= 0
    assert summary.log_loss >= 0


def test_bucketed_summary():
    obs = [(0.82, 1), (0.78, 1), (0.24, 0), (0.28, 0)]
    buckets = bucket_by_confidence(obs, bucket_size=0.2)
    labels = [label for label, _ in buckets]
    assert "0.2-0.4" in labels
    assert "0.8-1.0" in labels
