import pytest

from src.strategies.dispute.tier2 import (
    Tier2Output,
    normalize_final_probabilities,
    tier2_from_dict,
    validate_tier2_output,
)


def test_tier2_valid_payload_parses_and_validates():
    payload = {
        "p_dispute": 0.35,
        "p_yes_final": 0.52,
        "p_no_final": 0.43,
        "p_invalid_final": 0.05,
        "confidence": 0.72,
        "resolution_source_risk": "medium",
        "edge_cases": ["timezone ambiguity"],
        "decision_path": "pre_dispute",
        "no_trade_reason": None,
        "assumptions": ["source remains available"],
        "prompt_version": "t2.v1",
        "model": "gpt-5",
        "run_id": "run-2",
    }
    out = tier2_from_dict(payload)
    validate_tier2_output(out)


def test_tier2_no_trade_requires_reason():
    out = Tier2Output(0.2, 0.6, 0.3, 0.1, 0.7, "low", [], "no_trade", None, [], "v1", "m", "r")
    with pytest.raises(ValueError, match="no_trade_reason"):
        validate_tier2_output(out)


def test_tier2_normalization_fixes_small_drift():
    out = Tier2Output(0.2, 0.5, 0.4, 0.11, 0.7, "low", [], "pre_dispute", None, [], "v1", "m", "r")
    out = normalize_final_probabilities(out)
    assert pytest.approx(out.p_yes_final + out.p_no_final + out.p_invalid_final, abs=1e-9) == 1.0
