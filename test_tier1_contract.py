import pytest

from src.strategies.dispute.tier1 import Tier1Output, tier1_from_dict, validate_tier1_output


def test_tier1_valid_payload_parses_and_validates():
    payload = {
        "screen_decision": "FLAG",
        "ambiguity_score": 0.7,
        "dispute_prob_prior": 0.55,
        "top_risks": ["timing ambiguity", "source risk"],
        "rationale_short": "Potential edge case in resolution wording.",
        "prompt_version": "t1.v1",
        "model": "gpt-4o-mini",
        "run_id": "run-1",
    }
    out = tier1_from_dict(payload)
    validate_tier1_output(out)
    assert out.screen_decision == "FLAG"


def test_tier1_invalid_screen_decision_rejected():
    out = Tier1Output("MAYBE", 0.5, 0.5, [], "ok", "t1.v1", "model", "run-1")
    with pytest.raises(ValueError, match="screen_decision"):
        validate_tier1_output(out)


def test_tier1_too_many_risks_rejected():
    out = Tier1Output("PASS", 0.2, 0.1, ["a", "b", "c", "d", "e", "f"], "ok", "t1.v1", "model", "run-1")
    with pytest.raises(ValueError, match="top_risks"):
        validate_tier1_output(out)
