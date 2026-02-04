"""Tier 1 fast screening contract for resolution-advantage markets."""
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Tier1Output:
    screen_decision: str  # PASS | FLAG
    ambiguity_score: float
    dispute_prob_prior: float
    top_risks: List[str]
    rationale_short: str
    prompt_version: str
    model: str
    run_id: str


def tier1_from_dict(payload: Dict[str, Any]) -> Tier1Output:
    """Parse a model payload into the Tier 1 contract object."""
    return Tier1Output(
        screen_decision=str(payload["screen_decision"]),
        ambiguity_score=float(payload["ambiguity_score"]),
        dispute_prob_prior=float(payload["dispute_prob_prior"]),
        top_risks=[str(v) for v in payload.get("top_risks", [])],
        rationale_short=str(payload.get("rationale_short", "")),
        prompt_version=str(payload["prompt_version"]),
        model=str(payload["model"]),
        run_id=str(payload["run_id"]),
    )


def validate_tier1_output(output: Tier1Output) -> None:
    """Raise ValueError on contract violations."""
    if output.screen_decision not in {"PASS", "FLAG"}:
        raise ValueError("screen_decision must be PASS or FLAG")
    if not 0.0 <= output.ambiguity_score <= 1.0:
        raise ValueError("ambiguity_score out of bounds")
    if not 0.0 <= output.dispute_prob_prior <= 1.0:
        raise ValueError("dispute_prob_prior out of bounds")
    if len(output.top_risks) > 5:
        raise ValueError("top_risks max length is 5")
    if len(output.rationale_short) > 400:
        raise ValueError("rationale_short max length is 400")
    if not output.prompt_version:
        raise ValueError("prompt_version is required")
    if not output.model:
        raise ValueError("model is required")
    if not output.run_id:
        raise ValueError("run_id is required")
