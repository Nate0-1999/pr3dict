"""Tier 2 deep-analysis contract for resolution-advantage markets."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


DECISION_PATHS = {
    "pre_dispute",
    "post_proposal",
    "active_dispute",
    "initiate_dispute",
    "no_trade",
}
RISK_LEVELS = {"low", "medium", "high"}


@dataclass
class Tier2Output:
    p_dispute: float
    p_yes_final: float
    p_no_final: float
    p_invalid_final: float
    confidence: float
    resolution_source_risk: str
    edge_cases: List[str]
    decision_path: str
    no_trade_reason: Optional[str]
    assumptions: List[str]
    prompt_version: str
    model: str
    run_id: str


def tier2_from_dict(payload: Dict[str, Any]) -> Tier2Output:
    """Parse a model payload into the Tier 2 contract object."""
    return Tier2Output(
        p_dispute=float(payload["p_dispute"]),
        p_yes_final=float(payload["p_yes_final"]),
        p_no_final=float(payload["p_no_final"]),
        p_invalid_final=float(payload["p_invalid_final"]),
        confidence=float(payload["confidence"]),
        resolution_source_risk=str(payload["resolution_source_risk"]).lower(),
        edge_cases=[str(v) for v in payload.get("edge_cases", [])],
        decision_path=str(payload["decision_path"]),
        no_trade_reason=payload.get("no_trade_reason"),
        assumptions=[str(v) for v in payload.get("assumptions", [])],
        prompt_version=str(payload["prompt_version"]),
        model=str(payload["model"]),
        run_id=str(payload["run_id"]),
    )


def normalize_final_probabilities(output: Tier2Output) -> Tier2Output:
    """Renormalize final outcome probabilities if model drift is small."""
    total = output.p_yes_final + output.p_no_final + output.p_invalid_final
    if total <= 0:
        raise ValueError("final probability sum must be positive")
    output.p_yes_final = output.p_yes_final / total
    output.p_no_final = output.p_no_final / total
    output.p_invalid_final = output.p_invalid_final / total
    return output


def validate_tier2_output(output: Tier2Output, tolerance: float = 0.01) -> None:
    """Raise ValueError on probability or taxonomy violations."""
    probs = [output.p_dispute, output.p_yes_final, output.p_no_final, output.p_invalid_final, output.confidence]
    if any(p < 0.0 or p > 1.0 for p in probs):
        raise ValueError("probability value out of bounds")

    prob_sum = output.p_yes_final + output.p_no_final + output.p_invalid_final
    if abs(prob_sum - 1.0) > tolerance:
        raise ValueError("final outcome probabilities must sum to 1")

    if output.decision_path not in DECISION_PATHS:
        raise ValueError("invalid decision_path")

    if output.decision_path == "no_trade" and not output.no_trade_reason:
        raise ValueError("no_trade_reason required when decision_path=no_trade")
    if output.decision_path != "no_trade" and output.no_trade_reason:
        raise ValueError("no_trade_reason must be empty when decision_path is not no_trade")
    if output.resolution_source_risk not in RISK_LEVELS:
        raise ValueError("resolution_source_risk must be low, medium, or high")
    if not output.prompt_version:
        raise ValueError("prompt_version is required")
    if not output.model:
        raise ValueError("model is required")
    if not output.run_id:
        raise ValueError("run_id is required")
