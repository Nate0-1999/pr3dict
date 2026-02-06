"""Persistence wiring for Tier 1/Tier 2 contract outputs."""
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import uuid4

from ...data.database import MarketDatabase
from .tier1 import tier1_from_dict, validate_tier1_output
from .tier2 import normalize_final_probabilities, tier2_from_dict, validate_tier2_output

PROBABILITY_SUM_EPSILON = 0.01


@dataclass
class PersistResult:
    stage: str
    status: str
    run_db_id: int
    output_db_id: Optional[int]
    run_id: str
    error_message: Optional[str] = None
    normalization_applied: bool = False


def _derive_run_fields(payload: Dict[str, Any], stage: str) -> tuple[str, str, str]:
    run_id = str(payload.get("run_id") or f"{stage}-{uuid4().hex[:12]}")
    model = str(payload.get("model") or "unknown")
    prompt_version = str(payload.get("prompt_version") or "unknown")
    return run_id, model, prompt_version


async def persist_tier1_result(
    db: MarketDatabase,
    market_id: str,
    payload: Dict[str, Any],
    strategy_version: str,
    latency_ms: Optional[int] = None,
    token_cost_usd: Optional[float] = None,
) -> PersistResult:
    """Validate and persist Tier 1 output with deterministic run metadata."""
    run_id, model, prompt_version = _derive_run_fields(payload, "tier1")
    try:
        output = tier1_from_dict(payload)
        validate_tier1_output(output)
    except (KeyError, TypeError, ValueError) as exc:
        run_db_id = await db.save_analysis_run(
            market_id=market_id,
            stage="tier1",
            run_id=run_id,
            model=model,
            prompt_version=prompt_version,
            strategy_version=strategy_version,
            status="invalid",
            latency_ms=latency_ms,
            token_cost_usd=token_cost_usd,
            error_message=str(exc),
        )
        return PersistResult(
            stage="tier1",
            status="invalid",
            run_db_id=run_db_id,
            output_db_id=None,
            run_id=run_id,
            error_message=str(exc),
        )

    run_db_id = await db.save_analysis_run(
        market_id=market_id,
        stage="tier1",
        run_id=output.run_id,
        model=output.model,
        prompt_version=output.prompt_version,
        strategy_version=strategy_version,
        status="success",
        latency_ms=latency_ms,
        token_cost_usd=token_cost_usd,
    )
    output_db_id = await db.save_tier1_output(
        analysis_run_id=run_db_id,
        market_id=market_id,
        screen_decision=output.screen_decision,
        ambiguity_score=output.ambiguity_score,
        dispute_prob_prior=output.dispute_prob_prior,
        top_risks=output.top_risks,
        rationale_short=output.rationale_short,
    )
    return PersistResult(
        stage="tier1",
        status="success",
        run_db_id=run_db_id,
        output_db_id=output_db_id,
        run_id=output.run_id,
    )


async def persist_tier2_result(
    db: MarketDatabase,
    market_id: str,
    payload: Dict[str, Any],
    strategy_version: str,
    latency_ms: Optional[int] = None,
    token_cost_usd: Optional[float] = None,
) -> PersistResult:
    """Validate and persist Tier 2 output with deterministic run metadata."""
    run_id, model, prompt_version = _derive_run_fields(payload, "tier2")
    normalization_applied = False
    try:
        output = tier2_from_dict(payload)
        final_sum = output.p_yes_final + output.p_no_final + output.p_invalid_final
        drift = abs(final_sum - 1.0)
        if drift > PROBABILITY_SUM_EPSILON:
            raise ValueError(f"final probability sum drift too large: {drift:.6f}")
        if drift > 0.0:
            output = normalize_final_probabilities(output)
            normalization_applied = True
        validate_tier2_output(output, tolerance=1e-9)
    except (KeyError, TypeError, ValueError) as exc:
        run_db_id = await db.save_analysis_run(
            market_id=market_id,
            stage="tier2",
            run_id=run_id,
            model=model,
            prompt_version=prompt_version,
            strategy_version=strategy_version,
            status="invalid",
            latency_ms=latency_ms,
            token_cost_usd=token_cost_usd,
            error_message=str(exc),
        )
        return PersistResult(
            stage="tier2",
            status="invalid",
            run_db_id=run_db_id,
            output_db_id=None,
            run_id=run_id,
            error_message=str(exc),
            normalization_applied=False,
        )

    run_db_id = await db.save_analysis_run(
        market_id=market_id,
        stage="tier2",
        run_id=output.run_id,
        model=output.model,
        prompt_version=output.prompt_version,
        strategy_version=strategy_version,
        status="success",
        latency_ms=latency_ms,
        token_cost_usd=token_cost_usd,
    )
    output_db_id = await db.save_tier2_output(
        analysis_run_id=run_db_id,
        market_id=market_id,
        p_dispute=output.p_dispute,
        p_yes_final=output.p_yes_final,
        p_no_final=output.p_no_final,
        p_invalid_final=output.p_invalid_final,
        confidence=output.confidence,
        resolution_source_risk=output.resolution_source_risk,
        edge_cases=output.edge_cases,
        decision_path=output.decision_path,
        no_trade_reason=output.no_trade_reason,
        assumptions=output.assumptions,
        normalization_applied=normalization_applied,
    )
    return PersistResult(
        stage="tier2",
        status="success",
        run_db_id=run_db_id,
        output_db_id=output_db_id,
        run_id=output.run_id,
        normalization_applied=normalization_applied,
    )
