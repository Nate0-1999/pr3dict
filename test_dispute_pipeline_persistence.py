from pathlib import Path

import pytest

from src.data.database import MarketDatabase
from src.strategies.dispute.pipeline import persist_tier1_result, persist_tier2_result


async def _seed_market(db: MarketDatabase, market_id: str = "mkt-pipeline") -> None:
    await db.upsert_market(
        {
            "id": market_id,
            "conditionId": market_id,
            "question": "Will event X happen?",
            "description": "test",
            "resolutionSource": "https://example.com",
            "slug": "event-x",
            "outcomePrices": ["0.45", "0.55"],
            "volumeNum": 1000,
            "liquidityNum": 5000,
            "endDate": "2026-12-31T00:00:00Z",
            "createdAt": "2026-01-01T00:00:00Z",
            "umaResolutionStatus": None,
            "umaBond": None,
            "umaReward": None,
        }
    )


@pytest.mark.asyncio
async def test_persist_tier1_success(tmp_path: Path):
    db = MarketDatabase(db_path=tmp_path / "pipeline.db")
    await db.connect()
    await _seed_market(db)

    payload = {
        "screen_decision": "FLAG",
        "ambiguity_score": 0.62,
        "dispute_prob_prior": 0.41,
        "top_risks": ["source inconsistency"],
        "rationale_short": "Potential interpretation ambiguity.",
        "prompt_version": "t1.v1",
        "model": "gpt-4o-mini",
        "run_id": "tier1-run-1",
    }
    result = await persist_tier1_result(db, "mkt-pipeline", payload, "sv1")
    assert result.status == "success"
    assert result.output_db_id is not None

    async with db._connection.execute("SELECT COUNT(*) FROM analysis_outputs_t1") as cursor:
        count = (await cursor.fetchone())[0]
    assert count == 1
    await db.close()


@pytest.mark.asyncio
async def test_persist_tier1_invalid_payload(tmp_path: Path):
    db = MarketDatabase(db_path=tmp_path / "pipeline.db")
    await db.connect()
    await _seed_market(db)

    payload = {
        "screen_decision": "MAYBE",
        "ambiguity_score": 0.62,
        "dispute_prob_prior": 0.41,
        "top_risks": [],
        "rationale_short": "bad decision",
        "prompt_version": "t1.v1",
        "model": "gpt-4o-mini",
        "run_id": "tier1-run-invalid",
    }
    result = await persist_tier1_result(db, "mkt-pipeline", payload, "sv1")
    assert result.status == "invalid"
    assert result.output_db_id is None

    async with db._connection.execute("SELECT COUNT(*) FROM analysis_outputs_t1") as cursor:
        count = (await cursor.fetchone())[0]
    assert count == 0
    await db.close()


@pytest.mark.asyncio
async def test_persist_tier2_success(tmp_path: Path):
    db = MarketDatabase(db_path=tmp_path / "pipeline.db")
    await db.connect()
    await _seed_market(db)

    payload = {
        "p_dispute": 0.35,
        "p_yes_final": 0.50,
        "p_no_final": 0.45,
        "p_invalid_final": 0.05,
        "confidence": 0.73,
        "resolution_source_risk": "medium",
        "edge_cases": ["timezone boundary"],
        "decision_path": "pre_dispute",
        "no_trade_reason": None,
        "assumptions": ["source remains online"],
        "prompt_version": "t2.v1",
        "model": "gpt-5",
        "run_id": "tier2-run-1",
    }
    result = await persist_tier2_result(db, "mkt-pipeline", payload, "sv1")
    assert result.status == "success"
    assert result.output_db_id is not None
    assert result.normalization_applied is False

    async with db._connection.execute(
        "SELECT COUNT(*), normalization_applied FROM analysis_outputs_t2"
    ) as cursor:
        row = await cursor.fetchone()
    assert row[0] == 1
    assert row[1] == 0
    await db.close()


@pytest.mark.asyncio
async def test_persist_tier2_small_drift_is_normalized(tmp_path: Path):
    db = MarketDatabase(db_path=tmp_path / "pipeline.db")
    await db.connect()
    await _seed_market(db)

    payload = {
        "p_dispute": 0.35,
        "p_yes_final": 0.499,
        "p_no_final": 0.451,
        "p_invalid_final": 0.049,  # sum=0.999 -> normalize
        "confidence": 0.73,
        "resolution_source_risk": "medium",
        "edge_cases": ["timezone boundary"],
        "decision_path": "pre_dispute",
        "no_trade_reason": None,
        "assumptions": ["source remains online"],
        "prompt_version": "t2.v1",
        "model": "gpt-5",
        "run_id": "tier2-run-normalized",
    }
    result = await persist_tier2_result(db, "mkt-pipeline", payload, "sv1")
    assert result.status == "success"
    assert result.output_db_id is not None
    assert result.normalization_applied is True

    async with db._connection.execute(
        "SELECT p_yes_final, p_no_final, p_invalid_final, normalization_applied FROM analysis_outputs_t2 WHERE id = ?",
        (result.output_db_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    assert pytest.approx(row[0] + row[1] + row[2], abs=1e-9) == 1.0
    assert row[3] == 1
    await db.close()


@pytest.mark.asyncio
async def test_persist_tier2_large_drift_is_invalid(tmp_path: Path):
    db = MarketDatabase(db_path=tmp_path / "pipeline.db")
    await db.connect()
    await _seed_market(db)

    payload = {
        "p_dispute": 0.35,
        "p_yes_final": 0.50,
        "p_no_final": 0.45,
        "p_invalid_final": 0.10,  # sum=1.05 -> invalid
        "confidence": 0.73,
        "resolution_source_risk": "medium",
        "edge_cases": [],
        "decision_path": "pre_dispute",
        "no_trade_reason": None,
        "assumptions": [],
        "prompt_version": "t2.v1",
        "model": "gpt-5",
        "run_id": "tier2-run-large-drift",
    }
    result = await persist_tier2_result(db, "mkt-pipeline", payload, "sv1")
    assert result.status == "invalid"
    assert result.output_db_id is None

    async with db._connection.execute("SELECT COUNT(*) FROM analysis_outputs_t2") as cursor:
        count = (await cursor.fetchone())[0]
    assert count == 0
    await db.close()


@pytest.mark.asyncio
async def test_persist_tier2_invalid_payload(tmp_path: Path):
    db = MarketDatabase(db_path=tmp_path / "pipeline.db")
    await db.connect()
    await _seed_market(db)

    payload = {
        "p_dispute": 0.35,
        "p_yes_final": 0.50,
        "p_no_final": 0.45,
        "p_invalid_final": 0.05,
        "confidence": 0.73,
        "resolution_source_risk": "medium",
        "edge_cases": [],
        "decision_path": "no_trade",
        "no_trade_reason": None,
        "assumptions": [],
        "prompt_version": "t2.v1",
        "model": "gpt-5",
        "run_id": "tier2-run-invalid",
    }
    result = await persist_tier2_result(db, "mkt-pipeline", payload, "sv1")
    assert result.status == "invalid"
    assert result.output_db_id is None

    async with db._connection.execute("SELECT COUNT(*) FROM analysis_outputs_t2") as cursor:
        count = (await cursor.fetchone())[0]
    assert count == 0
    await db.close()
