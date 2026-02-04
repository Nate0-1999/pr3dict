from pathlib import Path

import pytest

from src.data.database import MarketDatabase


@pytest.mark.asyncio
async def test_resolution_tables_created(tmp_path: Path):
    db = MarketDatabase(db_path=tmp_path / "test_markets.db")
    await db.connect()

    table_names = {
        "analysis_runs",
        "analysis_outputs_t1",
        "analysis_outputs_t2",
        "signals",
        "market_outcomes",
        "calibration_metrics",
    }

    async with db._connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ) as cursor:
        rows = await cursor.fetchall()

    existing = {row[0] for row in rows}
    assert table_names.issubset(existing)

    await db.close()


@pytest.mark.asyncio
async def test_signal_replay_roundtrip(tmp_path: Path):
    db = MarketDatabase(db_path=tmp_path / "test_markets.db")
    await db.connect()

    await db.upsert_market(
        {
            "id": "mkt-1",
            "conditionId": "mkt-1",
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

    run_id = await db.save_analysis_run(
        market_id="mkt-1",
        stage="tier2",
        run_id="run-123",
        model="test-model",
        prompt_version="pv1",
        strategy_version="sv1",
        status="success",
    )

    signal_id = await db.save_signal(
        market_id="mkt-1",
        analysis_run_id=run_id,
        action="ENTER_YES",
        side="yes",
        confidence=0.74,
        edge_yes=0.08,
        edge_no=-0.02,
        edge_selected=0.08,
        yes_price_snapshot=0.45,
        no_price_snapshot=0.55,
        liquidity_snapshot=5000,
        reason_code="EDGE_OK",
        reason_detail="test",
        strategy_version="sv1",
    )

    replay = await db.get_signal_replay(signal_id)
    assert replay is not None
    assert replay["run_id"] == "run-123"
    assert replay["model"] == "test-model"
    assert replay["action"] == "ENTER_YES"

    await db.close()
