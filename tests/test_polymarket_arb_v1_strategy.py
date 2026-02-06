from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.platforms.base import Market, OrderBook, OrderSide
from src.strategies.polymarket_arb_v1 import PolymarketArbitrageV1Strategy


def _market(platform: str = "polymarket") -> Market:
    return Market(
        id="m1",
        ticker="TEST-1",
        title="Will test pass?",
        description="test",
        yes_price=Decimal("0.50"),
        no_price=Decimal("0.50"),
        volume=Decimal("10000"),
        liquidity=Decimal("5000"),
        close_time=datetime.now(timezone.utc) + timedelta(days=1),
        resolved=False,
        platform=platform,
    )


async def _provider(_: str) -> OrderBook:
    # Asks for YES-buy, bids for YES-sell (used to derive NO-buy).
    return OrderBook(
        market_id="m1",
        asks=[(Decimal("0.50"), 200), (Decimal("0.51"), 200)],
        bids=[(Decimal("0.53"), 200), (Decimal("0.52"), 200)],
        timestamp=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_scan_markets_emits_paired_signal_metadata():
    strategy = PolymarketArbitrageV1Strategy(orderbook_provider=_provider)
    signals = await strategy.scan_markets([_market()])
    assert len(signals) == 1

    signal = signals[0]
    assert signal.metadata["paired_leg"]["side"] == OrderSide.NO.value
    assert "target_price" in signal.metadata["paired_leg"]
    assert signal.metadata["suggested_size"] > 0
    assert signal.metadata["expected_edge_bps"] > 0


@pytest.mark.asyncio
async def test_scan_markets_skips_non_polymarket():
    strategy = PolymarketArbitrageV1Strategy(orderbook_provider=_provider)
    signals = await strategy.scan_markets([_market(platform="kalshi")])
    assert signals == []


def test_get_position_size_prefers_planned_size():
    strategy = PolymarketArbitrageV1Strategy(orderbook_provider=_provider)
    signal = type("SignalLike", (), {"metadata": {"suggested_size": 17}})()
    size = strategy.get_position_size(signal=signal, account_balance=Decimal("1000"))
    assert size == 17
