from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.platforms.base import OrderBook, OrderSide
from src.strategies.arbitrage_v1_plumbing import (
    ArbV1RiskConfig,
    ArbV1RiskGate,
    OpportunityV1,
    OrderbookExecutablePricer,
    RiskAction,
    RiskReason,
    estimate_executable_price,
    snapshot_age_ms,
)


def _book(age_ms: int = 0) -> OrderBook:
    ts = datetime.now(timezone.utc) - timedelta(milliseconds=age_ms)
    return OrderBook(
        market_id="m1",
        bids=[(Decimal("0.48"), 100), (Decimal("0.47"), 200)],
        asks=[(Decimal("0.52"), 100), (Decimal("0.54"), 200)],
        timestamp=ts,
    )


def test_estimate_executable_price_yes_side_depth():
    est = estimate_executable_price(_book(), OrderSide.YES, quantity=150)
    # 100 @ 0.52 + 50 @ 0.54 => 0.5266...
    assert est.liquidity_sufficient is True
    assert est.filled_quantity == 150
    assert est.depth_levels_used == 2
    assert est.executable_price > Decimal("0.526")
    assert est.executable_price < Decimal("0.527")
    assert est.slippage_bps > 100


def test_estimate_executable_price_stale_and_insufficient():
    est = estimate_executable_price(_book(age_ms=1500), OrderSide.YES, quantity=500, stale_after_ms=750)
    assert est.is_stale is True
    assert est.liquidity_sufficient is False
    assert est.filled_quantity == 300


def test_snapshot_age_ms_non_negative():
    age = snapshot_age_ms(_book(age_ms=250))
    assert age >= 200


def test_risk_gate_deny_edge():
    gate = ArbV1RiskGate(ArbV1RiskConfig(min_edge_bps_net_hard=100))
    opp = OpportunityV1(
        opportunity_id="o1",
        market_id="m1",
        side=OrderSide.YES,
        edge_bps_net=90,
        confidence=0.7,
        ttl_ms=500,
        created_at_ms=1,
        expires_at_ms=9999999999999,
    )
    decision = gate.evaluate(opp, requested_size_contracts=10, predicted_slippage_bps=20, snapshot_age_ms_value=100)
    assert decision.action == RiskAction.DENY
    assert decision.reason_code == RiskReason.RISK_EDGE


def test_risk_gate_adjust_exposure():
    gate = ArbV1RiskGate(ArbV1RiskConfig(max_position_contracts=50))
    opp = OpportunityV1(
        opportunity_id="o2",
        market_id="m1",
        side=OrderSide.YES,
        edge_bps_net=130,
        confidence=0.9,
        ttl_ms=500,
        created_at_ms=1,
        expires_at_ms=9999999999999,
        risk_multiplier=Decimal("1.5"),
    )
    decision = gate.evaluate(opp, requested_size_contracts=40, predicted_slippage_bps=50, snapshot_age_ms_value=100)
    assert decision.action == RiskAction.ADJUST
    assert decision.reason_code == RiskReason.RISK_EXPOSURE
    assert decision.size_adjusted_contracts == 50


def test_risk_gate_allow():
    gate = ArbV1RiskGate()
    opp = OpportunityV1(
        opportunity_id="o3",
        market_id="m1",
        side=OrderSide.NO,
        edge_bps_net=150,
        confidence=0.8,
        ttl_ms=500,
        created_at_ms=1,
        expires_at_ms=9999999999999,
    )
    decision = gate.evaluate(opp, requested_size_contracts=20, predicted_slippage_bps=30, snapshot_age_ms_value=100)
    assert decision.action == RiskAction.ALLOW
    assert decision.reason_code == RiskReason.RISK_OK
    assert decision.size_adjusted_contracts == 20


def test_orderbook_executable_pricer_complement_cost():
    pricer = OrderbookExecutablePricer(stale_after_ms=750)
    result = pricer.estimate_complement(_book(), quantity=50)
    assert result.yes_buy.liquidity_sufficient is True
    assert result.no_buy.liquidity_sufficient is True
    assert result.total_cost > 0
    assert result.total_cost < 1.1
