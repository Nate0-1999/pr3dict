import pytest

from src.strategies.dispute.signal_engine import (
    ACTIONS,
    SizingConfig,
    choose_action,
    compute_edge,
    compute_position_size_usd,
    compute_stop_loss_price,
)


def test_actions_include_expected_values():
    assert {"ENTER_YES", "ENTER_NO", "EXIT", "HOLD", "NO_TRADE"}.issubset(ACTIONS)


def test_choose_action_prefers_best_edge_when_confident():
    signal = choose_action(
        edge_yes=0.08,
        edge_no=0.02,
        confidence=0.75,
        min_edge=0.03,
        min_confidence=0.6,
        entry_yes_price=0.42,
        stop_loss_pct=0.15,
    )
    assert signal.action == "ENTER_YES"
    assert signal.reason_code == "EDGE_YES"
    assert signal.stop_loss_pct == 0.15
    assert signal.stop_loss_price == pytest.approx(0.357)


def test_choose_action_blocks_low_confidence():
    signal = choose_action(edge_yes=0.08, edge_no=0.07, confidence=0.4, min_edge=0.03, min_confidence=0.6)
    assert signal.action == "NO_TRADE"
    assert signal.reason_code == "LOW_CONFIDENCE"


def test_position_size_respects_caps():
    size = compute_position_size_usd(
        win_probability=0.70,
        payout_multiple=1.0,
        bankroll_usd=10000.0,
        confidence_discount=1.0,
        invalid_discount=1.0,
        liquidity_discount=1.0,
        current_market_exposure_usd=1490.0,
        current_strategy_exposure_usd=1000.0,
        config=SizingConfig(max_market_exposure_usd=1500.0, max_strategy_exposure_usd=5000.0),
    )
    assert size <= 10.0


def test_compute_edge_applies_haircuts():
    edge = compute_edge(prob=0.62, price=0.55, fee_haircut=0.01, slippage_haircut=0.02)
    assert pytest.approx(edge, abs=1e-9) == 0.04


def test_compute_stop_loss_price_validation():
    with pytest.raises(ValueError, match="entry_price"):
        compute_stop_loss_price(0.0, 0.15)
    with pytest.raises(ValueError, match="stop_loss_pct"):
        compute_stop_loss_price(0.5, 1.0)
