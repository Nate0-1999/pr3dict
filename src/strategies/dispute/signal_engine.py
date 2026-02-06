"""Signal generation and sizing primitives for resolution-advantage strategy."""
from dataclasses import dataclass
from typing import Optional


ACTIONS = {"ENTER_YES", "ENTER_NO", "EXIT", "HOLD", "NO_TRADE"}


@dataclass
class SignalDecision:
    action: str
    reason_code: str
    confidence: Optional[float] = None
    edge_yes: Optional[float] = None
    edge_no: Optional[float] = None
    edge_selected: Optional[float] = None
    stop_loss_price: Optional[float] = None
    stop_loss_pct: Optional[float] = None


@dataclass
class SizingConfig:
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05
    max_market_exposure_usd: float = 1500.0
    max_strategy_exposure_usd: float = 5000.0
    stop_loss_pct: float = 0.15


def compute_edge(prob: float, price: float, fee_haircut: float, slippage_haircut: float) -> float:
    """Simple expected-value edge after execution haircuts."""
    return prob - price - fee_haircut - slippage_haircut


def validate_action(action: str) -> None:
    if action not in ACTIONS:
        raise ValueError(f"unsupported action: {action}")


def compute_stop_loss_price(entry_price: float, stop_loss_pct: float) -> float:
    """Compute stop-loss trigger from entry price and fractional drawdown."""
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    if stop_loss_pct <= 0 or stop_loss_pct >= 1:
        raise ValueError("stop_loss_pct must be in (0,1)")
    return max(0.0, entry_price * (1.0 - stop_loss_pct))


def choose_action(
    edge_yes: float,
    edge_no: float,
    confidence: float,
    min_edge: float,
    min_confidence: float,
    entry_yes_price: Optional[float] = None,
    entry_no_price: Optional[float] = None,
    stop_loss_pct: Optional[float] = None,
) -> SignalDecision:
    """Choose trading action based on edge and confidence gates."""
    if confidence < min_confidence:
        return SignalDecision(
            action="NO_TRADE",
            reason_code="LOW_CONFIDENCE",
            confidence=confidence,
            edge_yes=edge_yes,
            edge_no=edge_no,
            stop_loss_pct=stop_loss_pct,
        )

    best_edge = max(edge_yes, edge_no)
    if best_edge < min_edge:
        return SignalDecision(
            action="NO_TRADE",
            reason_code="INSUFFICIENT_EDGE",
            confidence=confidence,
            edge_yes=edge_yes,
            edge_no=edge_no,
            stop_loss_pct=stop_loss_pct,
        )

    if edge_yes >= edge_no:
        stop_loss_price = None
        if stop_loss_pct is not None and entry_yes_price is not None:
            stop_loss_price = compute_stop_loss_price(entry_yes_price, stop_loss_pct)
        return SignalDecision(
            action="ENTER_YES",
            reason_code="EDGE_YES",
            confidence=confidence,
            edge_yes=edge_yes,
            edge_no=edge_no,
            edge_selected=edge_yes,
            stop_loss_price=stop_loss_price,
            stop_loss_pct=stop_loss_pct,
        )
    stop_loss_price = None
    if stop_loss_pct is not None and entry_no_price is not None:
        stop_loss_price = compute_stop_loss_price(entry_no_price, stop_loss_pct)
    return SignalDecision(
        action="ENTER_NO",
        reason_code="EDGE_NO",
        confidence=confidence,
        edge_yes=edge_yes,
        edge_no=edge_no,
        edge_selected=edge_no,
        stop_loss_price=stop_loss_price,
        stop_loss_pct=stop_loss_pct,
    )


def compute_position_size_usd(
    win_probability: float,
    payout_multiple: float,
    bankroll_usd: float,
    confidence_discount: float,
    invalid_discount: float,
    liquidity_discount: float,
    current_market_exposure_usd: float,
    current_strategy_exposure_usd: float,
    config: Optional[SizingConfig] = None,
) -> float:
    """
    Fractional Kelly with discounts and hard exposure caps.

    Returns a USD target size capped by per-market and per-strategy limits.
    """
    cfg = config or SizingConfig()
    loss_probability = 1.0 - win_probability
    kelly_full = ((payout_multiple * win_probability) - loss_probability) / max(payout_multiple, 1e-9)
    kelly_fractional = max(0.0, kelly_full) * cfg.kelly_fraction

    discounted_fraction = kelly_fractional * confidence_discount * invalid_discount * liquidity_discount
    discounted_fraction = max(0.0, min(discounted_fraction, cfg.max_position_pct))

    raw_size = bankroll_usd * discounted_fraction
    market_remaining = max(0.0, cfg.max_market_exposure_usd - current_market_exposure_usd)
    strategy_remaining = max(0.0, cfg.max_strategy_exposure_usd - current_strategy_exposure_usd)
    return max(0.0, min(raw_size, market_remaining, strategy_remaining))
