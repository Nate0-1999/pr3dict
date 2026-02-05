"""
PR3DICT: Polymarket Arbitrage v1 Plumbing

M1 scope:
- executable pricing from order book depth
- snapshot freshness checks
- lightweight risk gate with reason codes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from ..platforms.base import OrderBook, OrderSide


class RiskAction(str, Enum):
    """Risk decision action."""

    ALLOW = "allow"
    ADJUST = "adjust"
    DENY = "deny"


class RiskReason(str, Enum):
    """Normalized reason codes for arb v1 decisions."""

    RISK_OK = "RISK_OK"
    RISK_DAILY_LOSS = "RISK_DAILY_LOSS"
    RISK_EXPOSURE = "RISK_EXPOSURE"
    RISK_SLIPPAGE = "RISK_SLIPPAGE"
    RISK_STALE = "RISK_STALE"
    RISK_EDGE = "RISK_EDGE"
    RISK_UNKNOWN = "RISK_UNKNOWN"


@dataclass(frozen=True)
class OpportunityV1:
    """Minimal contract for opportunity evaluation in v1."""

    opportunity_id: str
    market_id: str
    side: OrderSide
    edge_bps_net: int
    confidence: float
    ttl_ms: int
    created_at_ms: int
    expires_at_ms: int
    risk_multiplier: Decimal = Decimal("1.0")
    reasons: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutablePrice:
    """Executable pricing estimate from depth."""

    market_id: str
    side: OrderSide
    target_quantity: int
    quoted_price: Decimal
    executable_price: Decimal
    slippage_bps: int
    filled_quantity: int
    depth_levels_used: int
    snapshot_age_ms: int
    is_stale: bool
    liquidity_sufficient: bool


@dataclass(frozen=True)
class ComplementPricing:
    """Executable pricing for binary complement validation."""

    yes_buy: ExecutablePrice
    no_buy: ExecutablePrice
    total_cost: float
    predicted_slippage_bps: int


@dataclass(frozen=True)
class RiskDecision:
    """Decision returned by arb v1 risk gate."""

    opportunity_id: str
    action: RiskAction
    size_adjusted_contracts: int
    reason_code: RiskReason
    details: str = ""


@dataclass
class ArbV1RiskConfig:
    """Risk thresholds from master spec."""

    min_edge_bps_net_hard: int = 100
    max_snapshot_age_ms: int = 750
    max_slippage_bps_hard_per_leg: int = 100
    max_position_contracts: int = 100


def epoch_ms(now: Optional[datetime] = None) -> int:
    """Return unix epoch in milliseconds."""
    moment = now or datetime.now(timezone.utc)
    return int(moment.timestamp() * 1000)


def snapshot_age_ms(orderbook: OrderBook, now: Optional[datetime] = None) -> int:
    """Compute snapshot age in milliseconds."""
    moment = now or datetime.now(timezone.utc)
    return max(0, int((moment - orderbook.timestamp).total_seconds() * 1000))


def is_snapshot_stale(orderbook: OrderBook, max_age_ms: int = 750, now: Optional[datetime] = None) -> bool:
    """Snapshot staleness check."""
    return snapshot_age_ms(orderbook, now=now) > max_age_ms


def estimate_executable_price(
    orderbook: OrderBook,
    side: OrderSide,
    quantity: int,
    quoted_price: Optional[Decimal] = None,
    stale_after_ms: int = 750,
    now: Optional[datetime] = None,
) -> ExecutablePrice:
    """
    Estimate executable average price for a requested quantity.

    Mapping follows current strategy conventions:
    - YES consumes asks (buy side)
    - NO consumes bids (sell side equivalent for NO exposure)
    """
    if quantity <= 0:
        raise ValueError("quantity must be > 0")

    age_ms = snapshot_age_ms(orderbook, now=now)
    stale = age_ms > stale_after_ms

    raw_levels = orderbook.asks if side == OrderSide.YES else orderbook.bids
    levels = sorted(raw_levels, key=lambda x: x[0], reverse=side == OrderSide.NO)

    if not levels:
        return ExecutablePrice(
            market_id=orderbook.market_id,
            side=side,
            target_quantity=quantity,
            quoted_price=Decimal("0"),
            executable_price=Decimal("0"),
            slippage_bps=0,
            filled_quantity=0,
            depth_levels_used=0,
            snapshot_age_ms=age_ms,
            is_stale=stale,
            liquidity_sufficient=False,
        )

    reference_price = quoted_price if quoted_price is not None else levels[0][0]

    remaining = quantity
    total_cost = Decimal("0")
    filled = 0
    depth_used = 0

    for price, size in levels:
        if remaining <= 0:
            break
        fill_qty = min(remaining, size)
        total_cost += price * fill_qty
        filled += fill_qty
        remaining -= fill_qty
        depth_used += 1

    executable = total_cost / filled if filled > 0 else Decimal("0")
    if reference_price > 0 and executable > 0:
        slippage_bps = int(abs((executable - reference_price) / reference_price) * 10000)
    else:
        slippage_bps = 0

    return ExecutablePrice(
        market_id=orderbook.market_id,
        side=side,
        target_quantity=quantity,
        quoted_price=reference_price,
        executable_price=executable,
        slippage_bps=slippage_bps,
        filled_quantity=filled,
        depth_levels_used=depth_used,
        snapshot_age_ms=age_ms,
        is_stale=stale,
        liquidity_sufficient=(filled == quantity),
    )


class ArbV1RiskGate:
    """Rule-based risk gate for v1 opportunities."""

    def __init__(self, config: Optional[ArbV1RiskConfig] = None):
        self.config = config or ArbV1RiskConfig()

    def evaluate(
        self,
        opportunity: OpportunityV1,
        requested_size_contracts: int,
        predicted_slippage_bps: int,
        snapshot_age_ms_value: int,
        current_time_ms: Optional[int] = None,
    ) -> RiskDecision:
        """Evaluate opportunity against v1 hard/soft constraints."""
        now_ms = current_time_ms if current_time_ms is not None else epoch_ms()

        if snapshot_age_ms_value > self.config.max_snapshot_age_ms:
            return RiskDecision(
                opportunity_id=opportunity.opportunity_id,
                action=RiskAction.DENY,
                size_adjusted_contracts=0,
                reason_code=RiskReason.RISK_STALE,
                details=f"snapshot_age_ms={snapshot_age_ms_value}",
            )

        if now_ms > opportunity.expires_at_ms:
            return RiskDecision(
                opportunity_id=opportunity.opportunity_id,
                action=RiskAction.DENY,
                size_adjusted_contracts=0,
                reason_code=RiskReason.RISK_STALE,
                details="opportunity expired",
            )

        if opportunity.edge_bps_net < self.config.min_edge_bps_net_hard:
            return RiskDecision(
                opportunity_id=opportunity.opportunity_id,
                action=RiskAction.DENY,
                size_adjusted_contracts=0,
                reason_code=RiskReason.RISK_EDGE,
                details=f"edge_bps_net={opportunity.edge_bps_net}",
            )

        if predicted_slippage_bps > self.config.max_slippage_bps_hard_per_leg:
            return RiskDecision(
                opportunity_id=opportunity.opportunity_id,
                action=RiskAction.DENY,
                size_adjusted_contracts=0,
                reason_code=RiskReason.RISK_SLIPPAGE,
                details=f"predicted_slippage_bps={predicted_slippage_bps}",
            )

        scaled_size = int(Decimal(str(requested_size_contracts)) * opportunity.risk_multiplier)
        scaled_size = max(1, scaled_size)

        if scaled_size > self.config.max_position_contracts:
            return RiskDecision(
                opportunity_id=opportunity.opportunity_id,
                action=RiskAction.ADJUST,
                size_adjusted_contracts=self.config.max_position_contracts,
                reason_code=RiskReason.RISK_EXPOSURE,
                details=f"scaled_size={scaled_size}",
            )

        return RiskDecision(
            opportunity_id=opportunity.opportunity_id,
            action=RiskAction.ALLOW,
            size_adjusted_contracts=scaled_size,
            reason_code=RiskReason.RISK_OK,
            details="all checks passed",
        )


class OrderbookExecutablePricer:
    """
    Default Python implementation of executable pricing boundary.

    This class is intentionally small and deterministic so it can be replaced
    by a Rust implementation behind the same method signature later.
    """

    def __init__(self, stale_after_ms: int = 750):
        self.stale_after_ms = stale_after_ms

    def estimate_complement(self, orderbook: OrderBook, quantity: int) -> ComplementPricing:
        """
        Estimate binary complement executable cost.

        YES buy cost uses ask-side depth.
        NO buy cost is derived from YES sell executable price: NO = 1 - YES_sell.
        """
        yes_buy = estimate_executable_price(
            orderbook=orderbook,
            side=OrderSide.YES,
            quantity=quantity,
            stale_after_ms=self.stale_after_ms,
        )
        yes_sell = estimate_executable_price(
            orderbook=orderbook,
            side=OrderSide.NO,
            quantity=quantity,
            stale_after_ms=self.stale_after_ms,
        )

        no_buy_price = Decimal("1.0") - yes_sell.executable_price if yes_sell.executable_price > 0 else Decimal("0")
        no_buy = ExecutablePrice(
            market_id=yes_sell.market_id,
            side=OrderSide.NO,
            target_quantity=yes_sell.target_quantity,
            quoted_price=(Decimal("1.0") - yes_sell.quoted_price) if yes_sell.quoted_price > 0 else Decimal("0"),
            executable_price=no_buy_price,
            slippage_bps=yes_sell.slippage_bps,
            filled_quantity=yes_sell.filled_quantity,
            depth_levels_used=yes_sell.depth_levels_used,
            snapshot_age_ms=yes_sell.snapshot_age_ms,
            is_stale=yes_sell.is_stale,
            liquidity_sufficient=yes_sell.liquidity_sufficient,
        )

        total_cost = float(yes_buy.executable_price + no_buy.executable_price)
        predicted_slippage = max(yes_buy.slippage_bps, no_buy.slippage_bps)

        return ComplementPricing(
            yes_buy=yes_buy,
            no_buy=no_buy,
            total_cost=total_cost,
            predicted_slippage_bps=predicted_slippage,
        )
