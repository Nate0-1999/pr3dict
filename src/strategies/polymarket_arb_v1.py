"""
Polymarket Arbitrage v1 opportunity engine.

Focus: binary complement opportunities using executable depth pricing and
hard risk gates. Designed with ports so pricing/risk internals can be replaced
without changing orchestration code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Awaitable, Callable, List, Optional

from .arb_v1_ports import ExecutablePricerPort, RiskGatePort
from .arbitrage_v1_plumbing import (
    ArbV1RiskConfig,
    ArbV1RiskGate,
    OpportunityV1,
    OrderbookExecutablePricer,
    RiskAction,
)
from .base import Signal, TradingStrategy
from ..platforms.base import Market, OrderBook, OrderSide, Position


@dataclass
class PolymarketArbV1Config:
    """Config for v1 opportunity generation."""

    min_liquidity: Decimal = Decimal("1000")
    max_time_to_resolution_hours: int = 24 * 30
    probe_quantity_contracts: int = 25
    fee_buffer_bps: int = 10
    max_signal_contracts: int = 200


class PolymarketArbitrageV1Strategy(TradingStrategy):
    """Depth-aware Polymarket arb strategy (v1 opportunity engine)."""

    def __init__(
        self,
        orderbook_provider: Callable[[str], Awaitable[OrderBook]],
        config: Optional[PolymarketArbV1Config] = None,
        pricer: Optional[ExecutablePricerPort] = None,
        risk_gate: Optional[RiskGatePort] = None,
    ):
        self.orderbook_provider = orderbook_provider
        self.config = config or PolymarketArbV1Config()
        self.pricer = pricer or OrderbookExecutablePricer(stale_after_ms=750)
        self.risk_gate = risk_gate or ArbV1RiskGate(ArbV1RiskConfig())

    @property
    def name(self) -> str:
        return "polymarket_arb_v1"

    async def scan_markets(self, markets: List[Market]) -> List[Signal]:
        signals: List[Signal] = []
        now = int(time.time() * 1000)

        for market in markets:
            if market.platform != "polymarket":
                continue
            if market.resolved or market.liquidity < self.config.min_liquidity:
                continue

            time_to_close_s = (market.close_time - datetime.now(market.close_time.tzinfo)).total_seconds()
            if time_to_close_s <= 0 or time_to_close_s > self.config.max_time_to_resolution_hours * 3600:
                continue

            try:
                orderbook = await self.orderbook_provider(market.id)
            except Exception:
                continue

            complement = self.pricer.estimate_complement(orderbook, self.config.probe_quantity_contracts)
            if (
                complement.yes_buy.is_stale
                or complement.no_buy.is_stale
                or not complement.yes_buy.liquidity_sufficient
                or not complement.no_buy.liquidity_sufficient
            ):
                continue

            gross_edge_bps = int((1.0 - complement.total_cost) * 10000)
            net_edge_bps = gross_edge_bps - self.config.fee_buffer_bps

            opportunity = OpportunityV1(
                opportunity_id=f"pm_{market.id}_{now}",
                market_id=market.id,
                side=OrderSide.YES,
                edge_bps_net=net_edge_bps,
                confidence=1.0,
                ttl_ms=500,
                created_at_ms=now,
                expires_at_ms=now + 500,
                reasons=["binary_complement_executable"],
            )
            decision = self.risk_gate.evaluate(
                opportunity=opportunity,
                requested_size_contracts=self.config.probe_quantity_contracts,
                predicted_slippage_bps=complement.predicted_slippage_bps,
                snapshot_age_ms_value=max(complement.yes_buy.snapshot_age_ms, complement.no_buy.snapshot_age_ms),
            )
            if decision.action == RiskAction.DENY:
                continue

            # Engine currently consumes single-leg signals; we annotate paired leg requirements in reason.
            signals.append(
                Signal(
                    market_id=market.id,
                    market=market,
                    side=OrderSide.YES,
                    strength=min(1.0, max(0.0, net_edge_bps / 300.0)),
                    reason=(
                        f"arb_v1 executable complement edge={net_edge_bps}bps "
                        f"qty={decision.size_adjusted_contracts} action={decision.action.value} "
                        f"(paired NO leg required)"
                    ),
                    target_price=complement.yes_buy.executable_price,
                    metadata={
                        "opportunity_id": opportunity.opportunity_id,
                        "ttl_ms": opportunity.ttl_ms,
                        "expected_edge_bps": net_edge_bps,
                        "suggested_size": int(min(self.config.max_signal_contracts, decision.size_adjusted_contracts)),
                        "paired_leg": {
                            "side": OrderSide.NO.value,
                            "target_price": str(complement.no_buy.executable_price),
                        },
                    },
                )
            )

        return signals

    async def check_exit(self, position: Position, market: Market) -> Optional[Signal]:
        # v1 focuses on entry plumbing; existing exit behavior remains unchanged.
        return None

    def get_position_size(
        self,
        signal: Signal,
        account_balance: Decimal,
        risk_pct: float = 0.02,
    ) -> int:
        planned = signal.metadata.get("suggested_size")
        if isinstance(planned, int) and planned > 0:
            return planned
        return super().get_position_size(signal, account_balance, risk_pct=risk_pct)
