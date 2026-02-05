"""
Ports for Polymarket arb v1 components.

These interfaces keep compute-heavy logic swappable (e.g., Rust implementation later)
while strategy orchestration remains in Python.
"""

from __future__ import annotations

from typing import Protocol

from ..platforms.base import OrderBook
from .arbitrage_v1_plumbing import ComplementPricing, OpportunityV1, RiskDecision


class ExecutablePricerPort(Protocol):
    """Boundary for depth-based pricing."""

    def estimate_complement(self, orderbook: OrderBook, quantity: int) -> ComplementPricing:
        """Estimate executable YES+NO cost for a target quantity."""


class RiskGatePort(Protocol):
    """Boundary for risk decisioning."""

    def evaluate(
        self,
        opportunity: OpportunityV1,
        requested_size_contracts: int,
        predicted_slippage_bps: int,
        snapshot_age_ms_value: int,
    ) -> RiskDecision:
        """Evaluate opportunity and return normalized decision."""
