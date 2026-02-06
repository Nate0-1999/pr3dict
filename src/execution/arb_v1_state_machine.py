"""
Arbitrage v1 execution state machine.

Keeps lifecycle transitions explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set


class ArbV1State(str, Enum):
    DISCOVERED = "DISCOVERED"
    PRICED_EXECUTABLE = "PRICED_EXECUTABLE"
    RISK_APPROVED = "RISK_APPROVED"
    RISK_REJECTED = "RISK_REJECTED"
    EXECUTION_SUBMITTED = "EXECUTION_SUBMITTED"
    FILLED = "FILLED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FAILED = "FAILED"
    HEDGED_OR_FLATTENED = "HEDGED_OR_FLATTENED"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class TransitionResult:
    from_state: ArbV1State
    to_state: ArbV1State
    valid: bool
    reason: str = ""


class ArbV1StateMachine:
    """Strict transition checker for v1 lifecycle."""

    _allowed: Dict[ArbV1State, Set[ArbV1State]] = {
        ArbV1State.DISCOVERED: {ArbV1State.PRICED_EXECUTABLE},
        ArbV1State.PRICED_EXECUTABLE: {ArbV1State.RISK_APPROVED, ArbV1State.RISK_REJECTED},
        ArbV1State.RISK_APPROVED: {ArbV1State.EXECUTION_SUBMITTED},
        ArbV1State.RISK_REJECTED: {ArbV1State.CLOSED},
        ArbV1State.EXECUTION_SUBMITTED: {ArbV1State.FILLED, ArbV1State.PARTIAL_FILL, ArbV1State.FAILED},
        ArbV1State.FILLED: {ArbV1State.CLOSED},
        ArbV1State.PARTIAL_FILL: {ArbV1State.HEDGED_OR_FLATTENED},
        ArbV1State.HEDGED_OR_FLATTENED: {ArbV1State.CLOSED},
        ArbV1State.FAILED: {ArbV1State.CLOSED},
        ArbV1State.CLOSED: set(),
    }

    def transition(self, from_state: ArbV1State, to_state: ArbV1State) -> TransitionResult:
        allowed = self._allowed.get(from_state, set())
        if to_state in allowed:
            return TransitionResult(from_state=from_state, to_state=to_state, valid=True)
        return TransitionResult(
            from_state=from_state,
            to_state=to_state,
            valid=False,
            reason=f"invalid transition {from_state.value}->{to_state.value}",
        )
