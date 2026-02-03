"""
PR3DICT: Execution Engine

Atomic multi-leg execution for arbitrage trades on Polygon.
All legs must execute in same block (<30ms) to lock in profit.

VWAP Integration: Validates execution quality before placing orders.
"""

from .parallel_executor import ParallelExecutor, ExecutionStrategy
from .metrics import ExecutionMetrics, MetricsCollector
from .vwap_integration import (
    VWAPTradingGate,
    VWAPEnrichedSignal,
    StrategyVWAPIntegration,
    get_vwap_gate,
)

__all__ = [
    # Execution
    "ParallelExecutor",
    "ExecutionStrategy",
    "ExecutionMetrics",
    "MetricsCollector",
    # VWAP Integration
    "VWAPTradingGate",
    "VWAPEnrichedSignal",
    "StrategyVWAPIntegration",
    "get_vwap_gate",
]
