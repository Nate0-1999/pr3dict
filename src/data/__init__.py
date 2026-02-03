"""
PR3DICT: Market Data Module

Market data ingestion, caching, and VWAP analysis.
"""

from .cache import MarketDataCache
from .vwap import (
    VWAPCalculator,
    VWAPValidator,
    VWAPMonitor,
    HistoricalVWAPAnalyzer,
    VWAPResult,
    LiquidityMetrics,
    PriceImpactCurve,
    quick_vwap_check,
)

__all__ = [
    # Caching
    "MarketDataCache",
    # VWAP Analysis
    "VWAPCalculator",
    "VWAPValidator",
    "VWAPMonitor",
    "HistoricalVWAPAnalyzer",
    "VWAPResult",
    "LiquidityMetrics",
    "PriceImpactCurve",
    "quick_vwap_check",
]
