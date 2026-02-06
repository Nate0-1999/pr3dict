"""
PR3DICT: Market Data Module

Market data ingestion, caching, and VWAP analysis.
"""

from .database import MarketDatabase
from .scanner import MarketScanner, run_scanner
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
    # Dispute market ingestion
    "MarketDatabase",
    "MarketScanner",
    "run_scanner",
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
