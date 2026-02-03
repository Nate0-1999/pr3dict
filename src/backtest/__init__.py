"""
PR3DICT Backtesting Framework

Simulate strategies against historical data to validate performance
before deploying to live markets.
"""
from .engine import BacktestEngine, BacktestConfig
from .data import HistoricalDataLoader, MarketSnapshot
from .metrics import PerformanceMetrics, calculate_metrics
from .report import BacktestReport, generate_report

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'HistoricalDataLoader',
    'MarketSnapshot',
    'PerformanceMetrics',
    'calculate_metrics',
    'BacktestReport',
    'generate_report',
]
