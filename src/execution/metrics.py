"""
PR3DICT: Execution Metrics

Comprehensive monitoring and analytics for parallel execution engine.

Tracks:
- Fill rate per strategy
- Average execution time
- Slippage vs expected
- Failed arbitrage rate
- Per-leg performance
- Gas costs (Polygon)
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Dict, Optional
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Metrics for a single trade execution."""
    trade_id: str
    timestamp: datetime
    strategy: str  # market/limit/hybrid
    
    # Execution performance
    num_legs: int
    legs_filled: int
    execution_time_ms: float
    committed: bool
    rolled_back: bool
    
    # Financial performance
    expected_profit: Optional[Decimal]
    actual_profit: Optional[Decimal]
    slippage_pct: Optional[Decimal]
    
    # Per-leg details
    leg_times_ms: List[float] = field(default_factory=list)
    leg_statuses: List[str] = field(default_factory=list)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    @property
    def fill_rate(self) -> float:
        """Percentage of legs successfully filled."""
        if self.num_legs == 0:
            return 0.0
        return self.legs_filled / self.num_legs
    
    @property
    def success(self) -> bool:
        """Was trade successfully committed."""
        return self.committed and not self.rolled_back
    
    @property
    def within_block_time(self) -> bool:
        """Did execution complete within 30ms (single Polygon block)."""
        return self.execution_time_ms <= 30.0


class MetricsCollector:
    """
    Collects and aggregates execution metrics.
    
    Provides real-time monitoring and historical analysis.
    """
    
    def __init__(self):
        self._metrics: List[ExecutionMetrics] = []
        
        # Aggregated stats (reset per session)
        self._total_trades = 0
        self._successful_trades = 0
        self._failed_trades = 0
        self._rolled_back_trades = 0
        
        # Strategy-specific stats
        self._strategy_stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "successful": 0,
            "total_exec_time": 0.0,
            "total_slippage": Decimal("0"),
            "within_block": 0,
        })
        
        # Financial tracking
        self._total_expected_profit = Decimal("0")
        self._total_actual_profit = Decimal("0")
        
    def record_trade(self, trade) -> ExecutionMetrics:
        """
        Record metrics from completed trade.
        
        Args:
            trade: MultiLegTrade instance
            
        Returns:
            ExecutionMetrics for this trade
        """
        from .parallel_executor import MultiLegTrade, LegStatus
        
        # Extract metrics
        metrics = ExecutionMetrics(
            trade_id=trade.trade_id,
            timestamp=datetime.now(timezone.utc),
            strategy=trade.strategy.value,
            num_legs=len(trade.legs),
            legs_filled=sum(1 for leg in trade.legs if leg.is_filled),
            execution_time_ms=trade.execution_time_ms or 0.0,
            committed=trade.committed,
            rolled_back=trade.rolled_back,
            expected_profit=trade.expected_profit,
            actual_profit=trade.actual_profit,
            slippage_pct=trade.slippage_pct
        )
        
        # Per-leg details
        for leg in trade.legs:
            if leg.execution_time_ms:
                metrics.leg_times_ms.append(leg.execution_time_ms)
            metrics.leg_statuses.append(leg.status.value)
            if leg.error:
                metrics.errors.append(f"{leg.market_id}: {leg.error}")
        
        # Update aggregates
        self._total_trades += 1
        
        if metrics.success:
            self._successful_trades += 1
        else:
            self._failed_trades += 1
        
        if trade.rolled_back:
            self._rolled_back_trades += 1
        
        # Strategy stats
        strategy = trade.strategy.value
        self._strategy_stats[strategy]["count"] += 1
        if metrics.success:
            self._strategy_stats[strategy]["successful"] += 1
        self._strategy_stats[strategy]["total_exec_time"] += metrics.execution_time_ms
        
        if metrics.slippage_pct:
            self._strategy_stats[strategy]["total_slippage"] += abs(metrics.slippage_pct)
        
        if metrics.within_block_time:
            self._strategy_stats[strategy]["within_block"] += 1
        
        # Financial tracking
        if trade.expected_profit:
            self._total_expected_profit += trade.expected_profit
        if trade.actual_profit:
            self._total_actual_profit += trade.actual_profit
        
        # Store
        self._metrics.append(metrics)
        
        # Log summary
        self._log_trade_metrics(metrics)
        
        return metrics
    
    def _log_trade_metrics(self, metrics: ExecutionMetrics) -> None:
        """Log trade metrics summary."""
        status = "✓ SUCCESS" if metrics.success else "✗ FAILED"
        
        logger.info(
            f"[METRICS] {status} | Trade: {metrics.trade_id} | "
            f"Strategy: {metrics.strategy.upper()} | "
            f"Legs: {metrics.legs_filled}/{metrics.num_legs} | "
            f"Time: {metrics.execution_time_ms:.1f}ms | "
            f"Profit: {metrics.actual_profit or 'N/A'}"
        )
        
        if metrics.slippage_pct:
            logger.info(f"  Slippage: {metrics.slippage_pct:.2%}")
        
        if metrics.errors:
            for error in metrics.errors:
                logger.warning(f"  Error: {error}")
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dictionary with aggregated statistics
        """
        # Overall stats
        success_rate = (self._successful_trades / self._total_trades * 100) if self._total_trades > 0 else 0.0
        
        summary = {
            "total_trades": self._total_trades,
            "successful": self._successful_trades,
            "failed": self._failed_trades,
            "rolled_back": self._rolled_back_trades,
            "success_rate_pct": round(success_rate, 2),
        }
        
        # Financial summary
        if self._total_expected_profit > 0:
            profit_capture_rate = (self._total_actual_profit / self._total_expected_profit) * 100
        else:
            profit_capture_rate = 0.0
        
        summary["financial"] = {
            "total_expected_profit": float(self._total_expected_profit),
            "total_actual_profit": float(self._total_actual_profit),
            "profit_capture_rate_pct": round(float(profit_capture_rate), 2),
        }
        
        # Strategy breakdown
        summary["by_strategy"] = {}
        for strategy, stats in self._strategy_stats.items():
            if stats["count"] == 0:
                continue
            
            avg_time = stats["total_exec_time"] / stats["count"]
            success_rate = (stats["successful"] / stats["count"]) * 100
            block_rate = (stats["within_block"] / stats["count"]) * 100
            avg_slippage = stats["total_slippage"] / stats["count"] if stats["count"] > 0 else Decimal("0")
            
            summary["by_strategy"][strategy] = {
                "count": stats["count"],
                "successful": stats["successful"],
                "success_rate_pct": round(success_rate, 2),
                "avg_execution_time_ms": round(avg_time, 2),
                "within_block_rate_pct": round(block_rate, 2),
                "avg_slippage_pct": round(float(avg_slippage) * 100, 2),
            }
        
        # Recent performance (last 20 trades)
        recent_metrics = self._metrics[-20:]
        if recent_metrics:
            recent_success = sum(1 for m in recent_metrics if m.success)
            recent_within_block = sum(1 for m in recent_metrics if m.within_block_time)
            
            summary["recent"] = {
                "trades": len(recent_metrics),
                "success_rate_pct": round((recent_success / len(recent_metrics)) * 100, 2),
                "within_block_rate_pct": round((recent_within_block / len(recent_metrics)) * 100, 2),
            }
        
        return summary
    
    def get_strategy_performance(self, strategy: str) -> Dict:
        """Get detailed performance metrics for specific strategy."""
        strategy_metrics = [m for m in self._metrics if m.strategy == strategy]
        
        if not strategy_metrics:
            return {
                "strategy": strategy,
                "count": 0,
                "message": "No trades recorded for this strategy"
            }
        
        # Calculate detailed stats
        total = len(strategy_metrics)
        successful = sum(1 for m in strategy_metrics if m.success)
        within_block = sum(1 for m in strategy_metrics if m.within_block_time)
        
        exec_times = [m.execution_time_ms for m in strategy_metrics]
        avg_time = sum(exec_times) / len(exec_times)
        min_time = min(exec_times)
        max_time = max(exec_times)
        
        # Slippage analysis
        slippages = [m.slippage_pct for m in strategy_metrics if m.slippage_pct]
        if slippages:
            avg_slippage = sum(slippages) / len(slippages)
            max_slippage = max(slippages)
        else:
            avg_slippage = None
            max_slippage = None
        
        # Fill rate analysis
        fill_rates = [m.fill_rate for m in strategy_metrics]
        avg_fill_rate = sum(fill_rates) / len(fill_rates)
        
        return {
            "strategy": strategy,
            "count": total,
            "successful": successful,
            "success_rate_pct": round((successful / total) * 100, 2),
            "within_block": within_block,
            "within_block_rate_pct": round((within_block / total) * 100, 2),
            "execution_time": {
                "avg_ms": round(avg_time, 2),
                "min_ms": round(min_time, 2),
                "max_ms": round(max_time, 2),
            },
            "fill_rate": {
                "avg_pct": round(avg_fill_rate * 100, 2),
            },
            "slippage": {
                "avg_pct": round(float(avg_slippage) * 100, 2) if avg_slippage else None,
                "max_pct": round(float(max_slippage) * 100, 2) if max_slippage else None,
            } if slippages else None
        }
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trade metrics."""
        recent = self._metrics[-limit:]
        
        return [
            {
                "trade_id": m.trade_id,
                "timestamp": m.timestamp.isoformat(),
                "strategy": m.strategy,
                "success": m.success,
                "legs": f"{m.legs_filled}/{m.num_legs}",
                "time_ms": round(m.execution_time_ms, 1),
                "profit": float(m.actual_profit) if m.actual_profit else None,
                "slippage_pct": round(float(m.slippage_pct) * 100, 2) if m.slippage_pct else None,
            }
            for m in recent
        ]
    
    def export_metrics(self) -> List[Dict]:
        """Export all metrics for analysis."""
        return [
            {
                "trade_id": m.trade_id,
                "timestamp": m.timestamp.isoformat(),
                "strategy": m.strategy,
                "num_legs": m.num_legs,
                "legs_filled": m.legs_filled,
                "fill_rate": m.fill_rate,
                "execution_time_ms": m.execution_time_ms,
                "within_block": m.within_block_time,
                "committed": m.committed,
                "rolled_back": m.rolled_back,
                "success": m.success,
                "expected_profit": float(m.expected_profit) if m.expected_profit else None,
                "actual_profit": float(m.actual_profit) if m.actual_profit else None,
                "slippage_pct": float(m.slippage_pct) if m.slippage_pct else None,
                "leg_times_ms": m.leg_times_ms,
                "leg_statuses": m.leg_statuses,
                "errors": m.errors,
            }
            for m in self._metrics
        ]
    
    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        self._metrics.clear()
        self._total_trades = 0
        self._successful_trades = 0
        self._failed_trades = 0
        self._rolled_back_trades = 0
        self._strategy_stats.clear()
        self._total_expected_profit = Decimal("0")
        self._total_actual_profit = Decimal("0")
        logger.info("Metrics collector reset")


class PolygonGasTracker:
    """
    Track gas costs for Polygon transactions.
    
    Monitors:
    - Gas price trends
    - Transaction costs
    - Optimization opportunities
    """
    
    def __init__(self):
        self._gas_prices: List[Tuple[datetime, Decimal]] = []  # (timestamp, gwei)
        self._tx_costs: List[Tuple[str, Decimal]] = []  # (trade_id, cost_matic)
    
    def record_gas_price(self, price_gwei: Decimal) -> None:
        """Record current gas price."""
        self._gas_prices.append((datetime.now(timezone.utc), price_gwei))
        
        # Keep last 1000 samples
        if len(self._gas_prices) > 1000:
            self._gas_prices = self._gas_prices[-1000:]
    
    def record_tx_cost(self, trade_id: str, cost_matic: Decimal) -> None:
        """Record transaction cost for a trade."""
        self._tx_costs.append((trade_id, cost_matic))
    
    def get_gas_stats(self) -> Dict:
        """Get gas price statistics."""
        if not self._gas_prices:
            return {"message": "No gas data recorded"}
        
        recent = [price for _, price in self._gas_prices[-100:]]
        
        return {
            "current_gwei": float(recent[-1]),
            "avg_gwei": float(sum(recent) / len(recent)),
            "min_gwei": float(min(recent)),
            "max_gwei": float(max(recent)),
            "samples": len(self._gas_prices),
        }
    
    def get_cost_stats(self) -> Dict:
        """Get transaction cost statistics."""
        if not self._tx_costs:
            return {"message": "No cost data recorded"}
        
        costs = [cost for _, cost in self._tx_costs]
        
        return {
            "total_trades": len(costs),
            "total_cost_matic": float(sum(costs)),
            "avg_cost_matic": float(sum(costs) / len(costs)),
            "min_cost_matic": float(min(costs)),
            "max_cost_matic": float(max(costs)),
        }
