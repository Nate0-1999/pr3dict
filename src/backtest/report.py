"""
Backtest Report Generation

Generates comprehensive reports from backtesting results including:
- Performance summary
- Trade log
- Equity curve visualization (text-based)
- Strategy breakdown
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import json

from .metrics import PerformanceMetrics, calculate_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestReport:
    """Container for backtest report data."""
    metrics: PerformanceMetrics
    results: Dict
    timestamp: datetime
    
    def to_text(self) -> str:
        """Generate text report."""
        lines = []
        lines.append("=" * 70)
        lines.append("PR3DICT BACKTEST REPORT")
        lines.append(f"Generated: {self.timestamp}")
        lines.append("=" * 70)
        lines.append("")
        
        # Performance metrics
        lines.append(str(self.metrics))
        lines.append("")
        
        # Strategy breakdown
        lines.append("Strategy Performance:")
        lines.append("-" * 70)
        strategy_stats = self._calculate_strategy_stats()
        for strategy, stats in strategy_stats.items():
            lines.append(f"\n{strategy}:")
            lines.append(f"  Trades: {stats['trades']}")
            lines.append(f"  Win Rate: {stats['win_rate']:.2%}")
            lines.append(f"  Avg P&L: ${stats['avg_pnl']:.2f}")
        lines.append("")
        
        # Recent trades
        lines.append("Recent Trades (last 10):")
        lines.append("-" * 70)
        trades = self.results['trades'][-10:]
        for trade in trades:
            lines.append(
                f"{trade.timestamp} | {trade.ticker:12} | "
                f"{trade.side.value.upper():4} | {trade.trade_type:5} | "
                f"${trade.price:.3f} x{trade.quantity:3} | {trade.reason}"
            )
        lines.append("")
        
        # Equity curve (text-based sparkline)
        lines.append("Equity Curve:")
        lines.append("-" * 70)
        lines.append(self._generate_equity_sparkline())
        lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Generate JSON report."""
        data = {
            'timestamp': self.timestamp.isoformat(),
            'metrics': {
                'total_return': str(self.metrics.total_return),
                'annualized_return': str(self.metrics.annualized_return),
                'sharpe_ratio': str(self.metrics.sharpe_ratio),
                'sortino_ratio': str(self.metrics.sortino_ratio),
                'max_drawdown': str(self.metrics.max_drawdown),
                'win_rate': str(self.metrics.win_rate),
                'total_trades': self.metrics.total_trades,
                'profit_factor': str(self.metrics.profit_factor),
            },
            'trades': [
                {
                    'timestamp': t.timestamp.isoformat(),
                    'market_id': t.market_id,
                    'ticker': t.ticker,
                    'side': t.side.value,
                    'quantity': t.quantity,
                    'price': str(t.price),
                    'commission': str(t.commission),
                    'trade_type': t.trade_type,
                    'strategy': t.strategy,
                    'reason': t.reason
                }
                for t in self.results['trades']
            ],
            'equity_curve': [
                {
                    'timestamp': ts.isoformat(),
                    'equity': str(eq)
                }
                for ts, eq in self.results['equity_curve']
            ]
        }
        
        return json.dumps(data, indent=2)
    
    def save(self, output_dir: Path, format: str = 'text') -> Path:
        """Save report to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"backtest_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.{format}"
        filepath = output_dir / filename
        
        if format == 'json':
            content = self.to_json()
        else:
            content = self.to_text()
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        logger.info(f"Report saved to {filepath}")
        return filepath
    
    def _calculate_strategy_stats(self) -> Dict:
        """Calculate per-strategy statistics."""
        trades = self.results['trades']
        
        strategy_trades = {}
        for trade in trades:
            if trade.strategy not in strategy_trades:
                strategy_trades[trade.strategy] = []
            strategy_trades[trade.strategy].append(trade)
        
        stats = {}
        for strategy, strat_trades in strategy_trades.items():
            exits = [t for t in strat_trades if t.trade_type == 'exit']
            entries = [t for t in strat_trades if t.trade_type == 'entry']
            
            if not exits:
                continue
            
            # Calculate P&L for each trade
            pnls = []
            wins = 0
            
            for exit_trade in exits:
                # Find matching entry
                entry = None
                for e in entries:
                    if (e.market_id == exit_trade.market_id and 
                        e.timestamp < exit_trade.timestamp):
                        entry = e
                        break
                
                if entry:
                    pnl = self._calculate_pnl(entry, exit_trade)
                    pnls.append(pnl)
                    if pnl > 0:
                        wins += 1
            
            stats[strategy] = {
                'trades': len(exits),
                'win_rate': Decimal(wins) / Decimal(len(exits)) if exits else Decimal("0"),
                'avg_pnl': sum(pnls) / len(pnls) if pnls else Decimal("0")
            }
        
        return stats
    
    def _calculate_pnl(self, entry, exit) -> Decimal:
        """Calculate P&L for a trade pair."""
        entry_cost = entry.price * entry.quantity + entry.commission
        exit_value = exit.price * exit.quantity - exit.commission
        return exit_value - entry_cost
    
    def _generate_equity_sparkline(self) -> str:
        """Generate ASCII sparkline of equity curve."""
        equity_curve = self.results['equity_curve']
        
        if not equity_curve:
            return "No data"
        
        # Sample points if too many
        max_points = 60
        if len(equity_curve) > max_points:
            step = len(equity_curve) // max_points
            sampled = equity_curve[::step]
        else:
            sampled = equity_curve
        
        values = [float(eq) for _, eq in sampled]
        
        if not values:
            return "No data"
        
        # Normalize to 0-10 range
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            normalized = [5] * len(values)
        else:
            normalized = [
                int(((v - min_val) / (max_val - min_val)) * 10)
                for v in values
            ]
        
        # Create sparkline
        chars = " ▁▂▃▄▅▆▇█"
        sparkline = "".join(chars[min(n, 8)] for n in normalized)
        
        # Add labels
        lines = []
        lines.append(f"${max_val:,.0f} ┤")
        lines.append(f"        │ {sparkline}")
        lines.append(f"${min_val:,.0f} ┤")
        lines.append(f"        └{'─' * len(sparkline)}")
        lines.append(f"        {sampled[0][0].strftime('%m/%d')}" + 
                    " " * (len(sparkline) - 10) + 
                    f"{sampled[-1][0].strftime('%m/%d')}")
        
        return "\n".join(lines)


def generate_report(results: Dict, output_dir: Optional[Path] = None) -> BacktestReport:
    """
    Generate a comprehensive backtest report.
    
    Args:
        results: Dictionary from BacktestEngine.run()
        output_dir: Optional directory to save report
        
    Returns:
        BacktestReport object
    """
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Create report
    report = BacktestReport(
        metrics=metrics,
        results=results,
        timestamp=datetime.now()
    )
    
    # Print to console
    print(report.to_text())
    
    # Save to file if directory provided
    if output_dir:
        report.save(output_dir, format='text')
        report.save(output_dir, format='json')
    
    return report
