"""
Performance Metrics for Backtesting

Calculates industry-standard trading metrics including Sharpe ratio,
maximum drawdown, win rate, and risk-adjusted returns.
"""
import logging
from dataclasses import dataclass
from typing import List, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a backtest."""
    
    # Returns
    total_return: Decimal
    annualized_return: Decimal
    
    # Risk metrics
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    max_drawdown: Decimal
    max_drawdown_duration: timedelta
    
    # Trade statistics
    total_trades: int
    wins: int
    losses: int
    win_rate: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: Decimal  # Gross profit / gross loss
    
    # Position metrics
    avg_trade_duration: timedelta
    max_concurrent_positions: int
    
    # Costs
    total_commission: Decimal
    commission_pct_of_returns: Decimal
    
    # Period
    start_date: datetime
    end_date: datetime
    total_days: int
    
    def __str__(self) -> str:
        """Human-readable summary."""
        return f"""
Performance Metrics
{'=' * 50}
Period: {self.start_date.date()} to {self.end_date.date()} ({self.total_days} days)

Returns:
  Total Return:        {self.total_return:>8.2%}
  Annualized Return:   {self.annualized_return:>8.2%}

Risk-Adjusted:
  Sharpe Ratio:        {self.sharpe_ratio:>8.2f}
  Sortino Ratio:       {self.sortino_ratio:>8.2f}
  Max Drawdown:        {self.max_drawdown:>8.2%}
  Max DD Duration:     {self.max_drawdown_duration.days} days

Trading:
  Total Trades:        {self.total_trades:>8}
  Win Rate:            {self.win_rate:>8.2%}
  Wins / Losses:       {self.wins} / {self.losses}
  Avg Win:             ${self.avg_win:>7.2f}
  Avg Loss:            ${self.avg_loss:>7.2f}
  Profit Factor:       {self.profit_factor:>8.2f}
  Avg Trade Duration:  {self.avg_trade_duration.days} days

Costs:
  Total Commission:    ${self.total_commission:>7.2f}
  Commission % Returns: {self.commission_pct_of_returns:>7.2%}
{'=' * 50}
"""


def calculate_metrics(results: dict) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from backtest results.
    
    Args:
        results: Dictionary returned by BacktestEngine.run()
        
    Returns:
        PerformanceMetrics object with all calculated statistics
    """
    config = results['config']
    equity_curve = results['equity_curve']
    trades = results['trades']
    
    # Basic returns
    initial_balance = config.initial_balance
    final_balance = results['final_balance']
    total_return = (final_balance - initial_balance) / initial_balance
    
    # Time period
    start_date = config.start_date
    end_date = config.end_date
    total_days = (end_date - start_date).days
    years = total_days / 365.25
    
    # Annualized return
    if years > 0:
        annualized_return = Decimal(str((1 + float(total_return)) ** (1 / years) - 1))
    else:
        annualized_return = total_return
    
    # Calculate returns series for risk metrics
    returns = _calculate_returns_series(equity_curve)
    
    # Sharpe ratio (assuming 2% risk-free rate)
    sharpe = _calculate_sharpe_ratio(returns, risk_free_rate=Decimal("0.02"))
    
    # Sortino ratio (downside deviation only)
    sortino = _calculate_sortino_ratio(returns, risk_free_rate=Decimal("0.02"))
    
    # Maximum drawdown
    max_dd, max_dd_duration = _calculate_max_drawdown(equity_curve)
    
    # Trade statistics
    entry_trades = [t for t in trades if t.trade_type == 'entry']
    exit_trades = [t for t in trades if t.trade_type == 'exit']
    
    total_trades = len(exit_trades)  # Count completed round trips
    wins = results['wins']
    losses = results['losses']
    win_rate = Decimal(wins) / Decimal(total_trades) if total_trades > 0 else Decimal("0")
    
    # Profit/loss analysis
    winning_trades = []
    losing_trades = []
    
    for exit_trade in exit_trades:
        # Find matching entry
        entry = _find_entry_trade(entry_trades, exit_trade)
        if not entry:
            continue
        
        pnl = _calculate_trade_pnl(entry, exit_trade)
        
        if pnl > 0:
            winning_trades.append(pnl)
        else:
            losing_trades.append(pnl)
    
    avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else Decimal("0")
    avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else Decimal("0")
    
    # Profit factor
    gross_profit = sum(winning_trades) if winning_trades else Decimal("0")
    gross_loss = abs(sum(losing_trades)) if losing_trades else Decimal("1")  # Avoid div by zero
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("0")
    
    # Position duration
    avg_duration = _calculate_avg_trade_duration(entry_trades, exit_trades)
    
    # Max concurrent positions
    max_concurrent = _calculate_max_concurrent_positions(entry_trades, exit_trades)
    
    # Commission analysis
    total_commission = results['total_commission']
    commission_pct = (total_commission / initial_balance) if initial_balance > 0 else Decimal("0")
    
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        avg_trade_duration=avg_duration,
        max_concurrent_positions=max_concurrent,
        total_commission=total_commission,
        commission_pct_of_returns=commission_pct,
        start_date=start_date,
        end_date=end_date,
        total_days=total_days
    )


def _calculate_returns_series(equity_curve: List[Tuple[datetime, Decimal]]) -> List[Decimal]:
    """Calculate period-over-period returns from equity curve."""
    if len(equity_curve) < 2:
        return []
    
    returns = []
    for i in range(1, len(equity_curve)):
        prev_equity = equity_curve[i-1][1]
        curr_equity = equity_curve[i][1]
        
        if prev_equity > 0:
            period_return = (curr_equity - prev_equity) / prev_equity
            returns.append(period_return)
    
    return returns


def _calculate_sharpe_ratio(returns: List[Decimal], risk_free_rate: Decimal = Decimal("0.02")) -> Decimal:
    """
    Calculate annualized Sharpe ratio.
    
    Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
    """
    if not returns:
        return Decimal("0")
    
    # Convert to float for math operations
    returns_float = [float(r) for r in returns]
    
    mean_return = sum(returns_float) / len(returns_float)
    
    # Standard deviation
    variance = sum((r - mean_return) ** 2 for r in returns_float) / len(returns_float)
    std_dev = math.sqrt(variance)
    
    if std_dev == 0:
        return Decimal("0")
    
    # Annualize (assuming daily returns, 252 trading days)
    annualized_return = mean_return * 252
    annualized_std = std_dev * math.sqrt(252)
    
    sharpe = (annualized_return - float(risk_free_rate)) / annualized_std
    
    return Decimal(str(sharpe))


def _calculate_sortino_ratio(returns: List[Decimal], risk_free_rate: Decimal = Decimal("0.02")) -> Decimal:
    """
    Calculate Sortino ratio (uses downside deviation instead of total volatility).
    
    Better metric than Sharpe for asymmetric return distributions.
    """
    if not returns:
        return Decimal("0")
    
    returns_float = [float(r) for r in returns]
    mean_return = sum(returns_float) / len(returns_float)
    
    # Downside deviation (only negative returns)
    downside_returns = [r for r in returns_float if r < 0]
    
    if not downside_returns:
        return Decimal("999")  # No downside = infinite Sortino
    
    downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
    downside_std = math.sqrt(downside_variance)
    
    if downside_std == 0:
        return Decimal("0")
    
    # Annualize
    annualized_return = mean_return * 252
    annualized_downside_std = downside_std * math.sqrt(252)
    
    sortino = (annualized_return - float(risk_free_rate)) / annualized_downside_std
    
    return Decimal(str(sortino))


def _calculate_max_drawdown(equity_curve: List[Tuple[datetime, Decimal]]) -> Tuple[Decimal, timedelta]:
    """
    Calculate maximum drawdown and its duration.
    
    Returns:
        (max_drawdown_pct, duration)
    """
    if not equity_curve:
        return Decimal("0"), timedelta(0)
    
    max_dd = Decimal("0")
    max_dd_duration = timedelta(0)
    
    peak = equity_curve[0][1]
    peak_date = equity_curve[0][0]
    
    for timestamp, equity in equity_curve:
        if equity > peak:
            peak = equity
            peak_date = timestamp
        else:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
                max_dd_duration = timestamp - peak_date
    
    return max_dd, max_dd_duration


def _find_entry_trade(entry_trades, exit_trade):
    """Find the entry trade matching an exit."""
    for entry in entry_trades:
        if (entry.market_id == exit_trade.market_id and 
            entry.timestamp < exit_trade.timestamp):
            return entry
    return None


def _calculate_trade_pnl(entry_trade, exit_trade) -> Decimal:
    """Calculate P&L for a round-trip trade."""
    entry_cost = entry_trade.price * entry_trade.quantity + entry_trade.commission
    exit_value = exit_trade.price * exit_trade.quantity - exit_trade.commission
    return exit_value - entry_cost


def _calculate_avg_trade_duration(entry_trades, exit_trades) -> timedelta:
    """Calculate average holding period."""
    durations = []
    
    for exit_trade in exit_trades:
        entry = _find_entry_trade(entry_trades, exit_trade)
        if entry:
            duration = exit_trade.timestamp - entry.timestamp
            durations.append(duration)
    
    if not durations:
        return timedelta(0)
    
    total_seconds = sum(d.total_seconds() for d in durations)
    avg_seconds = total_seconds / len(durations)
    
    return timedelta(seconds=avg_seconds)


def _calculate_max_concurrent_positions(entry_trades, exit_trades) -> int:
    """Calculate maximum number of positions held at once."""
    events = []
    
    for entry in entry_trades:
        events.append((entry.timestamp, 1, entry.market_id))  # +1 position
    
    for exit in exit_trades:
        events.append((exit.timestamp, -1, exit.market_id))  # -1 position
    
    events.sort(key=lambda x: x[0])
    
    current_positions = 0
    max_positions = 0
    
    for timestamp, delta, market_id in events:
        current_positions += delta
        max_positions = max(max_positions, current_positions)
    
    return max_positions
