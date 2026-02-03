"""
Backtesting Engine

Simulates the TradingEngine against historical data to validate strategies
before live deployment. Replicates order execution, position tracking, and
P&L without making real API calls.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
from decimal import Decimal

from ..platforms.base import Market, OrderSide, Position, Order, OrderStatus, OrderType
from ..strategies.base import TradingStrategy, Signal
from ..risk.manager import RiskManager
from .data import HistoricalDataLoader

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: datetime
    end_date: datetime
    initial_balance: Decimal = Decimal("10000")
    max_positions: int = 10
    commission_rate: Decimal = Decimal("0.01")  # 1% per trade
    slippage_bps: Decimal = Decimal("5")  # 5 basis points slippage
    platforms: List[str] = field(default_factory=lambda: ["kalshi"])


@dataclass
class BacktestTrade:
    """Record of a simulated trade."""
    timestamp: datetime
    market_id: str
    ticker: str
    side: OrderSide
    quantity: int
    price: Decimal
    commission: Decimal
    trade_type: str  # 'entry' or 'exit'
    strategy: str
    reason: str


@dataclass
class BacktestPosition:
    """Simulated position during backtest."""
    market_id: str
    ticker: str
    side: OrderSide
    quantity: int
    entry_price: Decimal
    entry_time: datetime
    strategy: str
    platform: str


class BacktestEngine:
    """
    Backtesting engine for PR3DICT strategies.
    
    Simulates the live TradingEngine but uses historical data replay
    instead of real-time market feeds.
    
    Key differences from live engine:
    - No async (historical data is synchronous)
    - Simulated order fills with slippage
    - Commission modeling
    - Perfect record of all trades for analysis
    """
    
    def __init__(self,
                 data_loader: HistoricalDataLoader,
                 strategies: List[TradingStrategy],
                 config: BacktestConfig,
                 risk_manager: Optional[RiskManager] = None):
        self.data_loader = data_loader
        self.strategies = {s.name: s for s in strategies}
        self.config = config
        self.risk = risk_manager
        
        # State
        self.balance = config.initial_balance
        self.positions: List[BacktestPosition] = []
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[tuple[datetime, Decimal]] = []
        
        # Performance tracking
        self.wins = 0
        self.losses = 0
        self.total_commission = Decimal("0")
        self.peak_equity = config.initial_balance
        self.max_drawdown = Decimal("0")
    
    def run(self) -> Dict:
        """
        Run the backtest over the configured date range.
        
        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info("=" * 60)
        logger.info("PR3DICT BACKTESTING ENGINE")
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Initial Balance: ${self.config.initial_balance}")
        logger.info(f"Strategies: {list(self.strategies.keys())}")
        logger.info(f"Max Positions: {self.config.max_positions}")
        logger.info(f"Commission: {self.config.commission_rate * 100}%")
        logger.info(f"Slippage: {self.config.slippage_bps} bps")
        logger.info("=" * 60)
        
        # Replay historical data
        for timestamp, markets in self.data_loader.replay(
            self.config.start_date,
            self.config.end_date
        ):
            self._process_timestamp(timestamp, markets)
        
        # Close any remaining positions at final prices
        self._close_all_positions(timestamp)
        
        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info(f"Final Balance: ${self.balance:.2f}")
        logger.info(f"Total Return: {self._calculate_return():.2%}")
        logger.info(f"Total Trades: {len(self.trades)}")
        logger.info(f"Win Rate: {self._calculate_win_rate():.2%}")
        logger.info(f"Max Drawdown: {self.max_drawdown:.2%}")
        logger.info("=" * 60)
        
        return self._generate_results()
    
    def _process_timestamp(self, timestamp: datetime, markets: List[Market]) -> None:
        """Process a single point in time during backtest."""
        # Update equity curve
        equity = self._calculate_equity(markets)
        self.equity_curve.append((timestamp, equity))
        
        # Track drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        else:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        # Check exits on existing positions
        self._check_exits(timestamp, markets)
        
        # Scan for new entries if we have capacity
        if len(self.positions) < self.config.max_positions:
            self._scan_entries(timestamp, markets)
    
    def _check_exits(self, timestamp: datetime, markets: List[Market]) -> None:
        """Check all positions for exit signals."""
        market_lookup = {m.id: m for m in markets}
        positions_to_close = []
        
        for position in self.positions:
            market = market_lookup.get(position.market_id)
            if not market:
                continue
            
            # Convert to Position object for strategy
            pos_obj = self._to_position_object(position, market)
            
            # Check each strategy
            strategy = self.strategies.get(position.strategy)
            if not strategy:
                continue
            
            # Async to sync adapter (strategies are async in live mode)
            import asyncio
            try:
                exit_signal = asyncio.run(strategy.check_exit(pos_obj, market))
            except:
                # If async doesn't work, try calling directly
                exit_signal = None
            
            if exit_signal:
                positions_to_close.append((position, exit_signal, market))
        
        # Execute exits
        for position, signal, market in positions_to_close:
            self._execute_exit(timestamp, position, signal, market)
    
    def _scan_entries(self, timestamp: datetime, markets: List[Market]) -> None:
        """Scan for entry opportunities."""
        for strategy in self.strategies.values():
            # Async to sync adapter
            import asyncio
            try:
                signals = asyncio.run(strategy.scan_markets(markets))
            except:
                signals = []
            
            for signal in signals:
                # Risk check
                if self.risk:
                    allowed, reason = self.risk.check_trade_allowed()
                    if not allowed:
                        logger.debug(f"Signal rejected by risk: {reason}")
                        continue
                
                # Calculate position size
                size = strategy.get_position_size(signal, self.balance)
                
                # Size validation
                if self.risk and not self.risk.validate_position_size(
                    size, signal.target_price or Decimal("0.5")
                ):
                    continue
                
                # Execute entry
                self._execute_entry(timestamp, signal, size, strategy.name)
    
    def _execute_entry(self, timestamp: datetime, signal: Signal, 
                       size: int, strategy_name: str) -> None:
        """Simulate an entry order with slippage and commission."""
        # Determine fill price with slippage
        if signal.target_price:
            fill_price = signal.target_price
        else:
            # Market order - use current price plus slippage
            base_price = (signal.market.yes_price if signal.side == OrderSide.YES 
                         else signal.market.no_price)
            slippage = base_price * (self.config.slippage_bps / Decimal("10000"))
            fill_price = base_price + slippage
        
        # Calculate costs
        trade_value = fill_price * size
        commission = trade_value * self.config.commission_rate
        total_cost = trade_value + commission
        
        # Check if we can afford it
        if total_cost > self.balance:
            logger.debug(f"Insufficient balance for trade: {total_cost} > {self.balance}")
            return
        
        # Deduct from balance
        self.balance -= total_cost
        self.total_commission += commission
        
        # Create position
        position = BacktestPosition(
            market_id=signal.market_id,
            ticker=signal.market.ticker,
            side=signal.side,
            quantity=size,
            entry_price=fill_price,
            entry_time=timestamp,
            strategy=strategy_name,
            platform=signal.market.platform
        )
        self.positions.append(position)
        
        # Record trade
        trade = BacktestTrade(
            timestamp=timestamp,
            market_id=signal.market_id,
            ticker=signal.market.ticker,
            side=signal.side,
            quantity=size,
            price=fill_price,
            commission=commission,
            trade_type='entry',
            strategy=strategy_name,
            reason=signal.reason
        )
        self.trades.append(trade)
        
        logger.info(f"[{timestamp}] ENTRY: {signal.market.ticker} {signal.side.value.upper()} "
                   f"x{size} @ ${fill_price:.3f} | {signal.reason}")
    
    def _execute_exit(self, timestamp: datetime, position: BacktestPosition,
                      signal: Signal, market: Market) -> None:
        """Simulate an exit order."""
        # Determine exit price
        exit_price = (market.yes_price if position.side == OrderSide.NO 
                     else market.no_price)
        
        # Add slippage
        slippage = exit_price * (self.config.slippage_bps / Decimal("10000"))
        fill_price = exit_price - slippage  # Negative for exits
        
        # Calculate P&L
        trade_value = fill_price * position.quantity
        commission = trade_value * self.config.commission_rate
        gross_pnl = trade_value - (position.entry_price * position.quantity)
        net_pnl = gross_pnl - commission - self.total_commission
        
        # Update balance
        self.balance += trade_value - commission
        self.total_commission += commission
        
        # Track win/loss
        if net_pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        
        # Remove position
        self.positions.remove(position)
        
        # Record trade
        trade = BacktestTrade(
            timestamp=timestamp,
            market_id=position.market_id,
            ticker=position.ticker,
            side=OrderSide.NO if position.side == OrderSide.YES else OrderSide.YES,
            quantity=position.quantity,
            price=fill_price,
            commission=commission,
            trade_type='exit',
            strategy=position.strategy,
            reason=signal.reason
        )
        self.trades.append(trade)
        
        logger.info(f"[{timestamp}] EXIT: {position.ticker} "
                   f"@ ${fill_price:.3f} | P&L: ${net_pnl:+.2f} | {signal.reason}")
    
    def _close_all_positions(self, final_timestamp: datetime) -> None:
        """Force close all remaining positions at end of backtest."""
        if not self.positions:
            return
        
        logger.info(f"Closing {len(self.positions)} remaining positions...")
        
        for position in self.positions[:]:  # Copy list since we'll modify it
            # Get final market price
            market = self.data_loader.get_market_at_time(position.market_id, final_timestamp)
            if not market:
                logger.warning(f"No final price for {position.ticker}, skipping")
                continue
            
            # Create synthetic exit signal
            exit_signal = Signal(
                market_id=position.market_id,
                market=market,
                side=OrderSide.NO if position.side == OrderSide.YES else OrderSide.YES,
                strength=1.0,
                reason="Backtest end - forced close"
            )
            
            self._execute_exit(final_timestamp, position, exit_signal, market)
    
    def _calculate_equity(self, markets: List[Market]) -> Decimal:
        """Calculate current equity (balance + unrealized P&L)."""
        equity = self.balance
        
        market_lookup = {m.id: m for m in markets}
        
        for position in self.positions:
            market = market_lookup.get(position.market_id)
            if market:
                # Current value of position
                current_price = (market.yes_price if position.side == OrderSide.NO 
                               else market.no_price)
                position_value = current_price * position.quantity
                equity += position_value
        
        return equity
    
    def _to_position_object(self, bp: BacktestPosition, market: Market) -> Position:
        """Convert BacktestPosition to Position object for strategy."""
        current_price = market.yes_price if bp.side == OrderSide.YES else market.no_price
        unrealized_pnl = (current_price - bp.entry_price) * bp.quantity
        
        return Position(
            market_id=bp.market_id,
            ticker=bp.ticker,
            side=bp.side,
            quantity=bp.quantity,
            avg_price=bp.entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            platform=bp.platform
        )
    
    def _calculate_return(self) -> Decimal:
        """Calculate total return percentage."""
        if self.config.initial_balance == 0:
            return Decimal("0")
        return (self.balance - self.config.initial_balance) / self.config.initial_balance
    
    def _calculate_win_rate(self) -> Decimal:
        """Calculate win rate (wins / total closed trades)."""
        total_trades = self.wins + self.losses
        if total_trades == 0:
            return Decimal("0")
        return Decimal(self.wins) / Decimal(total_trades)
    
    def _generate_results(self) -> Dict:
        """Generate comprehensive results dictionary."""
        return {
            'config': self.config,
            'final_balance': self.balance,
            'initial_balance': self.config.initial_balance,
            'total_return': self._calculate_return(),
            'total_trades': len(self.trades),
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self._calculate_win_rate(),
            'total_commission': self.total_commission,
            'max_drawdown': self.max_drawdown,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'strategies': list(self.strategies.keys())
        }
