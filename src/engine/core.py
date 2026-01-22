"""
PR3DICT: Unified Trading Engine

Core engine managing the trading lifecycle.
Follows ST0CK's Strategy Pattern - decouples execution from signal logic.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
from decimal import Decimal
from dataclasses import dataclass, field

from ..platforms.base import PlatformInterface, Market, Position, OrderSide, OrderType
from ..strategies.base import TradingStrategy, Signal
from ..risk.manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the trading engine."""
    scan_interval_seconds: int = 30
    max_positions: int = 10
    paper_mode: bool = True
    platforms: List[str] = field(default_factory=lambda: ["kalshi"])


@dataclass
class EngineState:
    """Runtime state of the engine."""
    running: bool = False
    cycle_count: int = 0
    last_scan: Optional[datetime] = None
    active_signals: List[Signal] = field(default_factory=list)
    daily_pnl: Decimal = Decimal("0")
    trades_today: int = 0


class TradingEngine:
    """
    Unified prediction market trading engine.
    
    Responsibilities:
    - Lifecycle management (start/stop)
    - Platform coordination (multi-platform)
    - Strategy execution (signal → order)
    - Position tracking
    - Risk gate checks
    """
    
    def __init__(self,
                 platforms: List[PlatformInterface],
                 strategies: List[TradingStrategy],
                 risk_manager: RiskManager,
                 config: Optional[EngineConfig] = None):
        self.platforms = {p.name: p for p in platforms}
        self.strategies = {s.name: s for s in strategies}
        self.risk = risk_manager
        self.config = config or EngineConfig()
        self.state = EngineState()
        
        self._task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the trading engine."""
        logger.info("=" * 50)
        logger.info("PR3DICT Trading Engine Starting")
        logger.info(f"Mode: {'PAPER' if self.config.paper_mode else 'LIVE'}")
        logger.info(f"Platforms: {list(self.platforms.keys())}")
        logger.info(f"Strategies: {list(self.strategies.keys())}")
        logger.info("=" * 50)
        
        # Connect to all platforms
        for name, platform in self.platforms.items():
            connected = await platform.connect()
            if not connected:
                logger.error(f"Failed to connect to {name}")
                return
            logger.info(f"Connected to {name}")
        
        self.state.running = True
        self._task = asyncio.create_task(self._main_loop())
    
    async def stop(self) -> None:
        """Gracefully stop the engine."""
        logger.info("Stopping trading engine...")
        self.state.running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # Disconnect platforms
        for platform in self.platforms.values():
            await platform.disconnect()
        
        logger.info("Engine stopped.")
    
    async def _main_loop(self) -> None:
        """Main trading loop."""
        while self.state.running:
            try:
                await self._run_trading_cycle()
                await asyncio.sleep(self.config.scan_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _run_trading_cycle(self) -> None:
        """Execute one trading cycle."""
        self.state.cycle_count += 1
        self.state.last_scan = datetime.now(timezone.utc)
        
        logger.debug(f"=== Cycle {self.state.cycle_count} ===")
        
        # 1. Check if trading allowed
        allowed, reason = self.risk.check_trade_allowed()
        if not allowed:
            logger.warning(f"Trading blocked: {reason}")
            return
        
        # 2. Fetch markets from all platforms
        all_markets = await self._fetch_all_markets()
        logger.debug(f"Fetched {len(all_markets)} markets")
        
        # 3. Get current positions
        all_positions = await self._fetch_all_positions()
        
        # 4. Check exits on existing positions
        await self._check_exits(all_positions, all_markets)
        
        # 5. Scan for new entry signals
        if len(all_positions) < self.config.max_positions:
            await self._scan_entries(all_markets)
    
    async def _fetch_all_markets(self) -> List[Market]:
        """Fetch markets from all connected platforms."""
        markets = []
        for platform in self.platforms.values():
            try:
                platform_markets = await platform.get_markets(status="open", limit=100)
                markets.extend(platform_markets)
            except Exception as e:
                logger.error(f"Failed to fetch markets from {platform.name}: {e}")
        return markets
    
    async def _fetch_all_positions(self) -> List[Position]:
        """Get positions from all platforms."""
        positions = []
        for platform in self.platforms.values():
            try:
                platform_positions = await platform.get_positions()
                positions.extend(platform_positions)
            except Exception as e:
                logger.error(f"Failed to fetch positions from {platform.name}: {e}")
        return positions
    
    async def _check_exits(self, positions: List[Position], markets: List[Market]) -> None:
        """Check all positions for exit signals."""
        market_lookup = {m.id: m for m in markets}
        
        for position in positions:
            market = market_lookup.get(position.market_id)
            if not market:
                continue
            
            for strategy in self.strategies.values():
                exit_signal = await strategy.check_exit(position, market)
                if exit_signal:
                    await self._execute_exit(position, exit_signal)
                    break
    
    async def _scan_entries(self, markets: List[Market]) -> None:
        """Scan markets for entry opportunities."""
        for strategy in self.strategies.values():
            if not strategy.enabled:
                continue
            
            signals = await strategy.scan_markets(markets)
            
            for signal in signals:
                # Risk check per signal
                allowed, reason = self.risk.check_trade_allowed()
                if not allowed:
                    logger.debug(f"Signal rejected by risk: {reason}")
                    continue
                
                # Calculate position size
                balance = await self._get_total_balance()
                size = strategy.get_position_size(signal, balance)
                
                # Size validation
                if not self.risk.validate_position_size(size, signal.target_price or Decimal("0.5")):
                    logger.debug(f"Position size rejected: {size}")
                    continue
                
                await self._execute_entry(signal, size)
    
    async def _execute_entry(self, signal: Signal, size: int) -> None:
        """Execute an entry order."""
        platform = self.platforms.get(signal.market.platform)
        if not platform:
            logger.error(f"Platform {signal.market.platform} not connected")
            return
        
        logger.info(f"ENTRY: {signal.market.ticker} {signal.side.value.upper()} "
                   f"x{size} @ {signal.target_price or 'MKT'} - {signal.reason}")
        
        if self.config.paper_mode:
            logger.info("[PAPER] Order simulated")
            return
        
        try:
            order = await platform.place_order(
                market_id=signal.market_id,
                side=signal.side,
                order_type=OrderType.LIMIT if signal.target_price else OrderType.MARKET,
                quantity=size,
                price=signal.target_price
            )
            logger.info(f"Order placed: {order.id}")
            self.state.trades_today += 1
        except Exception as e:
            logger.error(f"Order failed: {e}")
    
    async def _execute_exit(self, position: Position, signal: Signal) -> None:
        """Execute an exit order."""
        platform = self.platforms.get(position.platform)
        if not platform:
            return
        
        exit_side = OrderSide.NO if position.side == OrderSide.YES else OrderSide.YES
        
        logger.info(f"EXIT: {position.ticker} {exit_side.value.upper()} "
                   f"x{position.quantity} - {signal.reason}")
        
        if self.config.paper_mode:
            logger.info("[PAPER] Exit simulated")
            return
        
        try:
            await platform.place_order(
                market_id=position.market_id,
                side=exit_side,
                order_type=OrderType.MARKET,
                quantity=position.quantity
            )
            self.state.daily_pnl += position.unrealized_pnl
        except Exception as e:
            logger.error(f"Exit failed: {e}")
    
    async def _get_total_balance(self) -> Decimal:
        """Get combined balance across all platforms."""
        total = Decimal("0")
        for platform in self.platforms.values():
            try:
                total += await platform.get_balance()
            except:
                pass
        return total
