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
from ..execution import ArbV1State, ArbV1StateMachine
from ..risk.manager import RiskManager
from ..notifications.manager import NotificationManager

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
                 config: Optional[EngineConfig] = None,
                 notifications: Optional[NotificationManager] = None):
        self.platforms = {p.name: p for p in platforms}
        self.strategies = {s.name: s for s in strategies}
        self.risk = risk_manager
        self.notifications = notifications
        self.config = config or EngineConfig()
        self.state = EngineState()
        self._arb_v1_state_machine = ArbV1StateMachine()
        
        self._task: Optional[asyncio.Task] = None
        self._start_time: Optional[datetime] = None
    
    async def start(self) -> None:
        """Start the trading engine."""
        logger.info("=" * 50)
        logger.info("PR3DICT Trading Engine Starting")
        logger.info(f"Mode: {'PAPER' if self.config.paper_mode else 'LIVE'}")
        logger.info(f"Platforms: {list(self.platforms.keys())}")
        logger.info(f"Strategies: {list(self.strategies.keys())}")
        logger.info("=" * 50)
        
        # Connect notifications
        if self.notifications:
            await self.notifications.connect()
        
        # Connect to all platforms
        for name, platform in self.platforms.items():
            connected = await platform.connect()
            if not connected:
                logger.error(f"Failed to connect to {name}")
                if self.notifications:
                    await self.notifications.send_error(
                        f"Failed to connect to {name}",
                        context="Engine startup"
                    )
                return
            logger.info(f"Connected to {name}")
        
        self.state.running = True
        self._start_time = datetime.now(timezone.utc)
        self._task = asyncio.create_task(self._main_loop())
        
        # Send startup notification
        if self.notifications:
            mode = "PAPER" if self.config.paper_mode else "LIVE"
            await self.notifications.send_engine_status(
                status=f"STARTED ({mode})",
                cycle_count=0
            )
    
    async def stop(self) -> None:
        """Gracefully stop the engine."""
        logger.info("Stopping trading engine...")
        self.state.running = False
        
        # Calculate uptime
        uptime = None
        if self._start_time:
            elapsed = datetime.now(timezone.utc) - self._start_time
            hours = int(elapsed.total_seconds() // 3600)
            minutes = int((elapsed.total_seconds() % 3600) // 60)
            uptime = f"{hours}h {minutes}m"
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # Disconnect platforms
        for platform in self.platforms.values():
            await platform.disconnect()
        
        # Send shutdown notification
        if self.notifications:
            await self.notifications.send_engine_status(
                status="STOPPED",
                uptime=uptime,
                cycle_count=self.state.cycle_count
            )
            await self.notifications.disconnect()
        
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
                if self.notifications:
                    import traceback
                    await self.notifications.send_error(
                        error_msg=str(e),
                        context="Trading cycle",
                        traceback=traceback.format_exc()
                    )
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
            # Send risk alert on first block
            if self.notifications and reason != "OK":
                await self.notifications.send_risk_alert(
                    alert_type=reason,
                    details=f"Trading blocked: {reason}",
                    severity="WARNING"
                )
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
                
                # Send signal notification
                if self.notifications:
                    await self.notifications.send_signal(
                        ticker=signal.market.ticker,
                        side=signal.side.value.upper(),
                        price=float(signal.target_price or Decimal("0.5")),
                        size=size,
                        reason=signal.reason,
                        confidence=getattr(signal, 'confidence', None),
                        strategy=strategy.name
                    )
                
                await self._execute_entry(signal, size, strategy.name)
    
    async def _execute_entry(self, signal: Signal, size: int, strategy_name: str = None) -> None:
        """Execute an entry order."""
        platform = self.platforms.get(signal.market.platform)
        if not platform:
            logger.error(f"Platform {signal.market.platform} not connected")
            return

        paired_leg = signal.metadata.get("paired_leg")
        if strategy_name == "polymarket_arb_v1" and isinstance(paired_leg, dict):
            await self._execute_paired_entry(signal, size, platform)
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
            
            # Send order notification
            if self.notifications:
                await self.notifications.send_order_placed(
                    ticker=signal.market.ticker,
                    side=signal.side.value.upper(),
                    price=float(signal.target_price or order.price or Decimal("0.5")),
                    size=size,
                    order_id=order.id,
                    platform=platform.name
                )
                
        except Exception as e:
            logger.error(f"Order failed: {e}")
            if self.notifications:
                await self.notifications.send_error(
                    error_msg=f"Order placement failed: {e}",
                    context=f"Market: {signal.market.ticker}, Side: {signal.side.value}"
                )

    async def _execute_paired_entry(self, signal: Signal, size: int, platform: PlatformInterface) -> None:
        """
        Execute paired YES/NO legs for combinatorial arb.

        Conservative policy:
        - if both legs are not cleanly filled, treat as partial/failure
        - attempt immediate flatten for any filled exposure
        """
        state = ArbV1State.DISCOVERED
        opportunity_id = signal.metadata.get("opportunity_id", "unknown")
        paired_leg = signal.metadata.get("paired_leg", {})

        state = self._transition_state(state, ArbV1State.PRICED_EXECUTABLE, opportunity_id)
        state = self._transition_state(state, ArbV1State.RISK_APPROVED, opportunity_id)
        state = self._transition_state(state, ArbV1State.EXECUTION_SUBMITTED, opportunity_id)

        paired_side_raw = str(paired_leg.get("side", "")).lower()
        if paired_side_raw not in (OrderSide.YES.value, OrderSide.NO.value):
            logger.error(f"Invalid paired leg side for opportunity {opportunity_id}: {paired_side_raw}")
            state = self._transition_state(state, ArbV1State.FAILED, opportunity_id)
            self._transition_state(state, ArbV1State.CLOSED, opportunity_id)
            return

        paired_side = OrderSide.YES if paired_side_raw == OrderSide.YES.value else OrderSide.NO
        paired_price_raw = paired_leg.get("target_price")
        paired_price = None
        if paired_price_raw is not None:
            paired_price = Decimal(str(paired_price_raw))

        if self.config.paper_mode:
            logger.info(
                f"[PAPER] PAIRED ENTRY: {signal.market.ticker} "
                f"{signal.side.value.upper()}+{paired_side.value.upper()} x{size} "
                f"opp={opportunity_id}"
            )
            state = self._transition_state(state, ArbV1State.FILLED, opportunity_id)
            self._transition_state(state, ArbV1State.CLOSED, opportunity_id)
            return

        logger.info(
            f"PAIRED ENTRY: {signal.market.ticker} {signal.side.value.upper()}+{paired_side.value.upper()} "
            f"x{size} opp={opportunity_id}"
        )

        primary_order_type = OrderType.LIMIT if signal.target_price else OrderType.MARKET
        paired_order_type = OrderType.LIMIT if paired_price is not None else OrderType.MARKET

        results = await asyncio.gather(
            platform.place_order(
                market_id=signal.market_id,
                side=signal.side,
                order_type=primary_order_type,
                quantity=size,
                price=signal.target_price,
            ),
            platform.place_order(
                market_id=signal.market_id,
                side=paired_side,
                order_type=paired_order_type,
                quantity=size,
                price=paired_price,
            ),
            return_exceptions=True,
        )

        orders = []
        had_exception = False
        for result in results:
            if isinstance(result, Exception):
                had_exception = True
                logger.error(f"Paired leg placement error for {opportunity_id}: {result}")
            else:
                orders.append(result)

        if had_exception or len(orders) != 2:
            state = self._transition_state(state, ArbV1State.FAILED, opportunity_id)
            await self._flatten_residual_exposure(platform, signal.market_id, orders, size, opportunity_id)
            self._transition_state(state, ArbV1State.CLOSED, opportunity_id)
            return

        fully_filled = all(o.filled_quantity >= size for o in orders)
        if fully_filled:
            state = self._transition_state(state, ArbV1State.FILLED, opportunity_id)
            self._transition_state(state, ArbV1State.CLOSED, opportunity_id)
            self.state.trades_today += 1
            return

        state = self._transition_state(state, ArbV1State.PARTIAL_FILL, opportunity_id)
        await self._flatten_residual_exposure(platform, signal.market_id, orders, size, opportunity_id)
        state = self._transition_state(state, ArbV1State.HEDGED_OR_FLATTENED, opportunity_id)
        self._transition_state(state, ArbV1State.CLOSED, opportunity_id)

    def _transition_state(self, state: ArbV1State, target: ArbV1State, opportunity_id: str) -> ArbV1State:
        transition = self._arb_v1_state_machine.transition(state, target)
        if not transition.valid:
            logger.warning(f"Invalid arb v1 state transition for {opportunity_id}: {transition.reason}")
            return state
        return target

    async def _flatten_residual_exposure(
        self,
        platform: PlatformInterface,
        market_id: str,
        orders: List,
        expected_qty: int,
        opportunity_id: str,
    ) -> None:
        """Flatten any partial exposure from paired execution failures."""
        for order in orders:
            if order is None:
                continue
            filled = max(0, int(getattr(order, "filled_quantity", 0)))
            if filled <= 0:
                continue

            # Flatten only the filled exposure.
            flatten_qty = min(filled, expected_qty)
            flatten_side = OrderSide.NO if order.side == OrderSide.YES else OrderSide.YES
            try:
                await platform.place_order(
                    market_id=market_id,
                    side=flatten_side,
                    order_type=OrderType.MARKET,
                    quantity=flatten_qty,
                )
                logger.warning(
                    f"Flattened residual exposure for {opportunity_id}: "
                    f"{flatten_side.value.upper()} x{flatten_qty}"
                )
            except Exception as e:
                logger.error(f"Failed flatten order for {opportunity_id}: {e}")
    
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
            
            pnl = position.unrealized_pnl
            self.state.daily_pnl += pnl
            
            # Calculate hold time
            if hasattr(position, 'opened_at') and position.opened_at:
                hold_duration = datetime.now(timezone.utc) - position.opened_at
                hours = int(hold_duration.total_seconds() // 3600)
                minutes = int((hold_duration.total_seconds() % 3600) // 60)
                hold_time = f"{hours}h {minutes}m"
            else:
                hold_time = "Unknown"
            
            # Calculate P&L %
            entry_value = position.avg_price * Decimal(str(position.quantity))
            pnl_pct = float(pnl / entry_value) if entry_value > 0 else 0.0
            
            # Send exit notification
            if self.notifications:
                await self.notifications.send_position_closed(
                    ticker=position.ticker,
                    pnl=float(pnl),
                    pnl_pct=pnl_pct,
                    hold_time=hold_time,
                    reason=signal.reason,
                    entry_price=float(position.avg_price),
                    exit_price=float(signal.target_price) if signal.target_price else None
                )
                
        except Exception as e:
            logger.error(f"Exit failed: {e}")
            if self.notifications:
                await self.notifications.send_error(
                    error_msg=f"Exit order failed: {e}",
                    context=f"Position: {position.ticker}"
                )
    
    async def _get_total_balance(self) -> Decimal:
        """Get combined balance across all platforms."""
        total = Decimal("0")
        for platform in self.platforms.values():
            try:
                total += await platform.get_balance()
            except:
                pass
        return total
