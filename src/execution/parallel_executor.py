"""
PR3DICT: Parallel Execution Engine

Executes multi-leg arbitrage trades atomically within same Polygon block.
Critical requirement: All legs must fill within <30ms window.

Architecture:
1. Submit all orders simultaneously (asyncio.gather)
2. Track confirmation status per leg
3. Rollback incomplete arbitrage (cancel unfilled legs)
4. Slippage protection and gas optimization
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Optional, Tuple, Callable
from collections import defaultdict

from ..platforms.base import (
    PlatformInterface, Market, Order, OrderSide, 
    OrderType, OrderStatus
)
from ..risk.manager import RiskManager
from .metrics import ExecutionMetrics, MetricsCollector

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategy for multi-leg trades."""
    MARKET = "market"  # Fast, high slippage
    LIMIT = "limit"  # Slow, low slippage
    HYBRID = "hybrid"  # Limit with fallback to market


class LegStatus(Enum):
    """Status of individual leg in multi-leg trade."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TradeLeg:
    """Single leg of a multi-leg arbitrage trade."""
    market_id: str
    side: OrderSide
    quantity: int
    target_price: Optional[Decimal]
    platform: str
    
    # Execution state
    status: LegStatus = LegStatus.PENDING
    order: Optional[Order] = None
    filled_quantity: int = 0
    avg_fill_price: Optional[Decimal] = None
    submission_time: Optional[float] = None
    fill_time: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def is_filled(self) -> bool:
        """Check if leg is completely filled."""
        return self.status == LegStatus.FILLED
    
    @property
    def is_pending(self) -> bool:
        """Check if leg needs action."""
        return self.status in (LegStatus.PENDING, LegStatus.SUBMITTED, LegStatus.PARTIALLY_FILLED)
    
    @property
    def execution_time_ms(self) -> Optional[float]:
        """Calculate execution time in milliseconds."""
        if self.submission_time and self.fill_time:
            return (self.fill_time - self.submission_time) * 1000
        return None


@dataclass
class MultiLegTrade:
    """Multi-leg arbitrage trade requiring atomic execution."""
    trade_id: str
    legs: List[TradeLeg]
    strategy: ExecutionStrategy
    max_slippage_pct: Decimal = Decimal("0.02")  # 2% max slippage
    timeout_ms: int = 30  # Must execute within 30ms for same block
    
    # Execution state
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    committed: bool = False
    rolled_back: bool = False
    expected_profit: Optional[Decimal] = None
    actual_profit: Optional[Decimal] = None
    
    @property
    def all_filled(self) -> bool:
        """Check if all legs are filled."""
        return all(leg.is_filled for leg in self.legs)
    
    @property
    def any_failed(self) -> bool:
        """Check if any leg failed."""
        return any(leg.status == LegStatus.FAILED for leg in self.legs)
    
    @property
    def execution_time_ms(self) -> Optional[float]:
        """Total execution time in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    @property
    def slippage_pct(self) -> Optional[Decimal]:
        """Calculate actual slippage percentage."""
        if not self.expected_profit or not self.actual_profit:
            return None
        if self.expected_profit == 0:
            return None
        return (self.expected_profit - self.actual_profit) / self.expected_profit


@dataclass
class ExecutionConfig:
    """Configuration for parallel execution engine."""
    # Timing constraints
    max_execution_time_ms: int = 30  # Polygon block time
    order_submission_delay_ms: int = 0  # Delay between order submissions
    
    # Retry logic
    max_retries: int = 3
    retry_delay_ms: int = 50
    
    # Slippage protection
    max_slippage_pct: Decimal = Decimal("0.03")  # 3% max
    slippage_check_enabled: bool = True
    
    # Gas optimization (Polygon)
    use_batch_transactions: bool = True
    max_gas_price_gwei: Decimal = Decimal("500")  # Polygon typical: 30-100 gwei
    gas_price_multiplier: Decimal = Decimal("1.2")  # 20% buffer
    
    # Fallback behavior
    hybrid_fallback_timeout_ms: int = 15  # Switch to market after 15ms
    cancel_on_partial_fill: bool = True
    
    # RPC endpoint management
    primary_rpc: str = "https://polygon-rpc.com"
    fallback_rpcs: List[str] = field(default_factory=lambda: [
        "https://rpc-mainnet.matic.network",
        "https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY",
        "https://rpc-mainnet.maticvigil.com"
    ])


class ParallelExecutor:
    """
    Parallel execution engine for atomic multi-leg arbitrage trades.
    
    Key Features:
    - Simultaneous order submission across all legs
    - Per-leg status tracking and confirmation
    - Atomic commitment: all or nothing
    - Automatic rollback on incomplete execution
    - Strategy-based execution (market/limit/hybrid)
    - Polygon optimizations (batch txs, gas management)
    """
    
    def __init__(self,
                 platforms: Dict[str, PlatformInterface],
                 risk_manager: RiskManager,
                 config: Optional[ExecutionConfig] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        self.platforms = platforms
        self.risk = risk_manager
        self.config = config or ExecutionConfig()
        self.metrics = metrics_collector or MetricsCollector()
        
        # State tracking
        self._active_trades: Dict[str, MultiLegTrade] = {}
        self._trade_counter = 0
        
    async def execute_arbitrage(self,
                               legs: List[TradeLeg],
                               strategy: ExecutionStrategy = ExecutionStrategy.HYBRID,
                               expected_profit: Optional[Decimal] = None) -> MultiLegTrade:
        """
        Execute multi-leg arbitrage trade atomically.
        
        Args:
            legs: List of trade legs to execute
            strategy: Execution strategy (market/limit/hybrid)
            expected_profit: Expected profit for slippage calculation
            
        Returns:
            MultiLegTrade with execution results
        """
        # Create trade
        self._trade_counter += 1
        trade_id = f"arb_{int(time.time())}_{self._trade_counter}"
        
        trade = MultiLegTrade(
            trade_id=trade_id,
            legs=legs,
            strategy=strategy,
            max_slippage_pct=self.config.max_slippage_pct,
            timeout_ms=self.config.max_execution_time_ms,
            expected_profit=expected_profit
        )
        
        self._active_trades[trade_id] = trade
        
        try:
            # Pre-flight checks
            if not await self._preflight_checks(trade):
                logger.error(f"Trade {trade_id} failed preflight checks")
                return trade
            
            # Execute based on strategy
            if strategy == ExecutionStrategy.MARKET:
                await self._execute_market(trade)
            elif strategy == ExecutionStrategy.LIMIT:
                await self._execute_limit(trade)
            elif strategy == ExecutionStrategy.HYBRID:
                await self._execute_hybrid(trade)
            
            # Verify and commit/rollback
            await self._finalize_trade(trade)
            
        except Exception as e:
            logger.error(f"Trade {trade_id} execution error: {e}", exc_info=True)
            await self._rollback_trade(trade, reason=str(e))
        
        finally:
            # Record metrics
            self.metrics.record_trade(trade)
            del self._active_trades[trade_id]
        
        return trade
    
    async def _preflight_checks(self, trade: MultiLegTrade) -> bool:
        """
        Run pre-execution validation checks.
        
        Checks:
        - Risk manager approval
        - Capital availability
        - Position limits
        - Platform connectivity
        """
        # Risk gate check
        allowed, reason = self.risk.check_trade_allowed()
        if not allowed:
            logger.warning(f"Trade blocked by risk: {reason}")
            return False
        
        # Capital check - sum all leg requirements
        total_capital_needed = Decimal("0")
        for leg in trade.legs:
            price = leg.target_price or Decimal("0.5")  # Estimate if no target
            capital_needed = price * Decimal(str(leg.quantity))
            total_capital_needed += capital_needed
        
        # Get total available balance
        total_balance = Decimal("0")
        for platform in self.platforms.values():
            try:
                total_balance += await platform.get_balance()
            except Exception as e:
                logger.error(f"Failed to get balance from {platform.name}: {e}")
                return False
        
        if total_capital_needed > total_balance:
            logger.warning(f"Insufficient capital: need {total_capital_needed}, have {total_balance}")
            return False
        
        # Position size validation per leg
        for leg in trade.legs:
            price = leg.target_price or Decimal("0.5")
            if not self.risk.validate_position_size(leg.quantity, price):
                logger.warning(f"Leg position size rejected: {leg.quantity} @ {price}")
                return False
        
        # Platform connectivity check
        for leg in trade.legs:
            platform = self.platforms.get(leg.platform)
            if not platform:
                logger.error(f"Platform not available: {leg.platform}")
                return False
        
        logger.info(f"Trade {trade.trade_id} passed preflight checks")
        return True
    
    async def _execute_market(self, trade: MultiLegTrade) -> None:
        """
        Execute all legs with market orders (fast, high slippage).
        
        Submits all orders simultaneously for fastest execution.
        """
        logger.info(f"Executing trade {trade.trade_id} with MARKET orders")
        trade.start_time = time.time()
        
        # Submit all legs simultaneously
        tasks = []
        for leg in trade.legs:
            task = self._submit_order(leg, OrderType.MARKET)
            tasks.append(task)
        
        # Wait for all submissions
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Wait for fills (with timeout)
        await self._wait_for_fills(trade, timeout_ms=self.config.max_execution_time_ms)
        
        trade.end_time = time.time()
    
    async def _execute_limit(self, trade: MultiLegTrade) -> None:
        """
        Execute all legs with limit orders (slow, low slippage).
        
        Places orders at target prices and waits for fills.
        """
        logger.info(f"Executing trade {trade.trade_id} with LIMIT orders")
        trade.start_time = time.time()
        
        # Submit all limit orders
        tasks = []
        for leg in trade.legs:
            if not leg.target_price:
                logger.error(f"Leg missing target price for limit order")
                leg.status = LegStatus.FAILED
                leg.error = "No target price for limit order"
                continue
            
            task = self._submit_order(leg, OrderType.LIMIT)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Wait for fills (longer timeout for limit orders)
        await self._wait_for_fills(trade, timeout_ms=self.config.max_execution_time_ms * 10)
        
        trade.end_time = time.time()
    
    async def _execute_hybrid(self, trade: MultiLegTrade) -> None:
        """
        Execute with limit orders, fallback to market if timeout.
        
        Strategy:
        1. Submit all as limit orders
        2. Wait for fallback timeout
        3. Convert unfilled to market orders
        """
        logger.info(f"Executing trade {trade.trade_id} with HYBRID strategy")
        trade.start_time = time.time()
        
        # Phase 1: Submit limit orders
        tasks = []
        for leg in trade.legs:
            if leg.target_price:
                task = self._submit_order(leg, OrderType.LIMIT)
            else:
                # No target price, go straight to market
                task = self._submit_order(leg, OrderType.MARKET)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Phase 2: Wait for initial fills
        fallback_timeout = self.config.hybrid_fallback_timeout_ms
        await self._wait_for_fills(trade, timeout_ms=fallback_timeout)
        
        # Phase 3: Fallback to market for unfilled legs
        unfilled_legs = [leg for leg in trade.legs if leg.is_pending]
        
        if unfilled_legs:
            logger.info(f"Trade {trade.trade_id}: {len(unfilled_legs)} legs unfilled, converting to market")
            
            # Cancel existing limit orders
            cancel_tasks = []
            for leg in unfilled_legs:
                if leg.order:
                    cancel_tasks.append(self._cancel_order(leg))
            
            if cancel_tasks:
                await asyncio.gather(*cancel_tasks, return_exceptions=True)
            
            # Resubmit as market orders
            market_tasks = []
            for leg in unfilled_legs:
                leg.status = LegStatus.PENDING  # Reset for resubmission
                market_tasks.append(self._submit_order(leg, OrderType.MARKET))
            
            await asyncio.gather(*market_tasks, return_exceptions=True)
            
            # Wait for market fills
            remaining_time = self.config.max_execution_time_ms - fallback_timeout
            await self._wait_for_fills(trade, timeout_ms=max(remaining_time, 100))
        
        trade.end_time = time.time()
    
    async def _submit_order(self, leg: TradeLeg, order_type: OrderType) -> None:
        """Submit order for a single leg."""
        platform = self.platforms.get(leg.platform)
        if not platform:
            leg.status = LegStatus.FAILED
            leg.error = f"Platform {leg.platform} not available"
            return
        
        try:
            leg.status = LegStatus.SUBMITTED
            leg.submission_time = time.time()
            
            order = await platform.place_order(
                market_id=leg.market_id,
                side=leg.side,
                order_type=order_type,
                quantity=leg.quantity,
                price=leg.target_price if order_type == OrderType.LIMIT else None
            )
            
            leg.order = order
            logger.debug(f"Order submitted: {order.id} for {leg.market_id}")
            
        except Exception as e:
            leg.status = LegStatus.FAILED
            leg.error = str(e)
            logger.error(f"Order submission failed for {leg.market_id}: {e}")
    
    async def _wait_for_fills(self, trade: MultiLegTrade, timeout_ms: int) -> None:
        """
        Wait for all legs to fill within timeout.
        
        Polls order status at regular intervals.
        """
        start = time.time()
        poll_interval = 0.1  # 100ms polling
        
        while True:
            elapsed_ms = (time.time() - start) * 1000
            
            if elapsed_ms >= timeout_ms:
                logger.warning(f"Trade {trade.trade_id} timeout after {elapsed_ms:.1f}ms")
                break
            
            # Check all pending legs
            check_tasks = []
            for leg in trade.legs:
                if leg.is_pending and leg.order:
                    check_tasks.append(self._check_order_status(leg))
            
            if check_tasks:
                await asyncio.gather(*check_tasks, return_exceptions=True)
            
            # Check if all filled
            if trade.all_filled:
                logger.info(f"Trade {trade.trade_id} all legs filled in {elapsed_ms:.1f}ms")
                break
            
            # Check if any failed
            if trade.any_failed:
                logger.warning(f"Trade {trade.trade_id} has failed legs")
                break
            
            await asyncio.sleep(poll_interval)
    
    async def _check_order_status(self, leg: TradeLeg) -> None:
        """Check and update status of order for a leg."""
        if not leg.order:
            return
        
        platform = self.platforms.get(leg.platform)
        if not platform:
            return
        
        try:
            # Fetch updated order status
            orders = await platform.get_orders(status=None)
            updated_order = next((o for o in orders if o.id == leg.order.id), None)
            
            if updated_order:
                leg.order = updated_order
                leg.filled_quantity = updated_order.filled_quantity
                
                if updated_order.status == OrderStatus.FILLED:
                    leg.status = LegStatus.FILLED
                    leg.fill_time = time.time()
                    leg.avg_fill_price = updated_order.price
                    logger.debug(f"Leg filled: {leg.market_id} @ {leg.avg_fill_price}")
                    
                elif updated_order.status == OrderStatus.PARTIALLY_FILLED:
                    leg.status = LegStatus.PARTIALLY_FILLED
                    
                elif updated_order.status in (OrderStatus.CANCELLED, OrderStatus.EXPIRED):
                    leg.status = LegStatus.FAILED
                    leg.error = f"Order {updated_order.status.value}"
                    
        except Exception as e:
            logger.error(f"Failed to check order status for {leg.market_id}: {e}")
    
    async def _cancel_order(self, leg: TradeLeg) -> None:
        """Cancel order for a leg."""
        if not leg.order:
            return
        
        platform = self.platforms.get(leg.platform)
        if not platform:
            return
        
        try:
            success = await platform.cancel_order(leg.order.id)
            if success:
                leg.status = LegStatus.CANCELLED
                logger.debug(f"Order cancelled: {leg.order.id}")
        except Exception as e:
            logger.error(f"Failed to cancel order {leg.order.id}: {e}")
    
    async def _finalize_trade(self, trade: MultiLegTrade) -> None:
        """
        Finalize trade: commit if all filled, rollback if incomplete.
        """
        if trade.all_filled:
            await self._commit_trade(trade)
        else:
            await self._rollback_trade(trade, reason="Incomplete execution")
    
    async def _commit_trade(self, trade: MultiLegTrade) -> None:
        """
        Commit trade: validate fills and calculate actual profit.
        """
        logger.info(f"Committing trade {trade.trade_id}")
        
        # Calculate actual profit
        actual_profit = Decimal("0")
        for leg in trade.legs:
            if leg.avg_fill_price and leg.filled_quantity:
                leg_value = leg.avg_fill_price * Decimal(str(leg.filled_quantity))
                if leg.side == OrderSide.YES:
                    actual_profit -= leg_value  # Cost
                else:
                    actual_profit += leg_value  # Credit (selling NO)
        
        trade.actual_profit = actual_profit
        trade.committed = True
        
        # Slippage check
        if self.config.slippage_check_enabled and trade.expected_profit:
            slippage = trade.slippage_pct
            if slippage and slippage > trade.max_slippage_pct:
                logger.warning(f"Trade {trade.trade_id} exceeded slippage: {slippage:.2%}")
        
        logger.info(f"Trade {trade.trade_id} committed: "
                   f"Expected={trade.expected_profit}, Actual={actual_profit}, "
                   f"Time={trade.execution_time_ms:.1f}ms")
    
    async def _rollback_trade(self, trade: MultiLegTrade, reason: str) -> None:
        """
        Rollback trade: cancel unfilled orders, exit filled positions.
        """
        logger.warning(f"Rolling back trade {trade.trade_id}: {reason}")
        
        # Cancel all pending orders
        cancel_tasks = []
        for leg in trade.legs:
            if leg.is_pending and leg.order:
                cancel_tasks.append(self._cancel_order(leg))
        
        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)
        
        # For filled legs, attempt to exit (opposite side market order)
        exit_tasks = []
        for leg in trade.legs:
            if leg.is_filled and leg.filled_quantity > 0:
                exit_tasks.append(self._exit_leg(leg))
        
        if exit_tasks:
            await asyncio.gather(*exit_tasks, return_exceptions=True)
        
        trade.rolled_back = True
        logger.info(f"Trade {trade.trade_id} rolled back")
    
    async def _exit_leg(self, leg: TradeLeg) -> None:
        """Exit a filled leg with opposite side market order."""
        platform = self.platforms.get(leg.platform)
        if not platform:
            return
        
        try:
            # Opposite side
            exit_side = OrderSide.NO if leg.side == OrderSide.YES else OrderSide.YES
            
            await platform.place_order(
                market_id=leg.market_id,
                side=exit_side,
                order_type=OrderType.MARKET,
                quantity=leg.filled_quantity
            )
            
            logger.info(f"Exited leg: {leg.market_id} {exit_side.value} x{leg.filled_quantity}")
            
        except Exception as e:
            logger.error(f"Failed to exit leg {leg.market_id}: {e}")
    
    def get_active_trades(self) -> List[MultiLegTrade]:
        """Get list of currently active trades."""
        return list(self._active_trades.values())
    
    def get_metrics_summary(self) -> Dict:
        """Get execution metrics summary."""
        return self.metrics.get_summary()
