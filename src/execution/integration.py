"""
PR3DICT: Parallel Executor Integration

Integration layer connecting ParallelExecutor with TradingEngine
and ArbitrageStrategy.

This module shows how to:
1. Detect multi-leg arbitrage opportunities
2. Execute atomically with ParallelExecutor
3. Handle results and metrics
"""
import asyncio
import logging
from decimal import Decimal
from typing import List, Optional, Dict
from dataclasses import dataclass

from .parallel_executor import (
    ParallelExecutor, ExecutionStrategy, TradeLeg, MultiLegTrade
)
from .metrics import MetricsCollector
from .polygon_optimizer import PolygonOptimizer
from ..platforms.base import PlatformInterface, Market, OrderSide
from ..risk.manager import RiskManager
from ..strategies.base import Signal

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity."""
    opportunity_id: str
    opportunity_type: str  # "binary_complement", "cross_platform"
    markets: List[Market]
    legs: List[TradeLeg]
    expected_profit: Decimal
    confidence: float  # 0-1
    time_discovered: float
    
    @property
    def profit_per_unit(self) -> Decimal:
        """Profit per unit of arbitrage."""
        total_quantity = sum(leg.quantity for leg in self.legs)
        if total_quantity == 0:
            return Decimal("0")
        return self.expected_profit / Decimal(str(total_quantity))


class ArbitrageExecutionEngine:
    """
    High-level interface for detecting and executing arbitrage.
    
    Wraps ParallelExecutor with arbitrage-specific logic.
    """
    
    def __init__(self,
                 platforms: Dict[str, PlatformInterface],
                 risk_manager: RiskManager,
                 paper_mode: bool = True):
        """
        Initialize arbitrage execution engine.
        
        Args:
            platforms: Dictionary of platform name -> interface
            risk_manager: Risk manager instance
            paper_mode: If True, simulate execution without real trades
        """
        self.platforms = platforms
        self.risk = risk_manager
        self.paper_mode = paper_mode
        
        # Initialize components
        self.metrics = MetricsCollector()
        
        # Polygon optimizer (if using Polymarket)
        self.polygon_optimizer = PolygonOptimizer(
            rpc_endpoints=[
                "https://polygon-rpc.com",
                "https://rpc-mainnet.matic.network",
            ]
        )
        
        # Parallel executor
        self.executor = ParallelExecutor(
            platforms=platforms,
            risk_manager=risk_manager,
            metrics_collector=self.metrics
        )
        
        # Tracking
        self._opportunities_detected = 0
        self._opportunities_executed = 0
    
    async def detect_binary_complement_arb(self, 
                                          markets: List[Market],
                                          min_profit_pct: Decimal = Decimal("0.02")) -> List[ArbitrageOpportunity]:
        """
        Detect binary complement arbitrage opportunities.
        
        Binary complement arb: When YES + NO < $1.00, buy both for guaranteed profit.
        
        Args:
            markets: List of markets to scan
            min_profit_pct: Minimum profit percentage (e.g., 0.02 = 2%)
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        for market in markets:
            # Skip if resolved or no liquidity
            if market.resolved or market.liquidity < Decimal("1000"):
                continue
            
            # Check if YES + NO < 1.00
            total = market.yes_price + market.no_price
            if total >= Decimal("1.0"):
                continue
            
            # Calculate profit
            profit_per_unit = Decimal("1.0") - total
            profit_pct = profit_per_unit / total
            
            if profit_pct < min_profit_pct:
                continue
            
            # Calculate position size (risk-adjusted)
            balance = await self._get_total_balance()
            max_size = self.risk.calculate_position_size(
                account_value=balance,
                entry_price=total / Decimal("2"),  # Avg price
                signal_strength=float(profit_pct),
                win_rate=1.0,  # Guaranteed profit
                win_loss_ratio=10.0  # Asymmetric payoff
            )
            
            # Cap based on liquidity
            max_size = min(max_size, int(market.liquidity / total))
            
            if max_size < 1:
                continue
            
            # Create legs for both YES and NO
            legs = [
                TradeLeg(
                    market_id=market.id,
                    side=OrderSide.YES,
                    quantity=max_size,
                    target_price=market.yes_price,
                    platform=market.platform
                ),
                TradeLeg(
                    market_id=market.id,
                    side=OrderSide.NO,
                    quantity=max_size,
                    target_price=market.no_price,
                    platform=market.platform
                )
            ]
            
            expected_profit = profit_per_unit * Decimal(str(max_size))
            
            opportunity = ArbitrageOpportunity(
                opportunity_id=f"bc_{market.id}_{int(asyncio.get_event_loop().time())}",
                opportunity_type="binary_complement",
                markets=[market],
                legs=legs,
                expected_profit=expected_profit,
                confidence=1.0,  # Guaranteed if both fill
                time_discovered=asyncio.get_event_loop().time()
            )
            
            opportunities.append(opportunity)
            self._opportunities_detected += 1
            
            logger.info(f"Binary complement arb detected: {market.ticker} - "
                       f"{profit_pct:.2%} profit, size {max_size}")
        
        return opportunities
    
    async def detect_cross_platform_arb(self,
                                       markets: List[Market],
                                       min_differential: Decimal = Decimal("0.03")) -> List[ArbitrageOpportunity]:
        """
        Detect cross-platform arbitrage opportunities.
        
        Looks for same event priced differently on different platforms.
        
        Args:
            markets: List of markets to scan
            min_differential: Minimum price differential (e.g., 0.03 = 3%)
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        # Group markets by title/event (simplified - would need better matching)
        market_groups: Dict[str, List[Market]] = {}
        for market in markets:
            # Normalize title for matching
            key = market.title.lower().strip()
            if key not in market_groups:
                market_groups[key] = []
            market_groups[key].append(market)
        
        # Find pairs with price differentials
        for title, group in market_groups.items():
            if len(group) < 2:
                continue
            
            # Compare all pairs
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    market_a = group[i]
                    market_b = group[j]
                    
                    # Must be different platforms
                    if market_a.platform == market_b.platform:
                        continue
                    
                    # Check price differential
                    diff = abs(market_a.yes_price - market_b.yes_price)
                    
                    if diff < min_differential:
                        continue
                    
                    # Determine buy and sell sides
                    if market_a.yes_price < market_b.yes_price:
                        buy_market = market_a
                        sell_market = market_b
                    else:
                        buy_market = market_b
                        sell_market = market_a
                    
                    # Calculate size
                    balance = await self._get_total_balance()
                    max_size = self.risk.calculate_position_size(
                        account_value=balance,
                        entry_price=buy_market.yes_price,
                        signal_strength=float(diff),
                        win_rate=0.7,
                        win_loss_ratio=2.0
                    )
                    
                    # Cap by liquidity
                    max_size = min(
                        max_size,
                        int(buy_market.liquidity / buy_market.yes_price),
                        int(sell_market.liquidity / sell_market.yes_price)
                    )
                    
                    if max_size < 1:
                        continue
                    
                    # Create legs
                    legs = [
                        TradeLeg(
                            market_id=buy_market.id,
                            side=OrderSide.YES,
                            quantity=max_size,
                            target_price=buy_market.yes_price,
                            platform=buy_market.platform
                        ),
                        TradeLeg(
                            market_id=sell_market.id,
                            side=OrderSide.NO,  # Sell YES = Buy NO
                            quantity=max_size,
                            target_price=Decimal("1.0") - sell_market.yes_price,
                            platform=sell_market.platform
                        )
                    ]
                    
                    expected_profit = diff * Decimal(str(max_size))
                    
                    opportunity = ArbitrageOpportunity(
                        opportunity_id=f"xp_{buy_market.id}_{sell_market.id}",
                        opportunity_type="cross_platform",
                        markets=[buy_market, sell_market],
                        legs=legs,
                        expected_profit=expected_profit,
                        confidence=0.8,  # Less certain due to timing risk
                        time_discovered=asyncio.get_event_loop().time()
                    )
                    
                    opportunities.append(opportunity)
                    self._opportunities_detected += 1
                    
                    logger.info(f"Cross-platform arb detected: {title[:50]} - "
                               f"{diff:.2%} differential, size {max_size}")
        
        return opportunities
    
    async def execute_opportunity(self,
                                 opportunity: ArbitrageOpportunity,
                                 strategy: ExecutionStrategy = ExecutionStrategy.HYBRID) -> MultiLegTrade:
        """
        Execute an arbitrage opportunity.
        
        Args:
            opportunity: Arbitrage opportunity to execute
            strategy: Execution strategy (market/limit/hybrid)
            
        Returns:
            MultiLegTrade with execution results
        """
        if self.paper_mode:
            logger.info(f"[PAPER MODE] Would execute {opportunity.opportunity_id}")
            # Simulate successful execution
            trade = MultiLegTrade(
                trade_id=opportunity.opportunity_id,
                legs=opportunity.legs,
                strategy=strategy,
                expected_profit=opportunity.expected_profit
            )
            trade.committed = True
            trade.actual_profit = opportunity.expected_profit * Decimal("0.95")  # Simulate 5% slippage
            return trade
        
        logger.info(f"Executing arbitrage: {opportunity.opportunity_id} "
                   f"({opportunity.opportunity_type}) - "
                   f"Expected profit: {opportunity.expected_profit}")
        
        # Execute with parallel executor
        trade = await self.executor.execute_arbitrage(
            legs=opportunity.legs,
            strategy=strategy,
            expected_profit=opportunity.expected_profit
        )
        
        self._opportunities_executed += 1
        
        # Log results
        if trade.committed:
            logger.info(f"✓ Arbitrage successful: {opportunity.opportunity_id} - "
                       f"Profit: {trade.actual_profit}, Time: {trade.execution_time_ms:.1f}ms")
        else:
            logger.warning(f"✗ Arbitrage failed: {opportunity.opportunity_id} - "
                          f"Rolled back: {trade.rolled_back}")
        
        return trade
    
    async def scan_and_execute(self, 
                              markets: List[Market],
                              strategy: ExecutionStrategy = ExecutionStrategy.HYBRID,
                              max_opportunities: int = 5) -> List[MultiLegTrade]:
        """
        Scan for opportunities and execute them.
        
        Args:
            markets: Markets to scan
            strategy: Execution strategy
            max_opportunities: Maximum opportunities to execute per scan
            
        Returns:
            List of executed trades
        """
        # Detect opportunities
        bc_opportunities = await self.detect_binary_complement_arb(markets)
        xp_opportunities = await self.detect_cross_platform_arb(markets)
        
        all_opportunities = bc_opportunities + xp_opportunities
        
        # Sort by expected profit (highest first)
        all_opportunities.sort(key=lambda o: o.expected_profit, reverse=True)
        
        # Execute top opportunities
        trades = []
        for opportunity in all_opportunities[:max_opportunities]:
            trade = await self.execute_opportunity(opportunity, strategy)
            trades.append(trade)
            
            # Small delay between executions
            await asyncio.sleep(0.1)
        
        return trades
    
    async def _get_total_balance(self) -> Decimal:
        """Get total balance across all platforms."""
        total = Decimal("0")
        for platform in self.platforms.values():
            try:
                total += await platform.get_balance()
            except:
                pass
        return total
    
    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        metrics_summary = self.metrics.get_summary()
        
        return {
            "opportunities_detected": self._opportunities_detected,
            "opportunities_executed": self._opportunities_executed,
            "execution_rate": (self._opportunities_executed / self._opportunities_detected * 100)
                            if self._opportunities_detected > 0 else 0.0,
            "metrics": metrics_summary,
            "paper_mode": self.paper_mode
        }
