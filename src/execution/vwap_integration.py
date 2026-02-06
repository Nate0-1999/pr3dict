"""
PR3DICT: VWAP Integration Layer

Connects VWAP analysis to trading strategies and execution engine.
Validates signals before execution, adjusts position sizes, and enforces quality gates.
"""

import logging
from decimal import Decimal
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from ..data.vwap import (
    VWAPCalculator,
    VWAPValidator,
    VWAPMonitor,
    HistoricalVWAPAnalyzer,
    VWAPResult,
    LiquidityMetrics
)
from ..platforms.base import OrderBook, OrderSide

logger = logging.getLogger(__name__)


class VWAPTradingGate:
    """
    VWAP-based quality gate for trading signals.
    Prevents execution of signals that would result in poor fills.
    """
    
    def __init__(self,
                 max_slippage_pct: Decimal = Decimal("2.0"),
                 min_liquidity_contracts: int = 500,
                 max_spread_bps: int = 300,
                 enable_position_adjustment: bool = True):
        """
        Args:
            max_slippage_pct: Maximum acceptable slippage percentage
            min_liquidity_contracts: Minimum required depth per side
            max_spread_bps: Maximum bid-ask spread in basis points
            enable_position_adjustment: Auto-adjust position size if too large
        """
        self.calculator = VWAPCalculator(slippage_warning_threshold=max_slippage_pct)
        self.validator = VWAPValidator(
            calculator=self.calculator,
            max_slippage_pct=max_slippage_pct,
            min_liquidity_contracts=min_liquidity_contracts,
            max_spread_bps=max_spread_bps
        )
        self.monitor = VWAPMonitor(self.calculator)
        self.enable_adjustment = enable_position_adjustment
        
        # Statistics
        self.signals_processed = 0
        self.signals_blocked = 0
        self.signals_adjusted = 0
    
    def validate_signal(self,
                       market_id: str,
                       side: OrderSide,
                       quantity: int,
                       orderbook: OrderBook,
                       quoted_price: Optional[Decimal] = None) -> Tuple[bool, Optional[int], str]:
        """
        Validate a trading signal against VWAP quality standards.
        
        Args:
            market_id: Market identifier
            side: Order side (YES/NO)
            quantity: Desired position size
            orderbook: Current order book
            quoted_price: Reference price (mid or signal price)
        
        Returns:
            (is_valid, adjusted_quantity, reason) tuple
            - is_valid: True if signal passes quality checks
            - adjusted_quantity: Modified size if adjustment enabled, None otherwise
            - reason: Explanation of decision
        """
        self.signals_processed += 1
        
        # Convert OrderSide to string
        side_str = "buy" if side == OrderSide.YES else "sell"
        orders = orderbook.asks if side == OrderSide.YES else orderbook.bids
        
        # Calculate VWAP for requested quantity
        vwap_result = self.calculator.calculate_vwap(
            orders=orders,
            quantity=quantity,
            side=side_str,
            market_id=market_id,
            quoted_price=quoted_price
        )
        
        # Calculate liquidity metrics
        liquidity = self.calculator.calculate_liquidity_metrics(
            bids=orderbook.bids,
            asks=orderbook.asks,
            market_id=market_id
        )
        
        # Record for monitoring
        self.monitor.record_execution(vwap_result)
        self.monitor.record_liquidity_snapshot(liquidity)
        
        # Validate execution quality
        is_valid, reason = self.validator.validate_execution(vwap_result, liquidity)
        
        if is_valid:
            logger.info(
                f"Signal approved: {market_id} {side_str} {quantity} "
                f"(slippage: {vwap_result.slippage_pct:.2f}%, quality: {vwap_result.execution_quality})"
            )
            return True, quantity, "APPROVED"
        
        # If invalid but adjustment is enabled, try reducing size
        if self.enable_adjustment and not vwap_result.liquidity_sufficient:
            adjusted_qty = self._find_optimal_size(orders, side_str, market_id)
            
            if adjusted_qty >= quantity * Decimal("0.5"):  # At least 50% of original
                logger.warning(
                    f"Signal adjusted: {market_id} {side_str} "
                    f"{quantity} -> {adjusted_qty} (reason: {reason})"
                )
                self.signals_adjusted += 1
                return True, int(adjusted_qty), f"ADJUSTED: {reason}"
        
        # Block signal
        logger.warning(f"Signal blocked: {market_id} {side_str} {quantity} (reason: {reason})")
        self.signals_blocked += 1
        return False, None, f"BLOCKED: {reason}"
    
    def _find_optimal_size(self,
                          orders: List[Tuple[Decimal, int]],
                          side: str,
                          market_id: str) -> int:
        """
        Find largest position size that meets quality standards.
        
        Uses binary search to find optimal size.
        """
        if not orders:
            return 0
        
        max_available = sum(size for _, size in orders)
        low, high = 10, max_available
        best_size = 10
        
        while low <= high:
            mid = (low + high) // 2
            
            vwap = self.calculator.calculate_vwap(orders, mid, side, market_id)
            
            if vwap.liquidity_sufficient and vwap.slippage_pct <= self.validator.max_slippage_pct:
                best_size = mid
                low = mid + 1  # Try larger
            else:
                high = mid - 1  # Try smaller
        
        return best_size
    
    def get_statistics(self) -> Dict:
        """Get gate performance statistics."""
        return {
            "signals_processed": self.signals_processed,
            "signals_blocked": self.signals_blocked,
            "signals_adjusted": self.signals_adjusted,
            "block_rate_pct": (self.signals_blocked / self.signals_processed * 100)
                if self.signals_processed > 0 else 0,
            "adjustment_rate_pct": (self.signals_adjusted / self.signals_processed * 100)
                if self.signals_processed > 0 else 0
        }


class VWAPEnrichedSignal:
    """
    Trading signal enriched with VWAP analysis.
    Used by strategies to make informed execution decisions.
    """
    
    def __init__(self,
                 market_id: str,
                 side: OrderSide,
                 quantity: int,
                 signal_price: Decimal,
                 vwap_result: VWAPResult,
                 liquidity_metrics: LiquidityMetrics):
        self.market_id = market_id
        self.side = side
        self.quantity = quantity
        self.signal_price = signal_price
        self.vwap_result = vwap_result
        self.liquidity_metrics = liquidity_metrics
        self.timestamp = datetime.utcnow()
    
    @property
    def expected_profit_after_slippage(self) -> Decimal:
        """
        Calculate expected profit accounting for slippage.
        
        Returns:
            Profit per contract after VWAP execution
        """
        if self.side == OrderSide.YES:
            # Buying YES: profit if resolves to 1.00
            execution_cost = self.vwap_result.vwap_price
            max_payoff = Decimal("1.00")
            return max_payoff - execution_cost
        else:
            # Buying NO (selling YES): profit if resolves to 0.00
            execution_price = self.vwap_result.vwap_price
            max_payoff = Decimal("1.00")
            return execution_price  # We receive this when selling
    
    @property
    def is_profitable_after_slippage(self) -> bool:
        """Check if signal remains profitable after accounting for VWAP."""
        expected_profit = self.expected_profit_after_slippage
        
        # Need positive edge after execution costs
        min_edge = Decimal("0.02")  # 2% minimum edge
        return expected_profit > min_edge
    
    @property
    def quality_score(self) -> Decimal:
        """
        Calculate overall execution quality score (0-100).
        
        Factors:
        - Slippage (40%)
        - Liquidity depth (30%)
        - Spread (20%)
        - Execution certainty (10%)
        """
        # Slippage score (lower is better)
        slippage_score = max(Decimal("0"), Decimal("100") - self.vwap_result.slippage_pct * 20)
        
        # Liquidity score
        total_depth = self.liquidity_metrics.bid_depth + self.liquidity_metrics.ask_depth
        liquidity_score = min(Decimal("100"), total_depth / 50)  # 5000 contracts = perfect
        
        # Spread score (lower is better)
        spread_score = max(Decimal("0"), Decimal("100") - self.liquidity_metrics.spread_bps / 10)
        
        # Execution certainty (did we have enough liquidity?)
        certainty_score = Decimal("100") if self.vwap_result.liquidity_sufficient else Decimal("0")
        
        # Weighted average
        total = (
            slippage_score * Decimal("0.4") +
            liquidity_score * Decimal("0.3") +
            spread_score * Decimal("0.2") +
            certainty_score * Decimal("0.1")
        )
        
        return total
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary for logging/storage."""
        return {
            "market_id": self.market_id,
            "side": self.side.value,
            "quantity": self.quantity,
            "signal_price": str(self.signal_price),
            "vwap_price": str(self.vwap_result.vwap_price),
            "slippage_pct": str(self.vwap_result.slippage_pct),
            "expected_profit": str(self.expected_profit_after_slippage),
            "is_profitable": self.is_profitable_after_slippage,
            "quality_score": str(self.quality_score),
            "execution_quality": self.vwap_result.execution_quality,
            "liquidity_healthy": self.liquidity_metrics.is_healthy,
            "timestamp": self.timestamp.isoformat()
        }


class StrategyVWAPIntegration:
    """
    Mixin class for strategies to integrate VWAP analysis.
    Provides convenience methods for VWAP-aware signal generation.
    """
    
    def __init__(self, vwap_gate: VWAPTradingGate):
        self.vwap_gate = vwap_gate
    
    def enrich_signal_with_vwap(self,
                                market_id: str,
                                side: OrderSide,
                                quantity: int,
                                signal_price: Decimal,
                                orderbook: OrderBook) -> Optional[VWAPEnrichedSignal]:
        """
        Enrich a trading signal with VWAP analysis.
        
        Args:
            market_id: Market identifier
            side: Order side
            quantity: Position size
            signal_price: Strategy's quoted price
            orderbook: Current order book
        
        Returns:
            VWAPEnrichedSignal if valid, None if blocked
        """
        side_str = "buy" if side == OrderSide.YES else "sell"
        orders = orderbook.asks if side == OrderSide.YES else orderbook.bids
        
        # Calculate VWAP
        vwap_result = self.vwap_gate.calculator.calculate_vwap(
            orders=orders,
            quantity=quantity,
            side=side_str,
            market_id=market_id,
            quoted_price=signal_price
        )
        
        # Get liquidity metrics
        liquidity = self.vwap_gate.calculator.calculate_liquidity_metrics(
            bids=orderbook.bids,
            asks=orderbook.asks,
            market_id=market_id
        )
        
        # Record
        self.vwap_gate.monitor.record_execution(vwap_result)
        self.vwap_gate.monitor.record_liquidity_snapshot(liquidity)
        
        # Validate
        is_valid, reason = self.vwap_gate.validator.validate_execution(vwap_result, liquidity)
        
        if not is_valid:
            logger.warning(f"Signal enrichment failed: {reason}")
            return None
        
        return VWAPEnrichedSignal(
            market_id=market_id,
            side=side,
            quantity=quantity,
            signal_price=signal_price,
            vwap_result=vwap_result,
            liquidity_metrics=liquidity
        )
    
    def adjust_position_size_for_liquidity(self,
                                          market_id: str,
                                          side: OrderSide,
                                          target_capital: Decimal,
                                          orderbook: OrderBook) -> int:
        """
        Determine optimal position size based on liquidity constraints.
        
        Args:
            market_id: Market identifier
            side: Order side
            target_capital: How much capital to deploy (USDC)
            orderbook: Current order book
        
        Returns:
            Optimal position size in contracts
        """
        side_str = "buy" if side == OrderSide.YES else "sell"
        orders = orderbook.asks if side == OrderSide.YES else orderbook.bids
        
        if not orders:
            return 0
        
        # Start with naive quantity
        best_price = orders[0][0]
        naive_quantity = int(target_capital / best_price)
        
        # Find largest size that meets quality standards
        optimal_size = self.vwap_gate._find_optimal_size(orders, side_str, market_id)
        
        # Use smaller of naive or optimal
        final_size = min(naive_quantity, optimal_size)
        
        logger.info(
            f"Position sizing: {market_id} {side_str} "
            f"target_capital=${target_capital} -> {final_size} contracts "
            f"(naive: {naive_quantity}, optimal: {optimal_size})"
        )
        
        return final_size
    
    def split_large_order(self,
                         market_id: str,
                         side: OrderSide,
                         quantity: int,
                         orderbook: OrderBook,
                         max_chunks: int = 5) -> List[int]:
        """
        Split a large order into smaller chunks for better execution.
        
        Args:
            market_id: Market identifier
            side: Order side
            quantity: Total desired quantity
            orderbook: Current order book
            max_chunks: Maximum number of sub-orders
        
        Returns:
            List of order sizes
        """
        side_str = "buy" if side == OrderSide.YES else "sell"
        orders = orderbook.asks if side == OrderSide.YES else orderbook.bids
        
        # Calculate VWAP for full size
        vwap_result = self.vwap_gate.calculator.calculate_vwap(
            orders=orders,
            quantity=quantity,
            side=side_str,
            market_id=market_id
        )
        
        # Get split suggestion
        chunks = self.vwap_gate.validator.suggest_order_split(vwap_result, max_chunks)
        
        logger.info(
            f"Order split: {market_id} {side_str} {quantity} -> {len(chunks)} chunks: {chunks}"
        )
        
        return chunks


# Singleton instance for easy access
_global_vwap_gate: Optional[VWAPTradingGate] = None


def get_vwap_gate(
    max_slippage_pct: Decimal = Decimal("2.0"),
    min_liquidity_contracts: int = 500,
    max_spread_bps: int = 300
) -> VWAPTradingGate:
    """
    Get or create global VWAP trading gate instance.
    
    Args:
        max_slippage_pct: Maximum acceptable slippage
        min_liquidity_contracts: Minimum liquidity requirement
        max_spread_bps: Maximum spread tolerance
    
    Returns:
        VWAPTradingGate instance
    """
    global _global_vwap_gate
    
    if _global_vwap_gate is None:
        _global_vwap_gate = VWAPTradingGate(
            max_slippage_pct=max_slippage_pct,
            min_liquidity_contracts=min_liquidity_contracts,
            max_spread_bps=max_spread_bps
        )
    
    return _global_vwap_gate
