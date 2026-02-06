"""
PR3DICT: VWAP Risk Checks

Extends risk management with VWAP validation gates.
Prevents trading on signals that would execute with poor fills.
"""

import logging
from decimal import Decimal
from typing import Tuple, Optional
from dataclasses import dataclass

from .manager import RiskManager, RiskConfig
from ..data.vwap import VWAPCalculator, VWAPValidator, VWAPResult, LiquidityMetrics
from ..platforms.base import OrderBook, OrderSide

logger = logging.getLogger(__name__)


@dataclass
class VWAPRiskConfig:
    """VWAP-specific risk configuration."""
    # Slippage limits
    max_slippage_pct: Decimal = Decimal("2.0")  # Max 2% slippage
    slippage_warning_threshold: Decimal = Decimal("1.0")  # Warn at 1%
    
    # Liquidity requirements
    min_liquidity_contracts: int = 500  # Minimum depth per side
    min_top_of_book_size: int = 50  # Minimum size at best price
    
    # Spread limits
    max_spread_bps: int = 300  # Max 3% spread
    
    # Price impact
    max_price_impact_pct: Decimal = Decimal("3.0")  # Max 3% impact
    
    # Trade rejection
    reject_on_insufficient_liquidity: bool = True
    reject_on_high_slippage: bool = True
    reject_on_wide_spread: bool = True
    
    # Position adjustment
    enable_auto_adjustment: bool = True
    min_adjustment_pct: Decimal = Decimal("0.5")  # Adjust down to 50% of original


class VWAPRiskManager(RiskManager):
    """
    Enhanced risk manager with VWAP validation.
    
    Combines traditional risk checks with execution quality gates.
    """
    
    def __init__(self,
                 config: RiskConfig = None,
                 vwap_config: VWAPRiskConfig = None):
        """
        Args:
            config: Base risk configuration
            vwap_config: VWAP-specific risk configuration
        """
        super().__init__(config)
        self.vwap_config = vwap_config or VWAPRiskConfig()
        
        # VWAP components
        self.vwap_calculator = VWAPCalculator(
            slippage_warning_threshold=self.vwap_config.slippage_warning_threshold
        )
        self.vwap_validator = VWAPValidator(
            calculator=self.vwap_calculator,
            max_slippage_pct=self.vwap_config.max_slippage_pct,
            min_liquidity_contracts=self.vwap_config.min_liquidity_contracts,
            max_spread_bps=self.vwap_config.max_spread_bps
        )
        
        # Statistics
        self.vwap_rejections = 0
        self.vwap_adjustments = 0
    
    def check_trade_with_vwap(self,
                             market_id: str,
                             side: OrderSide,
                             quantity: int,
                             orderbook: OrderBook,
                             quoted_price: Optional[Decimal] = None) -> Tuple[bool, Optional[int], str]:
        """
        Comprehensive trade check including VWAP validation.
        
        Args:
            market_id: Market identifier
            side: Order side (YES/NO)
            quantity: Desired position size
            orderbook: Current order book
            quoted_price: Signal price (optional)
        
        Returns:
            (is_allowed, adjusted_quantity, reason) tuple
            - is_allowed: True if trade passes all checks
            - adjusted_quantity: Modified size if adjusted, None if rejected
            - reason: Explanation
        """
        # First, run traditional risk checks
        base_allowed, base_reason = self.check_trade_allowed()
        if not base_allowed:
            return False, None, base_reason
        
        # Convert side to string for VWAP
        side_str = "buy" if side == OrderSide.YES else "sell"
        orders = orderbook.asks if side == OrderSide.YES else orderbook.bids
        
        # Calculate VWAP
        vwap_result = self.vwap_calculator.calculate_vwap(
            orders=orders,
            quantity=quantity,
            side=side_str,
            market_id=market_id,
            quoted_price=quoted_price
        )
        
        # Calculate liquidity metrics
        liquidity = self.vwap_calculator.calculate_liquidity_metrics(
            bids=orderbook.bids,
            asks=orderbook.asks,
            market_id=market_id
        )
        
        # Validate execution quality
        is_valid, vwap_reason = self.vwap_validator.validate_execution(vwap_result, liquidity)
        
        if is_valid:
            # All checks passed
            logger.info(
                f"Trade approved: {market_id} {side_str} {quantity} "
                f"(VWAP: ${vwap_result.vwap_price:.4f}, "
                f"slippage: {vwap_result.slippage_pct:.2f}%, "
                f"quality: {vwap_result.execution_quality})"
            )
            return True, quantity, "APPROVED"
        
        # VWAP validation failed - check if we should adjust or reject
        
        # Insufficient liquidity
        if not vwap_result.liquidity_sufficient:
            if self.vwap_config.reject_on_insufficient_liquidity:
                if not self.vwap_config.enable_auto_adjustment:
                    self.vwap_rejections += 1
                    return False, None, f"VWAP_REJECTED: {vwap_reason}"
                
                # Try to find acceptable size
                adjusted_qty = self._find_acceptable_size(orders, side_str, market_id, quantity)
                
                if adjusted_qty >= int(quantity * self.vwap_config.min_adjustment_pct):
                    logger.warning(
                        f"Trade adjusted for liquidity: {market_id} {side_str} "
                        f"{quantity} -> {adjusted_qty}"
                    )
                    self.vwap_adjustments += 1
                    return True, adjusted_qty, f"ADJUSTED: {vwap_reason}"
                else:
                    self.vwap_rejections += 1
                    return False, None, f"VWAP_REJECTED: Insufficient liquidity, cannot adjust"
        
        # High slippage
        if vwap_result.slippage_pct > self.vwap_config.max_slippage_pct:
            if self.vwap_config.reject_on_high_slippage:
                self.vwap_rejections += 1
                return False, None, f"VWAP_REJECTED: Slippage {vwap_result.slippage_pct:.2f}% too high"
        
        # Wide spread
        if liquidity.spread_bps > self.vwap_config.max_spread_bps:
            if self.vwap_config.reject_on_wide_spread:
                self.vwap_rejections += 1
                return False, None, f"VWAP_REJECTED: Spread {liquidity.spread_bps}bps too wide"
        
        # Generic VWAP rejection
        self.vwap_rejections += 1
        return False, None, f"VWAP_REJECTED: {vwap_reason}"
    
    def _find_acceptable_size(self,
                             orders: list,
                             side: str,
                             market_id: str,
                             max_quantity: int) -> int:
        """
        Binary search for largest acceptable position size.
        
        Returns:
            Largest size that passes VWAP validation
        """
        if not orders:
            return 0
        
        max_available = sum(size for _, size in orders)
        low, high = 10, min(max_quantity, max_available)
        best_size = 0
        
        while low <= high:
            mid = (low + high) // 2
            
            vwap = self.vwap_calculator.calculate_vwap(orders, mid, side, market_id)
            
            # Check if this size is acceptable
            acceptable = (
                vwap.liquidity_sufficient and
                vwap.slippage_pct <= self.vwap_config.max_slippage_pct
            )
            
            if acceptable:
                best_size = mid
                low = mid + 1  # Try larger
            else:
                high = mid - 1  # Try smaller
        
        return best_size
    
    def validate_execution_price(self,
                                market_id: str,
                                side: OrderSide,
                                quantity: int,
                                expected_price: Decimal,
                                orderbook: OrderBook) -> Tuple[bool, str]:
        """
        Validate that expected execution price is realistic.
        
        Prevents the "quoted profit but executed loss" scenario.
        
        Args:
            market_id: Market identifier
            side: Order side
            quantity: Position size
            expected_price: Price strategy expects to get
            orderbook: Current order book
        
        Returns:
            (is_realistic, reason) tuple
        """
        side_str = "buy" if side == OrderSide.YES else "sell"
        orders = orderbook.asks if side == OrderSide.YES else orderbook.bids
        
        # Calculate actual VWAP
        vwap_result = self.vwap_calculator.calculate_vwap(
            orders=orders,
            quantity=quantity,
            side=side_str,
            market_id=market_id,
            quoted_price=expected_price
        )
        
        # Check deviation
        deviation_pct = abs(vwap_result.vwap_price - expected_price) / expected_price * 100
        
        if deviation_pct > Decimal("5.0"):  # >5% deviation
            return False, (
                f"Expected price ${expected_price:.4f} unrealistic. "
                f"VWAP: ${vwap_result.vwap_price:.4f} "
                f"(deviation: {deviation_pct:.2f}%)"
            )
        
        if deviation_pct > Decimal("2.0"):  # Warning level
            logger.warning(
                f"Price deviation: {market_id} {side_str} "
                f"expected ${expected_price:.4f}, VWAP ${vwap_result.vwap_price:.4f}"
            )
        
        return True, "OK"
    
    def check_minimum_profit_after_slippage(self,
                                           signal_price: Decimal,
                                           vwap_price: Decimal,
                                           side: OrderSide,
                                           min_edge_pct: Decimal = Decimal("2.0")) -> Tuple[bool, str]:
        """
        Verify trade maintains minimum edge after slippage.
        
        Args:
            signal_price: Price at signal generation
            vwap_price: Expected execution price (VWAP)
            side: Order side
            min_edge_pct: Minimum required edge (default 2%)
        
        Returns:
            (has_edge, reason) tuple
        """
        if side == OrderSide.YES:
            # Buying YES at VWAP price
            # Max payoff is 1.00, edge is 1.00 - vwap_price
            edge = (Decimal("1.00") - vwap_price) / vwap_price * 100
        else:
            # Buying NO (selling YES)
            # Edge is vwap_price (what we get paid)
            edge = vwap_price / (Decimal("1.00") - vwap_price) * 100
        
        if edge < min_edge_pct:
            return False, f"Edge {edge:.2f}% below minimum {min_edge_pct}% after slippage"
        
        return True, f"Edge {edge:.2f}% acceptable"
    
    def get_vwap_statistics(self) -> dict:
        """Get VWAP risk check statistics."""
        total_checks = self.vwap_rejections + self.vwap_adjustments
        
        return {
            "vwap_rejections": self.vwap_rejections,
            "vwap_adjustments": self.vwap_adjustments,
            "total_vwap_checks": total_checks,
            "rejection_rate_pct": (self.vwap_rejections / total_checks * 100) 
                if total_checks > 0 else 0,
            "config": {
                "max_slippage_pct": str(self.vwap_config.max_slippage_pct),
                "min_liquidity": self.vwap_config.min_liquidity_contracts,
                "max_spread_bps": self.vwap_config.max_spread_bps,
                "auto_adjustment_enabled": self.vwap_config.enable_auto_adjustment
            }
        }
    
    def get_status(self) -> dict:
        """Get comprehensive risk status including VWAP metrics."""
        base_status = super().get_status()
        vwap_stats = self.get_vwap_statistics()
        
        return {
            **base_status,
            "vwap_checks": vwap_stats
        }


# Global instance
_global_vwap_risk_manager: Optional[VWAPRiskManager] = None


def get_vwap_risk_manager(
    risk_config: RiskConfig = None,
    vwap_config: VWAPRiskConfig = None
) -> VWAPRiskManager:
    """
    Get or create global VWAP risk manager.
    
    Args:
        risk_config: Base risk configuration
        vwap_config: VWAP risk configuration
    
    Returns:
        VWAPRiskManager instance
    """
    global _global_vwap_risk_manager
    
    if _global_vwap_risk_manager is None:
        _global_vwap_risk_manager = VWAPRiskManager(risk_config, vwap_config)
    
    return _global_vwap_risk_manager
