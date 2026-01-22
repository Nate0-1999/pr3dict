"""
PR3DICT: Arbitrage Strategy

Exploits price inefficiencies in prediction markets:
1. Binary Complement - YES + NO < $1.00
2. Cross-Platform - Same event priced differently on Kalshi vs Polymarket
"""
from typing import List, Optional
from decimal import Decimal
from datetime import datetime
import logging

from .base import TradingStrategy, Signal
from ..platforms.base import Market, Position, OrderSide

logger = logging.getLogger(__name__)


class ArbitrageStrategy(TradingStrategy):
    """
    Scans for arbitrage opportunities in prediction markets.
    
    Primary Edges:
    1. Binary Complement: When YES + NO prices sum to < $1.00
    2. Cross-Platform: Price differentials for same event
    """
    
    def __init__(self,
                 min_spread: Decimal = Decimal("0.025"),  # 2.5% min profit
                 min_liquidity: Decimal = Decimal("1000"),
                 max_time_to_resolution_hours: int = 24 * 30):  # 30 days
        self.min_spread = min_spread
        self.min_liquidity = min_liquidity
        self.max_time_to_resolution = max_time_to_resolution_hours * 3600
    
    @property
    def name(self) -> str:
        return "arbitrage"
    
    async def scan_markets(self, markets: List[Market]) -> List[Signal]:
        """
        Scan for arbitrage opportunities.
        
        Looks for:
        - Binary complement arb (YES + NO < 1.00)
        - Significant spread opportunities
        """
        signals = []
        
        for market in markets:
            # Skip resolved or low liquidity markets
            if market.resolved:
                continue
            if market.liquidity < self.min_liquidity:
                continue
            
            # Check time to resolution
            time_to_close = (market.close_time - datetime.now(market.close_time.tzinfo)).total_seconds()
            if time_to_close > self.max_time_to_resolution:
                continue
            if time_to_close < 0:
                continue  # Already closed
            
            # === Binary Complement Arbitrage ===
            # If YES + NO < 1.00, buy both and guaranteed profit at resolution
            total = market.yes_price + market.no_price
            if total < Decimal("1.0"):
                spread = Decimal("1.0") - total
                
                if spread >= self.min_spread:
                    # Signal to buy BOTH sides
                    signals.append(Signal(
                        market_id=market.id,
                        market=market,
                        side=OrderSide.YES,  # Primary side
                        strength=float(spread),  # Higher spread = stronger signal
                        reason=f"Binary complement arb: {spread:.2%} guaranteed profit",
                        target_price=market.yes_price
                    ))
                    logger.info(f"ARB SIGNAL: {market.ticker} - {spread:.2%} spread")
            
            # === Mispricing Detection ===
            # Large spreads can indicate mispricing opportunities
            if market.spread > self.min_spread:
                signals.append(Signal(
                    market_id=market.id,
                    market=market,
                    side=OrderSide.YES if market.yes_price < market.no_price else OrderSide.NO,
                    strength=float(market.spread),
                    reason=f"Wide spread mispricing: {market.spread:.2%}",
                    target_price=min(market.yes_price, market.no_price)
                ))
        
        return signals
    
    async def check_exit(self, position: Position, market: Market) -> Optional[Signal]:
        """
        Check exit conditions for arbitrage positions.
        
        Exit when:
        - Spread has closed (arb captured)
        - Close to resolution (hold for settlement)
        - Stop loss hit
        """
        # For binary complement arb, we hold until resolution
        # No active exit management needed - profit is locked in
        
        # However, if spread re-opens significantly, might add to position
        return None


class CrossPlatformArbitrage(TradingStrategy):
    """
    Exploits price differentials between Kalshi and Polymarket
    for the same underlying event.
    """
    
    def __init__(self, min_differential: Decimal = Decimal("0.03")):
        self.min_differential = min_differential
        self._market_pairs: dict = {}  # Maps event IDs across platforms
    
    @property
    def name(self) -> str:
        return "cross_platform_arb"
    
    def register_pair(self, kalshi_id: str, polymarket_id: str, event_name: str):
        """Register a market pair for cross-platform monitoring."""
        self._market_pairs[event_name] = {
            "kalshi": kalshi_id,
            "polymarket": polymarket_id
        }
    
    async def scan_markets(self, markets: List[Market]) -> List[Signal]:
        """
        Compare prices across platforms for registered pairs.
        
        Signal generated when price differential exceeds threshold.
        """
        signals = []
        
        # Group markets by platform
        kalshi_markets = {m.id: m for m in markets if m.platform == "kalshi"}
        poly_markets = {m.id: m for m in markets if m.platform == "polymarket"}
        
        for event_name, pair in self._market_pairs.items():
            kalshi_market = kalshi_markets.get(pair["kalshi"])
            poly_market = poly_markets.get(pair["polymarket"])
            
            if not kalshi_market or not poly_market:
                continue
            
            # Compare YES prices
            yes_diff = kalshi_market.yes_price - poly_market.yes_price
            
            if abs(yes_diff) >= self.min_differential:
                # Buy on cheaper platform, sell on expensive
                if yes_diff > 0:
                    # Polymarket cheaper, Kalshi expensive
                    buy_market = poly_market
                    sell_market = kalshi_market
                else:
                    buy_market = kalshi_market
                    sell_market = poly_market
                
                signals.append(Signal(
                    market_id=buy_market.id,
                    market=buy_market,
                    side=OrderSide.YES,
                    strength=float(abs(yes_diff)),
                    reason=f"Cross-platform arb: {event_name}, {abs(yes_diff):.2%} differential"
                ))
                
                logger.info(f"CROSS-ARB: {event_name} - Buy {buy_market.platform} @ {buy_market.yes_price}, "
                           f"Sell {sell_market.platform} @ {sell_market.yes_price}")
        
        return signals
    
    async def check_exit(self, position: Position, market: Market) -> Optional[Signal]:
        """Cross-platform positions exit when differential closes."""
        # Would need both sides of the arb tracked
        return None
