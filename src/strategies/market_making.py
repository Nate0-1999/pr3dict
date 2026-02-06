"""
PR3DICT: Market Making Strategy

Provides liquidity to prediction markets by continuously quoting bid/ask prices
and profiting from the spread, while managing inventory risk.

Key Differences from Traditional Market Making:
1. Binary outcome constraint: YES + NO must sum to ≤ $1.00
2. Event resolution: All positions resolve to $0 or $1 at close
3. Time decay: Value changes accelerate near resolution
4. Inventory skew: Holding one side creates directional exposure
5. No short selling: Can only hold YES or NO, not negative positions

Strategy Components:
- Dynamic spread calculation based on volatility, liquidity, time-to-close
- Inventory management with skew-based pricing adjustments
- Adverse selection protection (don't get picked off on news)
- Risk limits to avoid getting stuck on wrong side near resolution
"""
from typing import List, Optional, Dict, Tuple
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import logging
from collections import defaultdict

from .base import TradingStrategy, Signal
from ..platforms.base import Market, Position, OrderSide, Order, OrderBook

logger = logging.getLogger(__name__)


@dataclass
class MarketMakingConfig:
    """Configuration for market making strategy."""
    
    # === Spread Parameters ===
    min_spread: Decimal = Decimal("0.02")  # 2% minimum spread
    base_spread: Decimal = Decimal("0.04")  # 4% base spread in normal conditions
    max_spread: Decimal = Decimal("0.12")  # 12% max spread (wide but still competitive)
    
    # Spread adjustments based on market conditions
    volatility_spread_multiplier: float = 1.5  # Widen spread in volatile markets
    low_liquidity_spread_add: Decimal = Decimal("0.02")  # Add 2% for illiquid markets
    
    # === Inventory Management ===
    max_inventory: int = 50  # Max contracts of a single side per market
    target_inventory: int = 0  # Ideal inventory (flat)
    inventory_skew_threshold: int = 10  # Start adjusting quotes when skew > this
    
    # Inventory skew pricing: shift quotes to encourage rebalancing
    # For every 10 contracts of skew, shift prices by 1%
    skew_price_adjustment_per_10: Decimal = Decimal("0.01")
    
    # === Risk Management ===
    max_markets: int = 10  # Max markets to make simultaneously
    min_liquidity: Decimal = Decimal("5000")  # Only make markets with sufficient liquidity
    max_time_to_resolution_hours: int = 24 * 7  # 7 days max
    min_time_to_resolution_hours: float = 0.5  # 30 min minimum (avoid last-minute risk)
    
    # Quote size
    quote_size: int = 10  # Number of contracts to quote per side
    max_position_size: int = 100  # Max total position per market
    
    # === Adverse Selection Protection ===
    max_quote_age_seconds: int = 30  # Refresh quotes at least every 30s
    price_change_requote_threshold: Decimal = Decimal("0.02")  # Requote if price moves 2%
    volume_spike_threshold: float = 3.0  # Pause if volume 3x normal (news event)
    
    # === Rebalancing ===
    rebalance_threshold: int = 20  # Actively rebalance when skew > this
    rebalance_urgency_hours: int = 24  # More aggressive rebalancing when close to resolution


@dataclass
class InventoryTracker:
    """Track inventory position per market."""
    market_id: str
    yes_contracts: int = 0
    no_contracts: int = 0
    avg_yes_price: Decimal = Decimal("0")
    avg_no_price: Decimal = Decimal("0")
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def net_position(self) -> int:
        """Net directional exposure (positive = long YES, negative = long NO)."""
        return self.yes_contracts - self.no_contracts
    
    @property
    def gross_position(self) -> int:
        """Total contracts held."""
        return self.yes_contracts + self.no_contracts
    
    @property
    def is_flat(self) -> bool:
        """True if no net directional exposure."""
        return self.net_position == 0
    
    @property
    def skew_ratio(self) -> float:
        """Inventory skew as ratio: +1 = fully YES, -1 = fully NO, 0 = balanced."""
        if self.gross_position == 0:
            return 0.0
        return float(self.net_position) / float(self.gross_position)


class MarketMakingStrategy(TradingStrategy):
    """
    Automated market making for prediction markets.
    
    Core Logic:
    1. Calculate fair value for each market (can use mid-price, models, or external data)
    2. Set bid/ask spreads around fair value based on:
       - Market volatility
       - Liquidity depth
       - Time to resolution
       - Current inventory skew
    3. Post limit orders on both sides
    4. Monitor fills and adjust inventory
    5. Adjust quotes dynamically to manage inventory risk
    6. Exit positions before resolution when skewed
    
    Revenue Sources:
    - Bid-ask spread capture (primary)
    - Mean reversion (markets often oscillate)
    - Liquidity provider rebates (if platform offers)
    
    Risks:
    - Adverse selection: Getting picked off when news breaks
    - Inventory risk: Holding wrong side when market resolves
    - Quote competition: Other MMs narrowing spreads
    """
    
    def __init__(self, config: MarketMakingConfig = None):
        self.config = config or MarketMakingConfig()
        
        # Track inventory per market
        self.inventory: Dict[str, InventoryTracker] = {}
        
        # Track active quotes per market
        self.active_quotes: Dict[str, List[Order]] = defaultdict(list)
        
        # Market statistics for spread adjustment
        self.market_stats: Dict[str, dict] = {}
        
        # Last quote times for staleness detection
        self.last_quote_time: Dict[str, datetime] = {}
    
    @property
    def name(self) -> str:
        return "market_making"
    
    # ============================================================================
    # Main Strategy Interface
    # ============================================================================
    
    async def scan_markets(self, markets: List[Market]) -> List[Signal]:
        """
        Generate market making signals.
        
        For each qualified market:
        1. Check if we should be making it
        2. Calculate fair value and spread
        3. Adjust for inventory skew
        4. Generate bid/ask signals
        """
        signals = []
        
        # Filter markets suitable for market making
        qualified_markets = self._filter_markets(markets)
        
        # Limit to max markets
        if len(qualified_markets) > self.config.max_markets:
            # Prioritize by liquidity
            qualified_markets = sorted(
                qualified_markets, 
                key=lambda m: m.liquidity, 
                reverse=True
            )[:self.config.max_markets]
        
        for market in qualified_markets:
            # Get or initialize inventory tracker
            if market.id not in self.inventory:
                self.inventory[market.id] = InventoryTracker(market_id=market.id)
            
            inventory = self.inventory[market.id]
            
            # Calculate fair value and spreads
            fair_value = self._calculate_fair_value(market)
            spread = self._calculate_dynamic_spread(market, inventory)
            
            # Adjust quotes for inventory skew
            bid_adjustment, ask_adjustment = self._calculate_inventory_adjustments(inventory)
            
            # Generate bid/ask prices
            bid_price = fair_value - (spread / 2) + bid_adjustment
            ask_price = fair_value + (spread / 2) + ask_adjustment
            
            # Ensure prices stay within valid range [0.01, 0.99]
            bid_price = max(Decimal("0.01"), min(bid_price, Decimal("0.99")))
            ask_price = max(Decimal("0.01"), min(ask_price, Decimal("0.99")))
            
            # Ensure bid < ask
            if bid_price >= ask_price:
                mid = (bid_price + ask_price) / 2
                bid_price = mid - Decimal("0.01")
                ask_price = mid + Decimal("0.01")
            
            # Check if we should quote both sides or rebalance
            should_quote_bid = self._should_quote_side(market, OrderSide.YES, inventory, "bid")
            should_quote_ask = self._should_quote_side(market, OrderSide.NO, inventory, "ask")
            
            # Generate signals for bid (buy YES at bid_price)
            if should_quote_bid:
                signals.append(Signal(
                    market_id=market.id,
                    market=market,
                    side=OrderSide.YES,
                    strength=0.5,  # Market making is not directional
                    reason=f"MM bid @ {bid_price:.3f} (spread={spread:.2%}, skew={inventory.net_position})",
                    target_price=bid_price
                ))
            
            # Generate signals for ask (buy NO at (1 - ask_price))
            # In prediction markets: buying NO at price P is equivalent to selling YES at P
            # So to "offer" YES at ask_price, we buy NO at (1 - ask_price)
            if should_quote_ask:
                no_price = Decimal("1.0") - ask_price
                signals.append(Signal(
                    market_id=market.id,
                    market=market,
                    side=OrderSide.NO,
                    strength=0.5,
                    reason=f"MM ask @ {ask_price:.3f} (NO @ {no_price:.3f}, skew={inventory.net_position})",
                    target_price=no_price
                ))
            
            # Update last quote time
            self.last_quote_time[market.id] = datetime.now(timezone.utc)
        
        if signals:
            logger.info(f"Market making: Generated {len(signals)} quote signals across {len(qualified_markets)} markets")
        
        return signals
    
    async def check_exit(self, position: Position, market: Market) -> Optional[Signal]:
        """
        Check if market making position should be exited.
        
        Exit conditions:
        1. Approaching resolution with inventory skew (reduce risk)
        2. Market conditions deteriorated (low liquidity, high volatility)
        3. Inventory exceeded max limits (forced rebalancing)
        """
        if market.id not in self.inventory:
            return None
        
        inventory = self.inventory[market.id]
        
        # Time to resolution check
        time_to_close = (market.close_time - datetime.now(market.close_time.tzinfo)).total_seconds() / 3600
        
        # Exit if too close to resolution with skewed inventory
        if time_to_close < self.config.min_time_to_resolution_hours:
            if abs(inventory.net_position) > 5:  # Any skew near resolution is risky
                exit_side = OrderSide.NO if inventory.net_position > 0 else OrderSide.YES
                return Signal(
                    market_id=market.id,
                    market=market,
                    side=exit_side,
                    strength=1.0,  # High urgency
                    reason=f"MM exit: {time_to_close:.1f}h to resolution, skew={inventory.net_position}",
                    target_price=market.no_price if exit_side == OrderSide.NO else market.yes_price
                )
        
        # Exit if inventory exceeds limits
        if abs(inventory.net_position) > self.config.max_inventory:
            exit_side = OrderSide.NO if inventory.net_position > 0 else OrderSide.YES
            return Signal(
                market_id=market.id,
                market=market,
                side=exit_side,
                strength=0.9,
                reason=f"MM exit: Inventory limit exceeded ({inventory.net_position})",
                target_price=market.no_price if exit_side == OrderSide.NO else market.yes_price
            )
        
        # Exit if market became unsuitable for MM
        if not self._is_market_suitable(market):
            # Close position at market
            if inventory.net_position > 0:
                exit_side = OrderSide.NO
                exit_price = market.no_price
            elif inventory.net_position < 0:
                exit_side = OrderSide.YES
                exit_price = market.yes_price
            else:
                return None  # Already flat
            
            return Signal(
                market_id=market.id,
                market=market,
                side=exit_side,
                strength=0.7,
                reason=f"MM exit: Market no longer suitable for MM",
                target_price=exit_price
            )
        
        return None
    
    # ============================================================================
    # Market Filtering & Selection
    # ============================================================================
    
    def _filter_markets(self, markets: List[Market]) -> List[Market]:
        """Filter markets suitable for market making."""
        qualified = []
        
        for market in markets:
            if self._is_market_suitable(market):
                qualified.append(market)
        
        return qualified
    
    def _is_market_suitable(self, market: Market) -> bool:
        """Check if market meets MM criteria."""
        
        # Skip resolved markets
        if market.resolved:
            return False
        
        # Liquidity check
        if market.liquidity < self.config.min_liquidity:
            return False
        
        # Time to resolution check
        time_to_close = (market.close_time - datetime.now(market.close_time.tzinfo)).total_seconds() / 3600
        
        if time_to_close < self.config.min_time_to_resolution_hours:
            return False  # Too close to resolution
        
        if time_to_close > self.config.max_time_to_resolution_hours:
            return False  # Too far out (less predictable)
        
        # Existing spread check - only make markets with reasonable spreads
        # If spread is already too tight, we can't profitably MM
        if market.spread < self.config.min_spread:
            return False
        
        return True
    
    # ============================================================================
    # Pricing & Spread Calculation
    # ============================================================================
    
    def _calculate_fair_value(self, market: Market) -> Decimal:
        """
        Calculate fair value for the market.
        
        Simple approach: Use mid-price between best bid/ask.
        Advanced: Could incorporate:
        - External probability models
        - Order flow analysis
        - Historical mean reversion
        - News sentiment
        """
        # For now, use mid-price as fair value
        # mid = (best_bid + best_ask) / 2
        # In binary markets: best_bid_yes + best_ask_no should ≈ 1.00
        
        # Simple mid-price
        mid_price = (market.yes_price + market.no_price) / 2
        
        # Ensure fair value is in valid range
        return max(Decimal("0.01"), min(mid_price, Decimal("0.99")))
    
    def _calculate_dynamic_spread(self, market: Market, inventory: InventoryTracker) -> Decimal:
        """
        Calculate bid-ask spread dynamically based on market conditions.
        
        Wider spreads when:
        - High volatility
        - Low liquidity
        - Close to resolution (higher risk)
        - High inventory (discourage more of same side)
        """
        spread = self.config.base_spread
        
        # Adjust for liquidity
        if market.liquidity < self.config.min_liquidity * 2:
            spread += self.config.low_liquidity_spread_add
        
        # Adjust for time to resolution
        time_to_close_hours = (market.close_time - datetime.now(market.close_time.tzinfo)).total_seconds() / 3600
        
        # Widen spread as we approach resolution
        if time_to_close_hours < 24:
            time_multiplier = Decimal("1.5")
            spread = spread * time_multiplier
        elif time_to_close_hours < 6:
            time_multiplier = Decimal("2.0")
            spread = spread * time_multiplier
        
        # Adjust for inventory risk
        # Wider spread when we have large inventory (more risk)
        if abs(inventory.net_position) > self.config.inventory_skew_threshold:
            inventory_multiplier = Decimal("1.3")
            spread = spread * inventory_multiplier
        
        # Clamp to min/max
        spread = max(self.config.min_spread, min(spread, self.config.max_spread))
        
        return spread
    
    def _calculate_inventory_adjustments(self, inventory: InventoryTracker) -> Tuple[Decimal, Decimal]:
        """
        Calculate price adjustments to quotes based on inventory skew.
        
        Goal: Encourage trading that reduces skew.
        
        If we're long YES (net_position > 0):
        - Lower our bids (less eager to buy more YES)
        - Raise our asks (more eager to sell YES / buy NO)
        
        Returns:
            (bid_adjustment, ask_adjustment) where positive = higher price
        """
        net_pos = inventory.net_position
        
        if abs(net_pos) <= self.config.inventory_skew_threshold:
            # Within acceptable skew, no adjustment
            return Decimal("0"), Decimal("0")
        
        # Calculate skew magnitude
        skew_magnitude = abs(net_pos)
        num_increments = skew_magnitude // 10  # Adjust per 10 contracts
        
        adjustment = self.config.skew_price_adjustment_per_10 * num_increments
        
        if net_pos > 0:
            # Long YES: discourage buying YES (lower bid), encourage buying NO (effectively lower ask)
            bid_adjustment = -adjustment  # Lower bid
            ask_adjustment = -adjustment  # Lower ask (cheaper NO)
        else:
            # Long NO (short YES): encourage buying YES, discourage buying NO
            bid_adjustment = adjustment  # Raise bid
            ask_adjustment = adjustment  # Raise ask
        
        return bid_adjustment, ask_adjustment
    
    def _should_quote_side(self, 
                           market: Market, 
                           side: OrderSide, 
                           inventory: InventoryTracker,
                           quote_type: str) -> bool:
        """
        Determine if we should post a quote on this side.
        
        Args:
            side: YES or NO
            quote_type: "bid" or "ask"
        """
        # Check position limits
        if inventory.gross_position >= self.config.max_position_size:
            # At max position, only quote to reduce inventory
            if quote_type == "bid" and side == OrderSide.YES:
                return inventory.net_position < 0  # Only bid if we're short
            elif quote_type == "ask" and side == OrderSide.NO:
                return inventory.net_position > 0  # Only offer (via NO) if we're long
        
        # Check inventory skew
        if abs(inventory.net_position) > self.config.rebalance_threshold:
            # Heavy skew - only quote on rebalancing side
            if inventory.net_position > 0:
                # Long YES, want to buy NO (reduce YES exposure)
                return side == OrderSide.NO
            else:
                # Short YES (long NO), want to buy YES
                return side == OrderSide.YES
        
        # Normal conditions: quote both sides
        return True
    
    # ============================================================================
    # Inventory Management
    # ============================================================================
    
    def update_inventory(self, market_id: str, side: OrderSide, quantity: int, price: Decimal):
        """
        Update inventory after a fill.
        
        Call this when orders are filled.
        """
        if market_id not in self.inventory:
            self.inventory[market_id] = InventoryTracker(market_id=market_id)
        
        inv = self.inventory[market_id]
        
        if side == OrderSide.YES:
            # Update YES position
            old_total_cost = inv.avg_yes_price * inv.yes_contracts
            new_total_cost = old_total_cost + (price * quantity)
            inv.yes_contracts += quantity
            if inv.yes_contracts > 0:
                inv.avg_yes_price = new_total_cost / inv.yes_contracts
        else:
            # Update NO position
            old_total_cost = inv.avg_no_price * inv.no_contracts
            new_total_cost = old_total_cost + (price * quantity)
            inv.no_contracts += quantity
            if inv.no_contracts > 0:
                inv.avg_no_price = new_total_cost / inv.no_contracts
        
        inv.last_update = datetime.now(timezone.utc)
        
        logger.info(f"Inventory updated for {market_id}: YES={inv.yes_contracts}, NO={inv.no_contracts}, net={inv.net_position}")
    
    def get_inventory_status(self) -> Dict[str, dict]:
        """Get current inventory status across all markets."""
        status = {}
        for market_id, inv in self.inventory.items():
            status[market_id] = {
                "yes_contracts": inv.yes_contracts,
                "no_contracts": inv.no_contracts,
                "net_position": inv.net_position,
                "gross_position": inv.gross_position,
                "skew_ratio": inv.skew_ratio,
                "avg_yes_price": float(inv.avg_yes_price),
                "avg_no_price": float(inv.avg_no_price),
            }
        return status
    
    # ============================================================================
    # Strategy Configuration
    # ============================================================================
    
    def get_position_size(self, 
                          signal: Signal, 
                          account_balance: Decimal,
                          risk_pct: float = 0.02) -> int:
        """
        Override position sizing for market making.
        
        Market making uses fixed quote sizes rather than risk-based sizing.
        """
        # For market making, use configured quote size
        return self.config.quote_size
