"""
PR3DICT: VWAP (Volume-Weighted Average Price) Calculator and Validator

Critical component to prevent "quoted price ≠ execution price" losses.
Analyzes order book depth, estimates slippage, and validates execution quality.

Key Features:
- Calculate VWAP from order book depth
- Account for slippage at different volumes
- Compare quoted vs expected execution price
- Alert when deviation exceeds threshold
- Historical VWAP analysis via Alchemy Polygon RPC
- Real-time liquidity monitoring
- Price impact curves per market
"""

import asyncio
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import json
import os

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logging.warning("httpx not installed. Historical VWAP analysis unavailable.")

logger = logging.getLogger(__name__)


@dataclass
class VWAPResult:
    """Result of VWAP calculation for a given order size."""
    market_id: str
    side: str  # "buy" or "sell"
    target_quantity: int
    quoted_price: Decimal  # Mid price or best bid/ask
    vwap_price: Decimal  # Actual execution price
    total_cost: Decimal
    slippage_pct: Decimal
    slippage_absolute: Decimal
    price_impact_pct: Decimal
    fills: List[Tuple[Decimal, int]]  # (price, quantity) pairs
    depth_used: int  # Number of order book levels consumed
    liquidity_sufficient: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def execution_quality(self) -> str:
        """Categorize execution quality based on slippage."""
        if not self.liquidity_sufficient:
            return "INSUFFICIENT_LIQUIDITY"
        elif self.slippage_pct > Decimal("5.0"):
            return "POOR"
        elif self.slippage_pct > Decimal("2.0"):
            return "FAIR"
        elif self.slippage_pct > Decimal("0.5"):
            return "GOOD"
        else:
            return "EXCELLENT"
    
    def to_dict(self) -> dict:
        """Serialize to dict for logging/storage."""
        return {
            "market_id": self.market_id,
            "side": self.side,
            "target_quantity": self.target_quantity,
            "quoted_price": str(self.quoted_price),
            "vwap_price": str(self.vwap_price),
            "total_cost": str(self.total_cost),
            "slippage_pct": str(self.slippage_pct),
            "slippage_absolute": str(self.slippage_absolute),
            "price_impact_pct": str(self.price_impact_pct),
            "depth_used": self.depth_used,
            "liquidity_sufficient": self.liquidity_sufficient,
            "execution_quality": self.execution_quality,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class LiquidityMetrics:
    """Metrics describing market liquidity depth."""
    market_id: str
    bid_depth: int  # Total contracts on bid side
    ask_depth: int
    bid_value: Decimal  # Total USDC value
    ask_value: Decimal
    spread_bps: int  # Spread in basis points
    top_of_book_size: int  # Size at best bid/ask
    depth_imbalance: Decimal  # bid_depth / (bid_depth + ask_depth)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_healthy(self) -> bool:
        """Check if liquidity is sufficient for trading."""
        return (
            self.bid_depth > 100 and
            self.ask_depth > 100 and
            self.spread_bps < 500  # < 5%
        )


@dataclass
class PriceImpactCurve:
    """Models how price degrades with increasing order size."""
    market_id: str
    side: str
    data_points: List[Tuple[int, Decimal]]  # (quantity, avg_price)
    fitted_params: Optional[Dict[str, float]] = None
    
    def estimate_impact(self, quantity: int) -> Decimal:
        """Estimate price impact for a given quantity."""
        if not self.data_points:
            return Decimal("0")
        
        # Simple linear interpolation
        for i, (qty, price) in enumerate(self.data_points):
            if qty >= quantity:
                if i == 0:
                    return price
                prev_qty, prev_price = self.data_points[i-1]
                # Linear interpolation
                slope = (price - prev_price) / (qty - prev_qty)
                return prev_price + slope * (quantity - prev_qty)
        
        # Extrapolate from last two points
        if len(self.data_points) >= 2:
            q1, p1 = self.data_points[-2]
            q2, p2 = self.data_points[-1]
            slope = (p2 - p1) / (q2 - q1)
            return p2 + slope * (quantity - q2)
        
        return self.data_points[-1][1]


class VWAPCalculator:
    """
    Core VWAP calculation engine.
    
    Calculates volume-weighted average price from order book data,
    accounting for real market depth and slippage.
    """
    
    def __init__(self, slippage_warning_threshold: Decimal = Decimal("2.0")):
        """
        Args:
            slippage_warning_threshold: Warn if slippage exceeds this % (default 2%)
        """
        self.slippage_threshold = slippage_warning_threshold
    
    def calculate_vwap(self,
                       orders: List[Tuple[Decimal, int]],
                       quantity: int,
                       side: str,
                       market_id: str,
                       quoted_price: Optional[Decimal] = None) -> VWAPResult:
        """
        Calculate VWAP for executing an order of given size.
        
        Args:
            orders: Order book levels as (price, size) tuples
            quantity: Target order size (number of contracts)
            side: "buy" or "sell"
            market_id: Market identifier
            quoted_price: Reference price for slippage calculation (default: best price)
        
        Returns:
            VWAPResult with execution details
        """
        if not orders:
            return self._insufficient_liquidity(market_id, side, quantity)
        
        # Sort orders appropriately
        if side == "buy":
            # For buying, we take asks (ascending price)
            sorted_orders = sorted(orders, key=lambda x: x[0])
        else:
            # For selling, we take bids (descending price)
            sorted_orders = sorted(orders, key=lambda x: x[0], reverse=True)
        
        # Use best price as quoted if not provided
        if quoted_price is None:
            quoted_price = sorted_orders[0][0]
        
        # Execute against order book
        fills = []
        remaining = quantity
        total_cost = Decimal("0")
        depth_levels = 0
        
        for price, size in sorted_orders:
            if remaining <= 0:
                break
            
            fill_qty = min(remaining, size)
            fills.append((price, fill_qty))
            total_cost += price * fill_qty
            remaining -= fill_qty
            depth_levels += 1
        
        # Check if we had sufficient liquidity
        liquidity_sufficient = remaining == 0
        
        if not liquidity_sufficient:
            logger.warning(
                f"Insufficient liquidity for {market_id} {side} {quantity} contracts. "
                f"Only {quantity - remaining} available."
            )
            return self._insufficient_liquidity(market_id, side, quantity, fills, total_cost)
        
        # Calculate VWAP
        vwap_price = total_cost / quantity
        
        # Calculate slippage
        slippage_absolute = abs(vwap_price - quoted_price)
        slippage_pct = (slippage_absolute / quoted_price * 100) if quoted_price > 0 else Decimal("0")
        
        # Calculate price impact (vs mid or quoted)
        price_impact_pct = slippage_pct  # Simplified, could use mid price
        
        result = VWAPResult(
            market_id=market_id,
            side=side,
            target_quantity=quantity,
            quoted_price=quoted_price,
            vwap_price=vwap_price,
            total_cost=total_cost,
            slippage_pct=slippage_pct,
            slippage_absolute=slippage_absolute,
            price_impact_pct=price_impact_pct,
            fills=fills,
            depth_used=depth_levels,
            liquidity_sufficient=True
        )
        
        # Warn if slippage exceeds threshold
        if slippage_pct > self.slippage_threshold:
            logger.warning(
                f"High slippage detected: {market_id} {side} "
                f"{slippage_pct:.2f}% (threshold: {self.slippage_threshold}%)"
            )
        
        return result
    
    def _insufficient_liquidity(self,
                                market_id: str,
                                side: str,
                                quantity: int,
                                fills: List[Tuple[Decimal, int]] = None,
                                total_cost: Decimal = None) -> VWAPResult:
        """Create result for insufficient liquidity scenario."""
        fills = fills or []
        total_cost = total_cost or Decimal("0")
        filled_qty = sum(f[1] for f in fills)
        
        vwap = total_cost / filled_qty if filled_qty > 0 else Decimal("0")
        
        return VWAPResult(
            market_id=market_id,
            side=side,
            target_quantity=quantity,
            quoted_price=Decimal("0"),
            vwap_price=vwap,
            total_cost=total_cost,
            slippage_pct=Decimal("100"),  # Indicate failure
            slippage_absolute=Decimal("0"),
            price_impact_pct=Decimal("100"),
            fills=fills,
            depth_used=len(fills),
            liquidity_sufficient=False
        )
    
    def calculate_liquidity_metrics(self,
                                    bids: List[Tuple[Decimal, int]],
                                    asks: List[Tuple[Decimal, int]],
                                    market_id: str) -> LiquidityMetrics:
        """
        Calculate comprehensive liquidity metrics from order book.
        
        Args:
            bids: Bid side as (price, size) tuples
            asks: Ask side as (price, size) tuples
            market_id: Market identifier
        
        Returns:
            LiquidityMetrics object
        """
        # Calculate depths
        bid_depth = sum(size for _, size in bids)
        ask_depth = sum(size for _, size in asks)
        
        # Calculate values
        bid_value = sum(Decimal(str(price)) * size for price, size in bids)
        ask_value = sum(Decimal(str(price)) * size for price, size in asks)
        
        # Spread calculation
        if bids and asks:
            best_bid = max(price for price, _ in bids)
            best_ask = min(price for price, _ in asks)
            spread = best_ask - best_bid
            spread_bps = int((spread / best_ask * 10000)) if best_ask > 0 else 9999
            
            # Top of book size
            top_bid_size = max((size for price, size in bids if price == best_bid), default=0)
            top_ask_size = max((size for price, size in asks if price == best_ask), default=0)
            top_of_book_size = min(top_bid_size, top_ask_size)
        else:
            spread_bps = 9999
            top_of_book_size = 0
        
        # Depth imbalance
        total_depth = bid_depth + ask_depth
        depth_imbalance = Decimal(bid_depth) / total_depth if total_depth > 0 else Decimal("0.5")
        
        return LiquidityMetrics(
            market_id=market_id,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            bid_value=bid_value,
            ask_value=ask_value,
            spread_bps=spread_bps,
            top_of_book_size=top_of_book_size,
            depth_imbalance=depth_imbalance
        )
    
    def build_price_impact_curve(self,
                                 orders: List[Tuple[Decimal, int]],
                                 side: str,
                                 market_id: str,
                                 sample_sizes: Optional[List[int]] = None) -> PriceImpactCurve:
        """
        Build a price impact curve by sampling different order sizes.
        
        Args:
            orders: Order book levels
            side: "buy" or "sell"
            market_id: Market identifier
            sample_sizes: Order sizes to sample (default: logarithmic scale)
        
        Returns:
            PriceImpactCurve model
        """
        if sample_sizes is None:
            # Default: logarithmic scale from 10 to max available
            max_liquidity = sum(size for _, size in orders)
            sample_sizes = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
            sample_sizes = [s for s in sample_sizes if s <= max_liquidity]
            if max_liquidity > 10000:
                sample_sizes.append(max_liquidity)
        
        data_points = []
        for qty in sample_sizes:
            result = self.calculate_vwap(orders, qty, side, market_id)
            if result.liquidity_sufficient:
                data_points.append((qty, result.vwap_price))
        
        return PriceImpactCurve(
            market_id=market_id,
            side=side,
            data_points=data_points
        )


class HistoricalVWAPAnalyzer:
    """
    Analyze historical trade data to learn typical slippage patterns.
    Uses Alchemy Polygon RPC to fetch past Polymarket trades.
    """
    
    def __init__(self, alchemy_api_key: Optional[str] = None):
        """
        Args:
            alchemy_api_key: Alchemy API key for Polygon access
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for historical analysis. pip install httpx")
        
        self.api_key = alchemy_api_key or os.getenv("ALCHEMY_API_KEY")
        if not self.api_key:
            logger.warning("No Alchemy API key provided. Historical analysis unavailable.")
        
        self.rpc_url = f"https://polygon-mainnet.g.alchemy.com/v2/{self.api_key}"
        
        # Cache for historical data
        self.price_impact_cache: Dict[str, PriceImpactCurve] = {}
        self.slippage_patterns: Dict[str, List[Decimal]] = defaultdict(list)
    
    async def fetch_historical_trades(self,
                                      market_id: str,
                                      from_block: Optional[int] = None,
                                      to_block: Optional[int] = None,
                                      limit: int = 1000) -> List[Dict]:
        """
        Fetch historical trades for a market from Polygon chain.
        
        Args:
            market_id: Market/token ID
            from_block: Starting block number
            to_block: Ending block number
            limit: Maximum number of trades to fetch
        
        Returns:
            List of trade events
        """
        if not self.api_key:
            logger.error("Cannot fetch historical trades without Alchemy API key")
            return []
        
        # TODO: Implement actual Polygon event log parsing
        # This would query the Polymarket CLOB contract for TokenSwap events
        # For now, return empty list as placeholder
        logger.info(f"Fetching historical trades for {market_id} (placeholder)")
        return []
    
    async def analyze_price_impact(self,
                                   market_id: str,
                                   lookback_days: int = 7) -> Optional[PriceImpactCurve]:
        """
        Analyze historical price impact patterns.
        
        Args:
            market_id: Market to analyze
            lookback_days: Number of days to analyze
        
        Returns:
            PriceImpactCurve if sufficient data, else None
        """
        # Check cache first
        if market_id in self.price_impact_cache:
            return self.price_impact_cache[market_id]
        
        trades = await self.fetch_historical_trades(market_id)
        
        if not trades:
            logger.warning(f"No historical data for {market_id}")
            return None
        
        # TODO: Build actual price impact curve from trade data
        # For now, return placeholder
        curve = PriceImpactCurve(
            market_id=market_id,
            side="buy",
            data_points=[]
        )
        
        self.price_impact_cache[market_id] = curve
        return curve
    
    async def detect_low_liquidity_traps(self,
                                        market_ids: List[str],
                                        threshold_volume: Decimal = Decimal("1000")) -> List[str]:
        """
        Detect markets with dangerously low liquidity.
        
        Args:
            market_ids: Markets to check
            threshold_volume: Minimum 24h volume (USDC)
        
        Returns:
            List of market IDs with low liquidity
        """
        low_liquidity_markets = []
        
        for market_id in market_ids:
            trades = await self.fetch_historical_trades(market_id)
            
            # Calculate 24h volume
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=1)
            recent_volume = Decimal("0")
            
            for trade in trades:
                # TODO: Parse actual trade timestamp and volume
                # Placeholder logic
                pass
            
            if recent_volume < threshold_volume:
                low_liquidity_markets.append(market_id)
                logger.warning(f"Low liquidity detected: {market_id} (24h vol: ${recent_volume})")
        
        return low_liquidity_markets


class VWAPValidator:
    """
    Real-time VWAP validation for trade execution.
    Gates trading signals based on liquidity and slippage checks.
    """
    
    def __init__(self,
                 calculator: VWAPCalculator,
                 max_slippage_pct: Decimal = Decimal("3.0"),
                 min_liquidity_contracts: int = 500,
                 max_spread_bps: int = 300):
        """
        Args:
            calculator: VWAPCalculator instance
            max_slippage_pct: Maximum acceptable slippage %
            min_liquidity_contracts: Minimum order book depth
            max_spread_bps: Maximum bid-ask spread in basis points
        """
        self.calculator = calculator
        self.max_slippage_pct = max_slippage_pct
        self.min_liquidity = min_liquidity_contracts
        self.max_spread = max_spread_bps
    
    def validate_execution(self,
                          vwap_result: VWAPResult,
                          liquidity_metrics: LiquidityMetrics) -> Tuple[bool, str]:
        """
        Validate if execution meets quality standards.
        
        Args:
            vwap_result: VWAP calculation result
            liquidity_metrics: Market liquidity metrics
        
        Returns:
            (is_valid, reason) tuple
        """
        # Check liquidity sufficiency
        if not vwap_result.liquidity_sufficient:
            return False, f"Insufficient liquidity: only {sum(f[1] for f in vwap_result.fills)} contracts available"
        
        # Check slippage threshold
        if vwap_result.slippage_pct > self.max_slippage_pct:
            return False, f"Slippage {vwap_result.slippage_pct:.2f}% exceeds max {self.max_slippage_pct}%"
        
        # Check minimum liquidity depth
        side_depth = liquidity_metrics.bid_depth if vwap_result.side == "sell" else liquidity_metrics.ask_depth
        if side_depth < self.min_liquidity:
            return False, f"Depth {side_depth} below minimum {self.min_liquidity}"
        
        # Check spread
        if liquidity_metrics.spread_bps > self.max_spread:
            return False, f"Spread {liquidity_metrics.spread_bps}bps exceeds max {self.max_spread}bps"
        
        return True, "OK"
    
    def suggest_order_split(self,
                           vwap_result: VWAPResult,
                           max_chunks: int = 5) -> List[int]:
        """
        Suggest optimal order size splits to minimize impact.
        
        Args:
            vwap_result: Initial VWAP calculation for full size
            max_chunks: Maximum number of sub-orders
        
        Returns:
            List of suggested order sizes
        """
        total_qty = vwap_result.target_quantity
        
        # If execution quality is excellent, no need to split
        if vwap_result.execution_quality == "EXCELLENT":
            return [total_qty]
        
        # Calculate optimal split based on price impact curve
        # Simple heuristic: split into chunks at liquidity boundaries
        depths = []
        cumulative = 0
        for price, qty in vwap_result.fills:
            cumulative += qty
            depths.append(cumulative)
        
        # Find natural breakpoints (where price jumps significantly)
        if len(vwap_result.fills) < 2:
            return [total_qty]
        
        breakpoints = [0]
        for i in range(1, len(vwap_result.fills)):
            prev_price = vwap_result.fills[i-1][0]
            curr_price = vwap_result.fills[i][0]
            price_jump = abs(curr_price - prev_price) / prev_price
            
            if price_jump > Decimal("0.01"):  # 1% jump
                breakpoints.append(depths[i-1])
        
        breakpoints.append(total_qty)
        
        # Convert breakpoints to chunk sizes
        chunks = []
        for i in range(1, len(breakpoints)):
            chunk_size = breakpoints[i] - breakpoints[i-1]
            if chunk_size > 0:
                chunks.append(chunk_size)
        
        # Limit to max_chunks by combining smallest
        while len(chunks) > max_chunks:
            chunks.sort()
            chunks[0] += chunks[1]
            chunks.pop(1)
        
        return chunks


class VWAPMonitor:
    """
    Real-time VWAP monitoring and alerting.
    Tracks execution quality and liquidity health.
    """
    
    def __init__(self, calculator: VWAPCalculator):
        self.calculator = calculator
        self.execution_history: List[VWAPResult] = []
        self.liquidity_snapshots: Dict[str, List[LiquidityMetrics]] = defaultdict(list)
    
    def record_execution(self, vwap_result: VWAPResult):
        """Record execution for historical analysis."""
        self.execution_history.append(vwap_result)
        
        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def record_liquidity_snapshot(self, metrics: LiquidityMetrics):
        """Record liquidity snapshot for market."""
        self.liquidity_snapshots[metrics.market_id].append(metrics)
        
        # Keep only last 100 snapshots per market
        if len(self.liquidity_snapshots[metrics.market_id]) > 100:
            self.liquidity_snapshots[metrics.market_id] = \
                self.liquidity_snapshots[metrics.market_id][-100:]
    
    def get_execution_stats(self, market_id: Optional[str] = None) -> Dict:
        """
        Get execution quality statistics.
        
        Args:
            market_id: Filter by market (optional)
        
        Returns:
            Dictionary of statistics
        """
        executions = [
            e for e in self.execution_history
            if market_id is None or e.market_id == market_id
        ]
        
        if not executions:
            return {}
        
        slippages = [e.slippage_pct for e in executions if e.liquidity_sufficient]
        
        return {
            "total_executions": len(executions),
            "avg_slippage_pct": float(sum(slippages) / len(slippages)) if slippages else 0,
            "max_slippage_pct": float(max(slippages)) if slippages else 0,
            "min_slippage_pct": float(min(slippages)) if slippages else 0,
            "insufficient_liquidity_count": len([e for e in executions if not e.liquidity_sufficient]),
            "quality_distribution": {
                quality: len([e for e in executions if e.execution_quality == quality])
                for quality in ["EXCELLENT", "GOOD", "FAIR", "POOR", "INSUFFICIENT_LIQUIDITY"]
            }
        }
    
    def get_liquidity_health(self, market_id: str) -> Optional[Dict]:
        """
        Get current liquidity health metrics.
        
        Args:
            market_id: Market to check
        
        Returns:
            Health metrics or None if no data
        """
        if market_id not in self.liquidity_snapshots:
            return None
        
        snapshots = self.liquidity_snapshots[market_id]
        if not snapshots:
            return None
        
        latest = snapshots[-1]
        
        # Calculate trends if we have historical data
        if len(snapshots) >= 10:
            recent_spreads = [s.spread_bps for s in snapshots[-10:]]
            spread_trend = "improving" if recent_spreads[-1] < sum(recent_spreads[:-1]) / 9 else "degrading"
        else:
            spread_trend = "stable"
        
        return {
            "is_healthy": latest.is_healthy,
            "spread_bps": latest.spread_bps,
            "spread_trend": spread_trend,
            "bid_depth": latest.bid_depth,
            "ask_depth": latest.ask_depth,
            "depth_imbalance": float(latest.depth_imbalance),
            "top_of_book_size": latest.top_of_book_size,
            "timestamp": latest.timestamp.isoformat()
        }


# Convenience function for quick VWAP checks
def quick_vwap_check(bids: List[Tuple[Decimal, int]],
                     asks: List[Tuple[Decimal, int]],
                     quantity: int,
                     side: str,
                     market_id: str = "unknown") -> VWAPResult:
    """
    Quick VWAP calculation without instantiating calculator.
    
    Args:
        bids: Bid side of order book
        asks: Ask side of order book
        quantity: Order size
        side: "buy" or "sell"
        market_id: Market identifier
    
    Returns:
        VWAPResult
    """
    calc = VWAPCalculator()
    orders = asks if side == "buy" else bids
    return calc.calculate_vwap(orders, quantity, side, market_id)
