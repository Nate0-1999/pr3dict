"""
PR3DICT: VWAP Monitoring Dashboard Example

Live monitoring of VWAP metrics, liquidity health, and execution quality.
Demonstrates real-time order book analysis and alerting.
"""

import asyncio
import logging
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List

from src.platforms.polymarket import PolymarketPlatform
from src.platforms.kalshi import KalshiPlatform
from src.data.vwap import (
    VWAPCalculator,
    VWAPMonitor,
    LiquidityMetrics,
    HistoricalVWAPAnalyzer
)
from src.risk.vwap_checks import VWAPRiskManager, VWAPRiskConfig
from src.platforms.base import OrderSide

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VWAPDashboard:
    """
    Real-time VWAP monitoring dashboard.
    
    Features:
    - Live liquidity monitoring
    - Spread tracking
    - Slippage alerts
    - Price impact curves
    - Execution quality statistics
    """
    
    def __init__(self, platforms: List, refresh_interval: int = 30):
        """
        Args:
            platforms: List of platform interfaces (Polymarket, Kalshi, etc.)
            refresh_interval: Seconds between updates
        """
        self.platforms = platforms
        self.refresh_interval = refresh_interval
        
        self.calculator = VWAPCalculator()
        self.monitor = VWAPMonitor(self.calculator)
        self.risk_manager = VWAPRiskManager(vwap_config=VWAPRiskConfig())
        
        # Watchlist markets
        self.watchlist: List[str] = []
        
        # Alert thresholds
        self.alert_spread_bps = 500  # Alert if spread > 5%
        self.alert_liquidity_depth = 200  # Alert if depth < 200 contracts
        self.alert_slippage_pct = Decimal("3.0")  # Alert if slippage > 3%
    
    async def add_market_to_watchlist(self, platform_name: str, market_id: str):
        """Add a market to the monitoring watchlist."""
        self.watchlist.append(f"{platform_name}:{market_id}")
        logger.info(f"Added {platform_name}:{market_id} to watchlist")
    
    async def monitor_loop(self):
        """Main monitoring loop."""
        logger.info("Starting VWAP monitoring dashboard...")
        
        while True:
            try:
                await self._update_all_markets()
                await asyncio.sleep(self.refresh_interval)
            except KeyboardInterrupt:
                logger.info("Dashboard stopped by user")
                break
            except Exception as e:
                logger.error(f"Dashboard error: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _update_all_markets(self):
        """Update metrics for all watchlist markets."""
        logger.info(f"\n{'='*80}")
        logger.info(f"VWAP Dashboard Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}")
        
        for market_ref in self.watchlist:
            try:
                platform_name, market_id = market_ref.split(":")
                platform = self._get_platform(platform_name)
                
                if not platform:
                    continue
                
                await self._analyze_market(platform, market_id)
                
            except Exception as e:
                logger.error(f"Error analyzing {market_ref}: {e}")
        
        # Print summary statistics
        self._print_summary()
    
    async def _analyze_market(self, platform, market_id: str):
        """Analyze a single market."""
        # Fetch order book
        orderbook = await platform.get_orderbook(market_id)
        
        if not orderbook.bids or not orderbook.asks:
            logger.warning(f"{market_id}: No order book data")
            return
        
        # Calculate liquidity metrics
        liquidity = self.calculator.calculate_liquidity_metrics(
            bids=orderbook.bids,
            asks=orderbook.asks,
            market_id=market_id
        )
        
        # Record snapshot
        self.monitor.record_liquidity_snapshot(liquidity)
        
        # Calculate VWAP for sample sizes
        sample_sizes = [100, 500, 1000]
        
        logger.info(f"\n{market_id}:")
        logger.info(f"  Liquidity:")
        logger.info(f"    Bid depth: {liquidity.bid_depth} contracts (${liquidity.bid_value:.2f})")
        logger.info(f"    Ask depth: {liquidity.ask_depth} contracts (${liquidity.ask_value:.2f})")
        logger.info(f"    Spread: {liquidity.spread_bps} bps")
        logger.info(f"    Top of book: {liquidity.top_of_book_size} contracts")
        logger.info(f"    Depth imbalance: {liquidity.depth_imbalance:.2%}")
        logger.info(f"    Health: {'✓ HEALTHY' if liquidity.is_healthy else '✗ UNHEALTHY'}")
        
        # VWAP analysis for different sizes
        logger.info(f"  VWAP Analysis (BUY):")
        for size in sample_sizes:
            if size > sum(qty for _, qty in orderbook.asks):
                continue
            
            vwap = self.calculator.calculate_vwap(
                orders=orderbook.asks,
                quantity=size,
                side="buy",
                market_id=market_id
            )
            
            self.monitor.record_execution(vwap)
            
            logger.info(
                f"    {size:4d} contracts: VWAP=${vwap.vwap_price:.4f}, "
                f"slippage={vwap.slippage_pct:.2f}%, "
                f"quality={vwap.execution_quality}"
            )
        
        # Check for alerts
        self._check_alerts(market_id, liquidity)
    
    def _check_alerts(self, market_id: str, liquidity: LiquidityMetrics):
        """Check for alert conditions."""
        alerts = []
        
        # Spread alert
        if liquidity.spread_bps > self.alert_spread_bps:
            alerts.append(f"WIDE SPREAD: {liquidity.spread_bps} bps")
        
        # Liquidity alert
        if liquidity.bid_depth < self.alert_liquidity_depth:
            alerts.append(f"LOW BID DEPTH: {liquidity.bid_depth} contracts")
        if liquidity.ask_depth < self.alert_liquidity_depth:
            alerts.append(f"LOW ASK DEPTH: {liquidity.ask_depth} contracts")
        
        # Depth imbalance alert
        if liquidity.depth_imbalance < Decimal("0.3") or liquidity.depth_imbalance > Decimal("0.7"):
            alerts.append(f"DEPTH IMBALANCE: {liquidity.depth_imbalance:.1%}")
        
        if alerts:
            logger.warning(f"  ⚠️  ALERTS for {market_id}:")
            for alert in alerts:
                logger.warning(f"      - {alert}")
    
    def _print_summary(self):
        """Print summary statistics."""
        stats = self.monitor.get_execution_stats()
        
        if not stats:
            return
        
        logger.info(f"\n{'='*80}")
        logger.info("Overall Statistics:")
        logger.info(f"  Total executions analyzed: {stats.get('total_executions', 0)}")
        logger.info(f"  Average slippage: {stats.get('avg_slippage_pct', 0):.2f}%")
        logger.info(f"  Max slippage: {stats.get('max_slippage_pct', 0):.2f}%")
        logger.info(f"  Insufficient liquidity events: {stats.get('insufficient_liquidity_count', 0)}")
        
        quality_dist = stats.get('quality_distribution', {})
        logger.info("  Execution quality distribution:")
        for quality, count in quality_dist.items():
            if count > 0:
                pct = count / stats['total_executions'] * 100
                logger.info(f"    {quality}: {count} ({pct:.1f}%)")
        
        # Risk manager stats
        vwap_stats = self.risk_manager.get_vwap_statistics()
        logger.info(f"\nVWAP Risk Checks:")
        logger.info(f"  Rejections: {vwap_stats['vwap_rejections']}")
        logger.info(f"  Adjustments: {vwap_stats['vwap_adjustments']}")
        logger.info(f"  Rejection rate: {vwap_stats['rejection_rate_pct']:.1f}%")
        logger.info(f"{'='*80}\n")
    
    def _get_platform(self, name: str):
        """Get platform by name."""
        for platform in self.platforms:
            if platform.name == name:
                return platform
        return None


async def main():
    """
    Example dashboard usage.
    
    Monitors a few popular markets and displays real-time VWAP metrics.
    """
    # Initialize platforms
    polymarket = PolymarketPlatform()
    
    try:
        # Connect to platform
        connected = await polymarket.connect()
        if not connected:
            logger.error("Failed to connect to Polymarket")
            return
        
        # Create dashboard
        dashboard = VWAPDashboard(
            platforms=[polymarket],
            refresh_interval=30  # Update every 30 seconds
        )
        
        # Add markets to watchlist
        # Example: Monitor popular political markets
        # You would replace these with actual market IDs
        example_markets = [
            # Format: "platform:market_id"
            # These are placeholder IDs - replace with real ones
            # "polymarket:0x123...",
            # "polymarket:0x456...",
        ]
        
        # If no markets specified, fetch some popular ones
        if not example_markets:
            logger.info("Fetching popular markets...")
            markets = await polymarket.get_markets(limit=5)
            for market in markets[:3]:  # Monitor top 3
                await dashboard.add_market_to_watchlist("polymarket", market.id)
        else:
            for market_ref in example_markets:
                platform_name, market_id = market_ref.split(":")
                await dashboard.add_market_to_watchlist(platform_name, market_id)
        
        # Run monitoring loop
        await dashboard.monitor_loop()
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped")
    finally:
        await polymarket.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
