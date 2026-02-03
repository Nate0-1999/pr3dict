"""
PR3DICT: WebSocket Feed Health Monitor

Real-time monitoring dashboard for WebSocket data feeds.
Displays connection status, latency metrics, orderbook health,
and trade flow statistics.

Run with: python -m src.data.websocket_monitor
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional
from decimal import Decimal

from .orderbook_manager import OrderBookManager
from .websocket_client import OrderBookSnapshot, TradeEvent

logger = logging.getLogger(__name__)


class WebSocketMonitor:
    """
    Real-time monitoring dashboard for WebSocket feeds.
    
    Displays:
    - Connection health
    - Message latency (avg, p50, p95, max)
    - Orderbook metrics per asset
    - Trade flow
    - Alert on stale data
    """
    
    def __init__(
        self,
        manager: OrderBookManager,
        refresh_interval: float = 1.0,
        stale_threshold_seconds: float = 10.0,
    ):
        """
        Initialize monitor.
        
        Args:
            manager: OrderBookManager instance to monitor
            refresh_interval: How often to refresh display (seconds)
            stale_threshold_seconds: Alert if no updates for this long
        """
        self.manager = manager
        self.refresh_interval = refresh_interval
        self.stale_threshold = stale_threshold_seconds
        
        # Event counters
        self._book_updates = 0
        self._trade_events = 0
        self._last_book_time: Optional[datetime] = None
        self._last_trade_time: Optional[datetime] = None
        
        # Register callbacks
        self.manager.register_book_callback(self._on_book)
        self.manager.register_trade_callback(self._on_trade)
    
    async def _on_book(self, snapshot: OrderBookSnapshot, metrics) -> None:
        """Track book updates."""
        self._book_updates += 1
        self._last_book_time = datetime.now()
    
    async def _on_trade(self, trade: TradeEvent) -> None:
        """Track trade events."""
        self._trade_events += 1
        self._last_trade_time = datetime.now()
    
    def _check_stale(self, last_time: Optional[datetime]) -> bool:
        """Check if data is stale."""
        if not last_time:
            return False
        
        elapsed = (datetime.now() - last_time).total_seconds()
        return elapsed > self.stale_threshold
    
    def _format_latency(self, latency_ms: float) -> str:
        """Format latency with color coding."""
        if latency_ms < 5:
            return f"✅ {latency_ms:.1f}ms"
        elif latency_ms < 10:
            return f"⚠️  {latency_ms:.1f}ms"
        else:
            return f"🔴 {latency_ms:.1f}ms"
    
    def _format_spread(self, spread_bps: Optional[int]) -> str:
        """Format spread in basis points."""
        if spread_bps is None:
            return "N/A"
        
        if spread_bps < 10:
            return f"✅ {spread_bps}bp"
        elif spread_bps < 50:
            return f"⚠️  {spread_bps}bp"
        else:
            return f"🔴 {spread_bps}bp"
    
    def _render_dashboard(self) -> str:
        """Render the monitoring dashboard."""
        stats = self.manager.get_stats()
        all_metrics = self.manager.get_all_metrics()
        
        lines = []
        lines.append("=" * 80)
        lines.append("🔌 POLYMARKET WEBSOCKET FEED MONITOR")
        lines.append("=" * 80)
        
        # Connection Status
        lines.append("\n📡 CONNECTION STATUS:")
        status = "🟢 CONNECTED" if stats["connected"] else "🔴 DISCONNECTED"
        lines.append(f"  Status: {status}")
        lines.append(f"  Subscribed Assets: {stats['subscribed_assets']}")
        lines.append(f"  Orderbooks Cached: {stats['orderbooks_cached']}")
        lines.append(f"  Reconnect Attempts: {stats['reconnect_attempts']}")
        
        # Latency Metrics
        lines.append("\n⚡ LATENCY METRICS:")
        lines.append(f"  Average: {self._format_latency(stats['latency_avg_ms'])}")
        lines.append(f"  P50: {self._format_latency(stats['latency_p50_ms'])}")
        lines.append(f"  P95: {self._format_latency(stats['latency_p95_ms'])}")
        lines.append(f"  Max: {self._format_latency(stats['latency_max_ms'])}")
        lines.append(f"  Last Message: {stats['last_message_ago_s']:.1f}s ago")
        
        # Data Flow
        lines.append("\n📊 DATA FLOW:")
        lines.append(f"  Book Updates: {self._book_updates}")
        lines.append(f"  Trade Events: {self._trade_events}")
        lines.append(f"  Total Trades Tracked: {stats['total_trades_tracked']}")
        
        # Staleness check
        if self._check_stale(self._last_book_time):
            lines.append(f"  🔴 WARNING: No book updates for {self.stale_threshold}s!")
        
        # Orderbook Health
        if all_metrics:
            lines.append("\n📖 ORDERBOOK HEALTH:")
            lines.append(f"  {'Asset ID':<12} {'Bid':<8} {'Ask':<8} {'Spread':<10} {'VWAP Buy':<10} {'Latency':<12}")
            lines.append("  " + "-" * 70)
            
            for asset_id, metrics in list(all_metrics.items())[:10]:  # Show top 10
                asset_short = asset_id[:12]
                bid = f"{metrics.best_bid:.3f}" if metrics.best_bid else "N/A"
                ask = f"{metrics.best_ask:.3f}" if metrics.best_ask else "N/A"
                spread = self._format_spread(metrics.spread_bps)
                vwap = f"{metrics.vwap_buy_100:.3f}" if metrics.vwap_buy_100 else "N/A"
                latency = self._format_latency(metrics.update_latency_ms)
                
                lines.append(f"  {asset_short:<12} {bid:<8} {ask:<8} {spread:<10} {vwap:<10} {latency:<12}")
        
        # Overall Stats
        lines.append("\n📈 OVERALL STATS:")
        lines.append(f"  Avg Spread: {stats['avg_spread_bps']:.1f} bps")
        lines.append(f"  Avg Update Latency: {stats['avg_update_latency_ms']:.2f}ms")
        
        lines.append("\n" + "=" * 80)
        lines.append(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    async def run(self) -> None:
        """Run the monitoring dashboard."""
        try:
            while True:
                # Clear screen (Unix/Mac)
                print("\033[2J\033[H", end="")
                
                # Render and display
                dashboard = self._render_dashboard()
                print(dashboard)
                
                # Wait for next refresh
                await asyncio.sleep(self.refresh_interval)
        
        except KeyboardInterrupt:
            logger.info("Monitor stopped by user")
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        stats = self.manager.get_stats()
        
        return {
            "connected": stats["connected"],
            "book_updates": self._book_updates,
            "trade_events": self._trade_events,
            "latency_avg_ms": stats["latency_avg_ms"],
            "latency_p95_ms": stats["latency_p95_ms"],
            "last_message_ago_s": stats["last_message_ago_s"],
            "is_stale": self._check_stale(self._last_book_time),
        }


async def main():
    """Run standalone monitoring dashboard."""
    import sys
    import os
    
    # Add project root to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Example asset IDs (replace with actual)
    # These are mock IDs for demonstration
    asset_ids = [
        "71321045679252212594626385532706912750332728571942532289631379312455583992563",
        "52114319501245915516055106046884209969926127482827954674443846427813813222426",
    ]
    
    # Create manager
    manager = OrderBookManager(asset_ids=asset_ids)
    
    # Create monitor
    monitor = WebSocketMonitor(manager)
    
    # Start manager
    await manager.start()
    
    try:
        # Run monitor
        await monitor.run()
    finally:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
