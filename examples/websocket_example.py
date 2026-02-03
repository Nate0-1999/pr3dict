"""
PR3DICT: WebSocket Real-Time Data Example

Demonstrates WebSocket integration for <5ms latency market data.
Shows orderbook tracking, VWAP calculation, and trade monitoring.

Usage:
    python examples/websocket_example.py
"""
import asyncio
import logging
import os
import sys
from decimal import Decimal

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.platforms.polymarket import PolymarketPlatform
from src.data.orderbook_manager import OrderBookManager
from src.data.websocket_client import OrderBookSnapshot, TradeEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


async def example_1_basic_websocket():
    """Example 1: Basic WebSocket orderbook tracking."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic WebSocket Orderbook Tracking")
    print("="*80 + "\n")
    
    # Example asset ID (replace with real asset ID from Gamma API)
    # This is a mock ID - get real ones from: https://gamma-api.polymarket.com/markets
    asset_id = "71321045679252212594626385532706912750332728571942532289631379312455583992563"
    
    # Create orderbook manager
    manager = OrderBookManager(
        asset_ids=[asset_id],
        redis_url="redis://localhost:6379/0",
        enable_custom_features=True
    )
    
    # Register callback for updates
    async def on_book_update(snapshot: OrderBookSnapshot, metrics):
        print(f"\n📖 Orderbook Update:")
        print(f"  Asset: {snapshot.asset_id[:12]}...")
        print(f"  Best Bid: {snapshot.best_bid}")
        print(f"  Best Ask: {snapshot.best_ask}")
        print(f"  Spread: {metrics.spread_bps} bps")
        print(f"  Mid Price: {snapshot.mid_price}")
        print(f"  Update Latency: {metrics.update_latency_ms:.2f}ms")
    
    manager.register_book_callback(on_book_update)
    
    # Start manager
    await manager.start()
    
    try:
        # Run for 30 seconds
        print("Listening for orderbook updates for 30 seconds...")
        await asyncio.sleep(30)
        
        # Get final stats
        stats = manager.get_stats()
        print(f"\n📊 Final Statistics:")
        print(f"  Connection: {'Connected' if stats['connected'] else 'Disconnected'}")
        print(f"  Avg Latency: {stats['latency_avg_ms']:.2f}ms")
        print(f"  P95 Latency: {stats['latency_p95_ms']:.2f}ms")
        print(f"  Max Latency: {stats['latency_max_ms']:.2f}ms")
    
    finally:
        await manager.stop()


async def example_2_vwap_calculation():
    """Example 2: Real-time VWAP calculation."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Real-Time VWAP Calculation")
    print("="*80 + "\n")
    
    asset_id = "71321045679252212594626385532706912750332728571942532289631379312455583992563"
    
    manager = OrderBookManager(asset_ids=[asset_id])
    
    async def on_book_update(snapshot: OrderBookSnapshot, metrics):
        # Calculate VWAP for different depths
        vwap_100 = snapshot.calculate_vwap("BUY", Decimal("100"))
        vwap_500 = snapshot.calculate_vwap("BUY", Decimal("500"))
        vwap_1000 = snapshot.calculate_vwap("BUY", Decimal("1000"))
        
        print(f"\n💰 VWAP Analysis:")
        print(f"  Mid Price: {snapshot.mid_price:.4f}")
        print(f"  VWAP $100:  {vwap_100:.4f} ({((vwap_100/snapshot.mid_price - 1) * 10000):.1f} bps slippage)")
        print(f"  VWAP $500:  {vwap_500:.4f} ({((vwap_500/snapshot.mid_price - 1) * 10000):.1f} bps slippage)")
        print(f"  VWAP $1000: {vwap_1000:.4f} ({((vwap_1000/snapshot.mid_price - 1) * 10000):.1f} bps slippage)")
    
    manager.register_book_callback(on_book_update)
    
    await manager.start()
    
    try:
        await asyncio.sleep(30)
    finally:
        await manager.stop()


async def example_3_trade_monitoring():
    """Example 3: Real-time trade monitoring."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Real-Time Trade Monitoring")
    print("="*80 + "\n")
    
    asset_id = "71321045679252212594626385532706912750332728571942532289631379312455583992563"
    
    manager = OrderBookManager(asset_ids=[asset_id])
    
    trade_count = 0
    total_volume = Decimal("0")
    
    async def on_trade(trade: TradeEvent):
        nonlocal trade_count, total_volume
        trade_count += 1
        total_volume += trade.size
        
        print(f"\n🔔 Trade #{trade_count}:")
        print(f"  Price: {trade.price}")
        print(f"  Size: {trade.size}")
        print(f"  Side: {trade.side}")
        print(f"  Total Volume: {total_volume}")
    
    manager.register_trade_callback(on_trade)
    
    await manager.start()
    
    try:
        await asyncio.sleep(30)
        print(f"\n📊 Trade Summary:")
        print(f"  Total Trades: {trade_count}")
        print(f"  Total Volume: {total_volume}")
    finally:
        await manager.stop()


async def example_4_platform_integration():
    """Example 4: Full platform integration with WebSocket."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Platform Integration with WebSocket")
    print("="*80 + "\n")
    
    # Initialize platform with WebSocket enabled
    platform = PolymarketPlatform(
        use_websocket=True,
        redis_url="redis://localhost:6379/0"
    )
    
    await platform.connect()
    
    try:
        # Example asset ID
        asset_id = "71321045679252212594626385532706912750332728571942532289631379312455583992563"
        
        print("Fetching market data...")
        await asyncio.sleep(2)  # Wait for WebSocket to connect
        
        # Get orderbook (uses WebSocket if available, falls back to REST)
        orderbook = await platform.get_orderbook(asset_id)
        print(f"\n📖 Orderbook:")
        print(f"  Best Bid: {orderbook.bids[0] if orderbook.bids else 'N/A'}")
        print(f"  Best Ask: {orderbook.asks[0] if orderbook.asks else 'N/A'}")
        
        # Get real-time metrics (WebSocket only)
        metrics = platform.get_orderbook_metrics(asset_id)
        if metrics:
            print(f"\n📊 Real-Time Metrics:")
            print(f"  Spread: {metrics['spread_bps']} bps")
            print(f"  VWAP Buy $100: {metrics['vwap_buy_100']}")
            print(f"  Update Latency: {metrics['update_latency_ms']:.2f}ms")
        
        # Calculate custom VWAP
        vwap = platform.calculate_vwap(asset_id, "BUY", Decimal("250"))
        if vwap:
            print(f"\n💰 VWAP for $250 buy: {vwap:.4f}")
        
        # Get WebSocket stats
        ws_stats = platform.get_websocket_stats()
        if ws_stats:
            print(f"\n⚡ WebSocket Performance:")
            print(f"  Connection: {'Connected' if ws_stats['connected'] else 'Disconnected'}")
            print(f"  Avg Latency: {ws_stats['latency_avg_ms']:.2f}ms")
            print(f"  P95 Latency: {ws_stats['latency_p95_ms']:.2f}ms")
        
        # Wait a bit to see updates
        print("\nListening for updates for 20 seconds...")
        await asyncio.sleep(20)
    
    finally:
        await platform.disconnect()


async def example_5_latency_comparison():
    """Example 5: Compare WebSocket vs REST latency."""
    print("\n" + "="*80)
    print("EXAMPLE 5: WebSocket vs REST Latency Comparison")
    print("="*80 + "\n")
    
    asset_id = "71321045679252212594626385532706912750332728571942532289631379312455583992563"
    
    # Platform with WebSocket
    platform_ws = PolymarketPlatform(use_websocket=True)
    await platform_ws.connect()
    
    # Platform without WebSocket
    platform_rest = PolymarketPlatform(use_websocket=False)
    await platform_rest.connect()
    
    try:
        await asyncio.sleep(2)  # Wait for WebSocket to sync
        
        # Benchmark WebSocket
        import time
        ws_times = []
        for i in range(10):
            start = time.time()
            book = await platform_ws.get_orderbook(asset_id)
            latency = (time.time() - start) * 1000
            ws_times.append(latency)
            await asyncio.sleep(0.1)
        
        # Benchmark REST
        rest_times = []
        for i in range(10):
            start = time.time()
            book = await platform_rest.get_orderbook(asset_id)
            latency = (time.time() - start) * 1000
            rest_times.append(latency)
            await asyncio.sleep(0.1)
        
        print(f"\n📊 Latency Comparison (10 samples each):")
        print(f"\n  WebSocket:")
        print(f"    Average: {sum(ws_times)/len(ws_times):.2f}ms")
        print(f"    Min: {min(ws_times):.2f}ms")
        print(f"    Max: {max(ws_times):.2f}ms")
        
        print(f"\n  REST API:")
        print(f"    Average: {sum(rest_times)/len(rest_times):.2f}ms")
        print(f"    Min: {min(rest_times):.2f}ms")
        print(f"    Max: {max(rest_times):.2f}ms")
        
        speedup = (sum(rest_times)/len(rest_times)) / (sum(ws_times)/len(ws_times))
        print(f"\n  🚀 WebSocket is {speedup:.1f}x faster!")
    
    finally:
        await platform_ws.disconnect()
        await platform_rest.disconnect()


async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("PR3DICT: WebSocket Real-Time Data Examples")
    print("="*80)
    
    examples = [
        ("Basic WebSocket", example_1_basic_websocket),
        ("VWAP Calculation", example_2_vwap_calculation),
        ("Trade Monitoring", example_3_trade_monitoring),
        ("Platform Integration", example_4_platform_integration),
        ("Latency Comparison", example_5_latency_comparison),
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning Example 1 (modify script to run others)...\n")
    
    # Run example 1 by default
    await examples[0][1]()
    
    print("\n" + "="*80)
    print("Example completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
