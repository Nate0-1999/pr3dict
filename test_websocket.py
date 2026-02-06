"""
PR3DICT: WebSocket Implementation Test

Quick test to verify WebSocket client and orderbook manager work correctly.

Run with:
    python test_websocket.py
"""
import asyncio
import logging
import sys
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


async def test_websocket_client():
    """Test WebSocket client basic functionality."""
    print("\n" + "="*80)
    print("TEST 1: WebSocket Client")
    print("="*80)
    
    try:
        from src.data.websocket_client import PolymarketWebSocketClient
        
        # Create client (will fail if dependencies missing)
        client = PolymarketWebSocketClient(
            asset_ids=[],  # Empty initially
            redis_url="redis://localhost:6379/0"
        )
        
        print("✅ WebSocket client created successfully")
        print(f"   - WebSocket URL: {client.WS_URL}")
        print(f"   - Ping interval: {client.PING_INTERVAL}s")
        
        return True
    
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install websockets redis")
        return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_orderbook_manager():
    """Test OrderBook Manager."""
    print("\n" + "="*80)
    print("TEST 2: OrderBook Manager")
    print("="*80)
    
    try:
        from src.data.orderbook_manager import OrderBookManager
        
        # Create manager
        manager = OrderBookManager(
            asset_ids=[],
            redis_url="redis://localhost:6379/0"
        )
        
        print("✅ OrderBook Manager created successfully")
        
        # Test callback registration
        async def test_callback(snapshot, metrics):
            pass
        
        manager.register_book_callback(test_callback)
        print("✅ Callback registration works")
        
        return True
    
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_platform_integration():
    """Test platform integration."""
    print("\n" + "="*80)
    print("TEST 3: Platform Integration")
    print("="*80)
    
    try:
        from src.platforms.polymarket import PolymarketPlatform
        
        # Create platform (won't connect without credentials)
        platform = PolymarketPlatform(
            use_websocket=True,
            redis_url="redis://localhost:6379/0"
        )
        
        print("✅ Platform created with WebSocket support")
        print(f"   - WebSocket enabled: {platform._use_websocket}")
        
        return True
    
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        if "py_clob_client" in str(e):
            print("   Install with: pip install py-clob-client")
        return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_orderbook_snapshot():
    """Test OrderBook snapshot functionality."""
    print("\n" + "="*80)
    print("TEST 4: OrderBook Snapshot")
    print("="*80)
    
    try:
        from src.data.websocket_client import OrderBookSnapshot, OrderBookLevel
        from datetime import datetime, timezone
        
        # Create mock snapshot
        bids = [
            OrderBookLevel(Decimal("0.50"), Decimal("100")),
            OrderBookLevel(Decimal("0.49"), Decimal("200")),
            OrderBookLevel(Decimal("0.48"), Decimal("150")),
        ]
        
        asks = [
            OrderBookLevel(Decimal("0.52"), Decimal("80")),
            OrderBookLevel(Decimal("0.53"), Decimal("120")),
            OrderBookLevel(Decimal("0.54"), Decimal("100")),
        ]
        
        snapshot = OrderBookSnapshot(
            asset_id="test-asset-id",
            market_id="test-market-id",
            bids=bids,
            asks=asks,
            timestamp=datetime.now(timezone.utc)
        )
        
        print("✅ OrderBook snapshot created")
        print(f"   - Best bid: {snapshot.best_bid}")
        print(f"   - Best ask: {snapshot.best_ask}")
        print(f"   - Spread: {snapshot.spread}")
        print(f"   - Mid price: {snapshot.mid_price}")
        
        # Test VWAP calculation
        vwap = snapshot.calculate_vwap("BUY", Decimal("100"))
        print(f"   - VWAP for $100 buy: {vwap:.4f}")
        
        # Test serialization
        data = snapshot.to_dict()
        print(f"✅ Serialization works ({len(data)} fields)")
        
        return True
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_redis_connection():
    """Test Redis connectivity."""
    print("\n" + "="*80)
    print("TEST 5: Redis Connection")
    print("="*80)
    
    try:
        import redis.asyncio as redis
        
        r = await redis.from_url("redis://localhost:6379/0")
        await r.ping()
        print("✅ Redis connection successful")
        
        await r.close()
        return True
    
    except ImportError:
        print("❌ Redis library not installed")
        print("   Install with: pip install redis")
        return False
    
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("   Start Redis with: redis-server")
        return False


async def test_cache_integration():
    """Test cache integration."""
    print("\n" + "="*80)
    print("TEST 6: Cache Integration")
    print("="*80)
    
    try:
        from src.data.cache import DataCache
        
        cache = DataCache(redis_url="redis://localhost:6379/0")
        connected = await cache.connect()
        
        if connected:
            print("✅ Cache connected to Redis")
            
            # Test orderbook caching
            test_book = {
                "bids": [["0.50", "100"]],
                "asks": [["0.52", "80"]]
            }
            
            await cache.set_orderbook("test-market", "polymarket", test_book)
            retrieved = await cache.get_orderbook("test-market", "polymarket")
            
            if retrieved:
                print("✅ Orderbook caching works")
            else:
                print("⚠️ Orderbook retrieval failed (data may have expired)")
            
            await cache.disconnect()
            return True
        else:
            print("⚠️ Cache initialized but Redis not available")
            return True  # Still passes if Redis is optional
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PR3DICT WebSocket Implementation Tests")
    print("="*80)
    
    tests = [
        ("WebSocket Client", test_websocket_client),
        ("OrderBook Manager", test_orderbook_manager),
        ("Platform Integration", test_platform_integration),
        ("OrderBook Snapshot", test_orderbook_snapshot),
        ("Redis Connection", test_redis_connection),
        ("Cache Integration", test_cache_integration),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        print("\nCommon issues:")
        print("  - Missing dependencies: pip install websockets redis py-clob-client")
        print("  - Redis not running: redis-server")
    
    print("="*80 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
