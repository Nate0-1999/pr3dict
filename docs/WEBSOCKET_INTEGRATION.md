# WebSocket Integration Guide

## Quick Start

### 1. Install Dependencies

```bash
# WebSocket client
pip install websockets

# Redis (for pub/sub and caching)
pip install redis

# Start Redis server
redis-server
```

### 2. Basic Usage

```python
from src.data.orderbook_manager import OrderBookManager

# Create manager
manager = OrderBookManager(
    asset_ids=["<asset-id-1>", "<asset-id-2>"],
    redis_url="redis://localhost:6379/0",
    enable_custom_features=True
)

# Start real-time feeds
await manager.start()

# Query orderbook
book = manager.get_orderbook("<asset-id>")
print(f"Best bid: {book.best_bid}, Best ask: {book.best_ask}")

# Calculate VWAP
vwap = manager.calculate_vwap("<asset-id>", side="BUY", depth_usdc=100)
print(f"VWAP for $100 buy: {vwap}")

# Get metrics
metrics = manager.get_metrics("<asset-id>")
print(f"Spread: {metrics.spread_bps} bps")
print(f"Update latency: {metrics.update_latency_ms:.2f}ms")

# Stop when done
await manager.stop()
```

### 3. Platform Integration

```python
from src.platforms.polymarket import PolymarketPlatform

# Initialize with WebSocket enabled
platform = PolymarketPlatform(
    use_websocket=True,  # Enable WebSocket (<5ms latency)
    redis_url="redis://localhost:6379/0"
)

await platform.connect()

# Get orderbook (uses WebSocket if available, falls back to REST)
orderbook = await platform.get_orderbook("<asset-id>")

# Get real-time metrics (WebSocket only)
metrics = platform.get_orderbook_metrics("<asset-id>")
print(f"Spread: {metrics['spread_bps']} bps")

# Calculate VWAP
vwap = platform.calculate_vwap("<asset-id>", "BUY", depth_usdc=250)
```

## Architecture

### Components

1. **WebSocket Client** (`src/data/websocket_client.py`)
   - Low-level WebSocket connection management
   - Message parsing and routing
   - Auto-reconnect with exponential backoff
   - Heartbeat (PING/PONG) handling

2. **OrderBook Manager** (`src/data/orderbook_manager.py`)
   - High-level orderbook management
   - Multi-asset tracking
   - Metrics calculation (spread, VWAP, liquidity)
   - Redis integration for pub/sub

3. **Platform Integration** (`src/platforms/polymarket.py`)
   - Seamless REST + WebSocket integration
   - Automatic fallback to REST if WebSocket unavailable
   - Unified API for both data sources

4. **Monitor** (`src/data/websocket_monitor.py`)
   - Real-time health dashboard
   - Latency tracking
   - Connection monitoring

### Data Flow

```
Polymarket WebSocket
        ↓
WebSocketClient (parsing, reconnect)
        ↓
OrderBookManager (metrics, callbacks)
        ↓         ↓
    Redis       Your Strategy
   (pub/sub)    (callbacks)
```

## Finding Asset IDs

Asset IDs (token IDs) are the unique identifiers for outcome tokens on Polymarket.

### Method 1: Gamma API

```python
import httpx

async def get_markets():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 10, "active": True}
        )
        markets = response.json()
        
        for market in markets:
            print(f"Question: {market['question']}")
            print(f"Condition ID: {market['conditionId']}")
            
            # Asset IDs for YES and NO tokens
            for i, outcome in enumerate(market['outcomes']):
                asset_id = market['tokens'][i]['token_id']
                print(f"  {outcome}: {asset_id}")
```

### Method 2: py-clob-client

```python
from py_clob_client.client import ClobClient

client = ClobClient("https://clob.polymarket.com")
markets = client.get_simplified_markets()

for market in markets['data'][:5]:
    print(f"Question: {market['question']}")
    print(f"Condition ID: {market['condition_id']}")
    print(f"YES token: {market['tokens'][0]['token_id']}")
    print(f"NO token: {market['tokens'][1]['token_id']}")
```

## Callbacks and Events

### Register Callbacks

```python
manager = OrderBookManager(asset_ids=["..."])

# Orderbook updates
async def on_book(snapshot, metrics):
    print(f"Book update: {snapshot.best_bid}/{snapshot.best_ask}")
    print(f"Spread: {metrics.spread_bps} bps")

manager.register_book_callback(on_book)

# Trade events
async def on_trade(trade):
    print(f"Trade: {trade.side} {trade.size} @ {trade.price}")

manager.register_trade_callback(on_trade)

await manager.start()
```

### Redis Pub/Sub

Subscribe to updates in separate processes/services:

```python
import redis.asyncio as redis
import json

r = await redis.from_url("redis://localhost:6379/0")
pubsub = r.pubsub()

# Subscribe to orderbook updates
await pubsub.subscribe("polymarket:orderbook:<asset-id>")

async for message in pubsub.listen():
    if message['type'] == 'message':
        data = json.loads(message['data'])
        print(f"Orderbook: {data['best_bid']}/{data['best_ask']}")

# Subscribe to trades
await pubsub.subscribe("polymarket:trade:<asset-id>")
```

## Monitoring

### Dashboard

Run the real-time monitoring dashboard:

```bash
python -m src.data.websocket_monitor
```

Displays:
- Connection status
- Message latency (avg, p50, p95, max)
- Orderbook health per asset
- Trade flow
- Staleness alerts

### Programmatic Stats

```python
# OrderBook Manager stats
stats = manager.get_stats()
print(f"Connected: {stats['connected']}")
print(f"Avg Latency: {stats['latency_avg_ms']:.2f}ms")
print(f"P95 Latency: {stats['latency_p95_ms']:.2f}ms")
print(f"Tracked Assets: {stats['tracked_assets']}")

# Platform stats
ws_stats = platform.get_websocket_stats()
if ws_stats:
    print(f"WebSocket Status: {ws_stats['connected']}")
    print(f"Subscribed Assets: {ws_stats['subscribed_assets']}")
```

## Performance Tuning

### Latency Optimization

1. **Run close to exchange**
   - Deploy in same cloud region as Polymarket (US-based)
   - Use low-latency network

2. **Process messages efficiently**
   ```python
   # Use orjson for faster JSON parsing
   import orjson
   data = orjson.loads(message)
   ```

3. **Limit orderbook depth**
   ```python
   # Only track top N levels
   MAX_DEPTH = 20
   bids = bids[:MAX_DEPTH]
   asks = asks[:MAX_DEPTH]
   ```

### Resource Management

1. **Limit trade history**
   ```python
   manager = OrderBookManager(asset_ids=["..."])
   manager._max_trade_history = 50  # Keep only 50 recent trades
   ```

2. **Use Redis TTL**
   - Orderbooks automatically expire in 5 seconds
   - Trades not cached (pub/sub only)

3. **Subscribe selectively**
   ```python
   # Only subscribe to assets you're actively trading
   manager = OrderBookManager(asset_ids=[])  # Start empty
   
   # Subscribe on-demand
   await manager.subscribe(["<asset-id>"])
   
   # Unsubscribe when done
   await manager.unsubscribe(["<asset-id>"])
   ```

## Common Patterns

### Pattern 1: Price Alert

```python
async def on_book(snapshot, metrics):
    if snapshot.best_bid and snapshot.best_bid > Decimal("0.75"):
        print(f"🚨 ALERT: Best bid hit target: {snapshot.best_bid}")

manager.register_book_callback(on_book)
```

### Pattern 2: Spread Monitoring

```python
async def on_book(snapshot, metrics):
    if metrics.spread_bps and metrics.spread_bps < 10:
        print(f"✅ Tight spread: {metrics.spread_bps} bps - good liquidity!")

manager.register_book_callback(on_book)
```

### Pattern 3: Trade Flow Analysis

```python
buy_volume = Decimal("0")
sell_volume = Decimal("0")

async def on_trade(trade):
    global buy_volume, sell_volume
    
    if trade.side == "BUY":
        buy_volume += trade.size
    else:
        sell_volume += trade.size
    
    # Calculate imbalance
    total = buy_volume + sell_volume
    if total > 0:
        imbalance = (buy_volume - sell_volume) / total
        print(f"Order flow imbalance: {imbalance:.2%}")

manager.register_trade_callback(on_trade)
```

### Pattern 4: Latency-Sensitive Strategy

```python
async def on_book(snapshot, metrics):
    # Only act if data is fresh (<10ms latency)
    if metrics.update_latency_ms < 10:
        # Your high-frequency logic here
        if snapshot.spread and snapshot.spread < Decimal("0.01"):
            # Tight spread, good for market making
            pass
    else:
        print(f"⚠️ Stale data: {metrics.update_latency_ms:.2f}ms")

manager.register_book_callback(on_book)
```

## Troubleshooting

### WebSocket not connecting

**Check Redis:**
```bash
redis-cli ping
# Should return "PONG"
```

**Check dependencies:**
```bash
pip list | grep websockets
pip list | grep redis
```

**Check logs:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### High latency

**Check network:**
```bash
ping clob.polymarket.com
```

**Check Redis:**
```bash
redis-cli --latency
```

**Check message processing:**
```python
# Profile callback execution time
import time

async def on_book(snapshot, metrics):
    start = time.time()
    # Your logic
    elapsed = (time.time() - start) * 1000
    if elapsed > 1:
        print(f"⚠️ Callback took {elapsed:.2f}ms")
```

### Missing orderbook updates

**Verify subscription:**
```python
stats = manager.get_stats()
print(f"Subscribed: {stats['subscribed_assets']}")
print(f"Cached: {stats['orderbooks_cached']}")
```

**Check asset ID:**
```python
# Ensure using token ID (asset_id), not condition ID (market_id)
# token_id: 71321045679252212594626385532706912750332728571942532289631379312455583992563
# market_id: 0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af
```

## Next Steps

- **Strategy Integration:** Use real-time data in your trading strategies
- **Backtesting:** Record WebSocket data for replay in backtests
- **Alerting:** Set up alerts for price movements, spread changes, etc.
- **Multi-Platform:** Extend WebSocket support to other platforms (Kalshi, etc.)

## References

- [Polymarket WebSocket Docs](https://docs.polymarket.com/developers/CLOB/websocket/wss-overview)
- [WebSocket API Reference](./WEBSOCKET_API.md)
- [Example Code](../examples/websocket_example.py)
- [Monitor Dashboard](../src/data/websocket_monitor.py)
