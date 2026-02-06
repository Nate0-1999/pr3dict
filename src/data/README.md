# PR3DICT Real-Time Data Layer

## Overview

High-performance WebSocket-based data feeds for prediction markets, providing <5ms latency market data (vs 50-100ms REST polling).

## Components

### 1. WebSocket Client (`websocket_client.py`)

Low-level WebSocket connection to Polymarket CLOB.

**Features:**
- Auto-reconnect with exponential backoff
- L2 orderbook reconstruction from incremental updates
- Real-time trade stream
- VWAP calculation from orderbook depth
- Heartbeat (PING/PONG) management
- Latency tracking

**Usage:**
```python
from src.data.websocket_client import PolymarketWebSocketClient

client = PolymarketWebSocketClient(
    asset_ids=["<asset-id>"],
    redis_url="redis://localhost:6379/0"
)

await client.run()
```

### 2. OrderBook Manager (`orderbook_manager.py`)

High-level orderbook management with metrics and callbacks.

**Features:**
- Multi-asset orderbook tracking
- Real-time metrics (spread, VWAP, liquidity)
- Event callbacks for book updates and trades
- Redis pub/sub integration
- Dynamic subscribe/unsubscribe

**Usage:**
```python
from src.data.orderbook_manager import OrderBookManager

manager = OrderBookManager(asset_ids=["<asset-id>"])
await manager.start()

# Query
book = manager.get_orderbook("<asset-id>")
metrics = manager.get_metrics("<asset-id>")
vwap = manager.calculate_vwap("<asset-id>", "BUY", 100)

# Callbacks
manager.register_book_callback(my_callback)
manager.register_trade_callback(my_trade_callback)
```

### 3. Monitor (`websocket_monitor.py`)

Real-time monitoring dashboard for feed health.

**Features:**
- Connection status
- Latency metrics (avg, p50, p95, max)
- Orderbook health per asset
- Trade flow statistics
- Staleness alerts

**Usage:**
```bash
python -m src.data.websocket_monitor
```

### 4. Cache (`cache.py`)

Redis-based caching layer with multi-TTL for different data types.

**TTLs:**
- Orderbooks: 5 seconds
- Market prices: 30 seconds
- Market metadata: 5 minutes
- Historical trades: 1 hour

## Performance

### Latency

| Method | Latency | Use Case |
|--------|---------|----------|
| WebSocket | <5ms | Real-time trading, HFT |
| REST | 50-100ms | Batch queries, backtesting |
| Redis Cache | <1ms | Frequent reads, shared state |

### Throughput

- **Message rate:** 1000+ msg/sec
- **Orderbook updates:** <1ms reconstruction
- **VWAP calculation:** <0.1ms

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Polymarket WebSocket Server           │
│        wss://ws-subscriptions-clob.polymarket   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  WebSocketClient       │
        │  - Connection mgmt     │
        │  - Message parsing     │
        │  - Auto-reconnect      │
        └────────┬───────────────┘
                 │
                 ▼
        ┌────────────────────────┐
        │  OrderBookManager      │
        │  - Multi-asset track   │
        │  - Metrics calc        │
        │  - Callbacks           │
        └────┬───────────┬───────┘
             │           │
             ▼           ▼
        ┌────────┐  ┌──────────────┐
        │ Redis  │  │  Your        │
        │ Pub/Sub│  │  Strategy    │
        └────────┘  └──────────────┘
```

## Message Types

### Market Channel

1. **book** - Full L2 orderbook snapshot
2. **price_change** - Incremental orderbook update
3. **last_trade_price** - Trade execution event
4. **tick_size_change** - Minimum tick size update
5. **best_bid_ask** - Top of book update (custom feature)
6. **new_market** - New market created (custom feature)
7. **market_resolved** - Market resolved (custom feature)

### User Channel (Authenticated)

1. **trade** - User trade execution
2. **order** - User order placement/update/cancellation

## Quick Start

### Install Dependencies

```bash
pip install websockets redis
```

### Start Redis

```bash
redis-server
```

### Run Example

```python
from src.data.orderbook_manager import OrderBookManager

# Create manager
manager = OrderBookManager(
    asset_ids=["71321045679252212594626385532706912750332728571942532289631379312455583992563"]
)

# Start
await manager.start()

# Query
book = manager.get_orderbook("<asset-id>")
print(f"Best bid: {book.best_bid}, Best ask: {book.best_ask}")

# Stop
await manager.stop()
```

## Integration with Platforms

The WebSocket layer integrates seamlessly with the platform interface:

```python
from src.platforms.polymarket import PolymarketPlatform

# Enable WebSocket
platform = PolymarketPlatform(use_websocket=True)
await platform.connect()

# Get orderbook (uses WebSocket if available)
orderbook = await platform.get_orderbook("<asset-id>")

# Get metrics
metrics = platform.get_orderbook_metrics("<asset-id>")
vwap = platform.calculate_vwap("<asset-id>", "BUY", 100)
```

## Monitoring

### Dashboard

Real-time terminal dashboard:

```bash
python -m src.data.websocket_monitor
```

### Programmatic

```python
stats = manager.get_stats()
print(f"Connected: {stats['connected']}")
print(f"Avg Latency: {stats['latency_avg_ms']:.2f}ms")
print(f"P95 Latency: {stats['latency_p95_ms']:.2f}ms")
```

## Best Practices

### 1. Connection Management
- Always implement auto-reconnect
- Monitor connection uptime
- Validate orderbook after reconnect

### 2. Latency Optimization
- Deploy close to exchange (same region)
- Process messages asynchronously
- Use efficient JSON parsing (orjson)

### 3. Resource Management
- Limit orderbook depth to what you need
- Trim trade history periodically
- Use Redis TTL for auto-expiry

### 4. Error Handling
- Log all unknown message types
- Validate orderbook integrity
- Alert on stale data (>10s without updates)

## Troubleshooting

### High Latency

**Symptoms:** Avg latency >10ms

**Solutions:**
- Check network latency to exchange
- Profile callback execution time
- Optimize message processing

### Connection Drops

**Symptoms:** Frequent reconnects

**Solutions:**
- Ensure PING loop is running
- Check network stability
- Verify not hitting rate limits

### Missing Updates

**Symptoms:** No orderbook updates

**Solutions:**
- Verify asset ID (use token_id, not condition_id)
- Check subscription list
- Monitor Redis connectivity

## Performance Benchmarks

**Expected performance:**
- Connection latency: <2ms (same region)
- Message latency: <5ms average
- Message throughput: 1000+ msg/sec
- Orderbook reconstruction: <1ms
- VWAP calculation: <0.1ms

**Monitoring thresholds:**
- ✅ Green: <5ms average latency
- ⚠️ Warning: 5-10ms average latency
- 🔴 Alert: >10ms average latency

## Documentation

- [WebSocket API Reference](../../docs/WEBSOCKET_API.md)
- [Integration Guide](../../docs/WEBSOCKET_INTEGRATION.md)
- [Example Code](../../examples/websocket_example.py)

## Future Enhancements

- [ ] Support for additional platforms (Kalshi, Manifold)
- [ ] Historical data recording for backtesting
- [ ] Advanced order flow analytics
- [ ] Multi-exchange aggregation
- [ ] Tick-by-tick replay
