# Polymarket WebSocket API Documentation

## Overview

The Polymarket WebSocket API provides real-time market data with <5ms latency, replacing REST polling (50-100ms). This gives a significant speed advantage for high-frequency trading strategies.

**WebSocket Endpoint:** `wss://ws-subscriptions-clob.polymarket.com`

## Channels

### 1. Market Channel (`/ws/market`)

Public channel for real-time orderbook and trade data.

**Subscription Message:**
```json
{
  "type": "market",
  "assets_ids": ["<asset-id-1>", "<asset-id-2>"],
  "custom_feature_enabled": true
}
```

**Dynamic Subscribe/Unsubscribe:**
```json
{
  "operation": "subscribe",  // or "unsubscribe"
  "assets_ids": ["<asset-id>"]
}
```

### 2. User Channel (`/ws/user`)

Authenticated channel for user-specific order and trade events.

**Subscription Message:**
```json
{
  "type": "user",
  "markets": ["<condition-id-1>", "<condition-id-2>"],
  "auth": {
    "apiKey": "<api-key>",
    "secret": "<api-secret>",
    "passphrase": "<api-passphrase>"
  }
}
```

## Message Types

### Market Channel Messages

#### 1. `book` - Full Orderbook Snapshot

Emitted when first subscribing to a market or after a trade.

```json
{
  "event_type": "book",
  "asset_id": "65818619657568813474341868652308942079804919287380422192892211131408793125422",
  "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
  "bids": [
    { "price": "0.48", "size": "30" },
    { "price": "0.49", "size": "20" },
    { "price": "0.50", "size": "15" }
  ],
  "asks": [
    { "price": "0.52", "size": "25" },
    { "price": "0.53", "size": "60" },
    { "price": "0.54", "size": "10" }
  ],
  "timestamp": "123456789000",
  "hash": "0x0...."
}
```

**Fields:**
- `asset_id`: Asset ID (token ID) - unique identifier for the outcome token
- `market`: Condition ID - identifies the prediction market
- `bids`: Array of bid levels (descending by price)
- `asks`: Array of ask levels (ascending by price)
- `timestamp`: Unix timestamp in milliseconds
- `hash`: Hash summary of orderbook content

#### 2. `price_change` - Incremental Orderbook Update

Emitted when an order is placed or cancelled.

```json
{
  "event_type": "price_change",
  "market": "0x5f65177b394277fd294cd75650044e32ba009a95022d88a0c1d565897d72f8f1",
  "price_changes": [
    {
      "asset_id": "71321045679252212594626385532706912750332728571942532289631379312455583992563",
      "price": "0.5",
      "size": "200",
      "side": "BUY",
      "hash": "56621a121a47ed9333273e21c83b660cff37ae50",
      "best_bid": "0.5",
      "best_ask": "1"
    }
  ],
  "timestamp": "1757908892351"
}
```

**Fields:**
- `price`: Price level affected
- `size`: New aggregate size at this price (0 = level removed)
- `side`: "BUY" or "SELL"
- `best_bid`/`best_ask`: Current top of book after update

**Reconstruction Logic:**
1. Find the price level in your local orderbook
2. If `size == 0`: Remove the level
3. If `size > 0`: Update the level's size
4. If level doesn't exist: Add new level and resort

#### 3. `last_trade_price` - Trade Event

Emitted when a maker and taker order are matched.

```json
{
  "event_type": "last_trade_price",
  "asset_id": "114122071509644379678018727908709560226618148003371446110114509806601493071694",
  "market": "0x6a67b9d828d53862160e470329ffea5246f338ecfffdf2cab45211ec578b0347",
  "price": "0.456",
  "size": "219.217767",
  "side": "BUY",
  "timestamp": "1750428146322",
  "fee_rate_bps": "0"
}
```

**Fields:**
- `price`: Execution price
- `size`: Trade size (shares)
- `side`: Aggressor side ("BUY" or "SELL")
- `fee_rate_bps`: Fee rate in basis points

#### 4. `tick_size_change` - Minimum Tick Update

Emitted when the minimum tick size changes (typically at price extremes >0.96 or <0.04).

```json
{
  "event_type": "tick_size_change",
  "asset_id": "65818619657568813474341868652308942079804919287380422192892211131408793125422",
  "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
  "old_tick_size": "0.01",
  "new_tick_size": "0.001",
  "timestamp": "100000000"
}
```

#### 5. `best_bid_ask` - Top of Book (Custom Feature)

**Requires:** `custom_feature_enabled: true`

Emitted when the best bid or ask changes.

```json
{
  "event_type": "best_bid_ask",
  "market": "0x0005c0d312de0be897668695bae9f32b624b4a1ae8b140c49f08447fcc74f442",
  "asset_id": "85354956062430465315924116860125388538595433819574542752031640332592237464430",
  "best_bid": "0.73",
  "best_ask": "0.77",
  "spread": "0.04",
  "timestamp": "1766789469958"
}
```

#### 6. `new_market` - Market Creation (Custom Feature)

**Requires:** `custom_feature_enabled: true`

Emitted when a new market is created.

```json
{
  "event_type": "new_market",
  "id": "1031769",
  "question": "Will NVIDIA (NVDA) close above $240 end of January?",
  "market": "0x311d0c4b6671ab54af4970c06fcf58662516f5168997bdda209ec3db5aa6b0c1",
  "slug": "nvda-above-240-on-january-30-2026",
  "description": "...",
  "assets_ids": ["...", "..."],
  "outcomes": ["Yes", "No"],
  "timestamp": "1766790415550"
}
```

#### 7. `market_resolved` - Market Resolution (Custom Feature)

**Requires:** `custom_feature_enabled: true`

Emitted when a market is resolved.

```json
{
  "event_type": "market_resolved",
  "id": "1031769",
  "question": "Will NVIDIA (NVDA) close above $240 end of January?",
  "market": "0x311d0c4b6671ab54af4970c06fcf58662516f5168997bdda209ec3db5aa6b0c1",
  "winning_asset_id": "76043073756653678226373981964075571318267289248134717369284518995922789326425",
  "winning_outcome": "Yes",
  "timestamp": "1766790415550"
}
```

### User Channel Messages

#### 1. `trade` - Trade Execution

Emitted for:
- Market order matched ("MATCHED")
- Limit order included in trade ("MATCHED")
- Subsequent status changes ("MINED", "CONFIRMED", "RETRYING", "FAILED")

```json
{
  "event_type": "trade",
  "id": "28c4d2eb-bbea-40e7-a9f0-b2fdb56b2c2e",
  "asset_id": "52114319501245915516055106046884209969926127482827954674443846427813813222426",
  "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
  "price": "0.57",
  "size": "10",
  "side": "BUY",
  "status": "MATCHED",
  "timestamp": "1672290701",
  "maker_orders": [
    {
      "order_id": "0xff354cd7ca7539dfa9c28d90943ab5779a4eac34b9b37a757d7b32bdfb11790b",
      "price": "0.57",
      "matched_amount": "10"
    }
  ]
}
```

**Status Progression:**
1. `MATCHED` - Order matched on CLOB
2. `MINED` - Transaction mined on blockchain
3. `CONFIRMED` - Transaction confirmed (final)
4. `RETRYING` - Retry in progress
5. `FAILED` - Transaction failed

#### 2. `order` - Order Updates

Emitted for:
- Order placed (`PLACEMENT`)
- Order partially filled (`UPDATE`)
- Order cancelled (`CANCELLATION`)

```json
{
  "event_type": "order",
  "id": "0xff354cd7ca7539dfa9c28d90943ab5779a4eac34b9b37a757d7b32bdfb11790b",
  "type": "PLACEMENT",
  "asset_id": "52114319501245915516055106046884209969926127482827954674443846427813813222426",
  "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
  "price": "0.57",
  "original_size": "10",
  "size_matched": "0",
  "side": "SELL",
  "timestamp": "1672290687"
}
```

## Connection Management

### Heartbeat (PING/PONG)

The server requires periodic PING messages to keep the connection alive.

**Send:** `"PING"` (as plain text, not JSON)
**Receive:** `"PONG"` (as plain text)

**Recommended interval:** Every 10 seconds

### Reconnection Strategy

Implement exponential backoff for reconnections:
1. 1 second
2. 2 seconds
3. 5 seconds
4. 10 seconds
5. 30 seconds
6. 60 seconds (max)

### Error Handling

- **Connection closed:** Reconnect with backoff
- **Invalid JSON:** Log and skip message
- **Missing `event_type`:** Ignore message
- **Unknown event type:** Log for future support

## Data Processing

### Orderbook Reconstruction

**Initial State:**
```python
# Receive 'book' message
orderbook = {
    'bids': [(price, size), ...],  # sorted descending
    'asks': [(price, size), ...],  # sorted ascending
}
```

**Incremental Updates:**
```python
# Receive 'price_change' message
for change in price_changes:
    price = change['price']
    size = change['size']
    side = change['side']  # 'BUY' or 'SELL'
    
    levels = orderbook['bids'] if side == 'BUY' else orderbook['asks']
    
    # Find level
    for i, (p, s) in enumerate(levels):
        if p == price:
            if size == 0:
                # Remove level
                del levels[i]
            else:
                # Update size
                levels[i] = (price, size)
            break
    else:
        # Add new level
        if size > 0:
            levels.append((price, size))
            # Resort (bids desc, asks asc)
            levels.sort(reverse=(side == 'BUY'))
```

### VWAP Calculation

**Volume-Weighted Average Price** for a given depth:

```python
def calculate_vwap(levels, depth_usdc):
    total_cost = 0
    total_size = 0
    remaining = depth_usdc
    
    for price, size in levels:
        level_cost = price * size
        
        if level_cost <= remaining:
            total_cost += level_cost
            total_size += size
            remaining -= level_cost
        else:
            # Partial fill
            partial_size = remaining / price
            total_cost += remaining
            total_size += partial_size
            break
    
    return total_cost / total_size if total_size > 0 else None
```

### Latency Monitoring

Track message latency to ensure real-time performance:

```python
receive_time = time.time()
msg_timestamp = int(message['timestamp']) / 1000  # Convert to seconds
latency_ms = (receive_time - msg_timestamp) * 1000

# Target: <5ms average
# Warning: >10ms
# Alert: >50ms
```

## Integration with PR3DICT

### Using the WebSocket Client

```python
from src.data.websocket_client import PolymarketWebSocketClient
from src.data.orderbook_manager import OrderBookManager

# Create manager
manager = OrderBookManager(
    asset_ids=["71321045679252212594626385532706912750332728571942532289631379312455583992563"],
    redis_url="redis://localhost:6379/0",
    enable_custom_features=True
)

# Start real-time feeds
await manager.start()

# Query orderbook
book = manager.get_orderbook(asset_id)
print(f"Best bid: {book.best_bid}, Best ask: {book.best_ask}")

# Calculate VWAP
vwap = manager.calculate_vwap(asset_id, side="BUY", depth_usdc=100)
print(f"VWAP for $100 buy: {vwap}")

# Get metrics
metrics = manager.get_metrics(asset_id)
print(f"Spread: {metrics.spread_bps} bps")
print(f"Update latency: {metrics.update_latency_ms:.2f}ms")

# Stop when done
await manager.stop()
```

### Redis Pub/Sub Integration

The WebSocket client publishes to Redis channels:

**Orderbook Updates:**
- Channel: `polymarket:orderbook:<asset_id>`
- TTL Cache: `orderbook:polymarket:<asset_id>` (5 seconds)

**Trade Events:**
- Channel: `polymarket:trade:<asset_id>`

**Subscribing in Strategy:**
```python
import redis.asyncio as redis

r = await redis.from_url("redis://localhost:6379/0")
pubsub = r.pubsub()

await pubsub.subscribe("polymarket:orderbook:71321045679252212594626385532706912750332728571942532289631379312455583992563")

async for message in pubsub.listen():
    if message['type'] == 'message':
        data = json.loads(message['data'])
        print(f"Orderbook update: {data['best_bid']} / {data['best_ask']}")
```

## Best Practices

### 1. Connection Resilience
- Always implement auto-reconnect with exponential backoff
- Track connection uptime and alert on frequent disconnects
- Validate orderbook integrity after reconnect (request full snapshot)

### 2. Message Processing
- Process messages asynchronously to avoid blocking
- Use a message queue if processing is slow
- Log all unknown message types for future support

### 3. Latency Optimization
- Run WebSocket client close to exchange (same region/cloud)
- Use dedicated connection (don't share with other services)
- Monitor end-to-end latency (WebSocket → your strategy)

### 4. Data Validation
- Verify orderbook integrity (bids < asks)
- Check for crossed markets (arbitrage detection)
- Alert on stale data (no updates for >10 seconds)

### 5. Resource Management
- Limit orderbook depth to what you need (top 20 levels)
- Trim trade history (keep only recent 100 trades per asset)
- Use Redis TTL to auto-expire stale cache entries

## Troubleshooting

### Issue: Connection keeps dropping

**Possible causes:**
- Missing PING messages → Ensure ping loop is running
- Network instability → Check network latency
- Geographic restrictions → Verify IP not blocked

**Solution:**
```python
# Implement ping loop
async def ping_loop(ws):
    while True:
        await asyncio.sleep(10)
        await ws.send("PING")
```

### Issue: High latency (>50ms)

**Possible causes:**
- Client location far from exchange
- Network congestion
- Processing bottleneck

**Solution:**
- Measure latency at each stage (network, parsing, processing)
- Move client to same cloud region as exchange
- Optimize message parsing (use orjson instead of json)

### Issue: Orderbook state divergence

**Possible causes:**
- Missed `price_change` messages
- Incorrect reconstruction logic
- Out-of-order message processing

**Solution:**
- Validate orderbook hash against server
- Request full `book` snapshot periodically
- Ensure messages processed in order (use asyncio.Queue)

### Issue: Memory leak from trade history

**Possible causes:**
- Unbounded trade list growth
- Not clearing old orderbook snapshots

**Solution:**
```python
# Limit trade history per asset
MAX_TRADES = 100
trades[asset_id] = trades[asset_id][-MAX_TRADES:]

# Clear inactive orderbooks
if time.time() - book.timestamp > 3600:
    del orderbooks[asset_id]
```

## Performance Benchmarks

**Expected Performance:**
- Connection latency: <2ms (same region)
- Message latency: <5ms average
- Message throughput: 1000+ msg/sec
- Orderbook reconstruction: <1ms
- VWAP calculation: <0.1ms

**Monitoring Thresholds:**
- ✅ Green: <5ms average latency
- ⚠️ Warning: 5-10ms average latency
- 🔴 Alert: >10ms average latency or >30s since last message

## Additional Resources

- [Polymarket WebSocket Docs](https://docs.polymarket.com/developers/CLOB/websocket/wss-overview)
- [py-clob-client GitHub](https://github.com/Polymarket/py-clob-client)
- [Gamma Markets API](https://docs.polymarket.com/developers/gamma-markets-api/overview)
