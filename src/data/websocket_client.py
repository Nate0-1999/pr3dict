"""
PR3DICT: Polymarket WebSocket Real-Time Data Client

High-performance WebSocket client for <5ms latency market data.
Replaces REST polling (50-100ms) with real-time order book streams.

Features:
- Auto-reconnect with exponential backoff
- L2 orderbook reconstruction
- Real-time trade stream
- VWAP calculation from depth
- Latency monitoring
- Redis integration for pub/sub
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Dict, List, Callable, Set
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: Decimal
    size: Decimal
    
    def __hash__(self):
        return hash((self.price, self.size))


@dataclass
class OrderBookSnapshot:
    """L2 order book snapshot."""
    asset_id: str
    market_id: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hash: Optional[str] = None
    
    @property
    def best_bid(self) -> Optional[Decimal]:
        """Top of book bid."""
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[Decimal]:
        """Top of book ask."""
        return self.asks[0].price if self.asks else None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    def calculate_vwap(self, side: str, depth_usdc: Decimal = Decimal("100")) -> Optional[Decimal]:
        """
        Calculate VWAP for a given USDC depth.
        
        Args:
            side: "BUY" or "SELL"
            depth_usdc: USDC amount to calculate VWAP over
        
        Returns:
            VWAP price or None if insufficient liquidity
        """
        levels = self.asks if side == "BUY" else self.bids
        
        total_cost = Decimal("0")
        total_size = Decimal("0")
        remaining = depth_usdc
        
        for level in levels:
            # Cost for this level (price * size, but capped by remaining)
            level_cost = level.price * level.size
            
            if level_cost <= remaining:
                total_cost += level_cost
                total_size += level.size
                remaining -= level_cost
            else:
                # Partial fill on this level
                partial_size = remaining / level.price
                total_cost += remaining
                total_size += partial_size
                remaining = Decimal("0")
                break
        
        if total_size > 0:
            return total_cost / total_size
        return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for caching/serialization."""
        return {
            "asset_id": self.asset_id,
            "market_id": self.market_id,
            "bids": [[str(b.price), str(b.size)] for b in self.bids],
            "asks": [[str(a.price), str(a.size)] for a in self.asks],
            "timestamp": self.timestamp.isoformat(),
            "hash": self.hash,
            "best_bid": str(self.best_bid) if self.best_bid else None,
            "best_ask": str(self.best_ask) if self.best_ask else None,
            "spread": str(self.spread) if self.spread else None,
            "mid_price": str(self.mid_price) if self.mid_price else None,
        }


@dataclass
class TradeEvent:
    """Real-time trade event."""
    asset_id: str
    market_id: str
    price: Decimal
    size: Decimal
    side: str  # "BUY" or "SELL"
    timestamp: datetime
    fee_rate_bps: str = "0"
    
    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "market_id": self.market_id,
            "price": str(self.price),
            "size": str(self.size),
            "side": self.side,
            "timestamp": self.timestamp.isoformat(),
            "fee_rate_bps": self.fee_rate_bps,
        }


class PolymarketWebSocketClient:
    """
    Async WebSocket client for Polymarket real-time data feeds.
    
    Subscribes to market channel for orderbook and trade updates.
    Maintains local orderbook state with incremental updates.
    Publishes updates to Redis for downstream consumers.
    """
    
    WS_URL = "wss://ws-subscriptions-clob.polymarket.com"
    PING_INTERVAL = 10  # seconds
    RECONNECT_DELAYS = [1, 2, 5, 10, 30, 60]  # exponential backoff
    
    def __init__(
        self,
        asset_ids: Optional[List[str]] = None,
        redis_url: str = "redis://localhost:6379/0",
        enable_custom_features: bool = True,
        on_book_update: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
    ):
        """
        Initialize WebSocket client.
        
        Args:
            asset_ids: List of asset IDs (token IDs) to subscribe to
            redis_url: Redis connection URL for pub/sub
            enable_custom_features: Enable best_bid_ask, new_market events
            on_book_update: Callback for orderbook updates
            on_trade: Callback for trade events
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library required: pip install websockets")
        
        self.asset_ids: Set[str] = set(asset_ids or [])
        self.redis_url = redis_url
        self.enable_custom_features = enable_custom_features
        self.on_book_update = on_book_update
        self.on_trade = on_trade
        
        # Connection state
        self._ws: Optional[WebSocketClientProtocol] = None
        self._redis: Optional[redis.Redis] = None
        self._connected = False
        self._running = False
        self._reconnect_attempt = 0
        
        # Orderbook state (asset_id -> OrderBookSnapshot)
        self._orderbooks: Dict[str, OrderBookSnapshot] = {}
        
        # Latency tracking
        self._last_message_time = time.time()
        self._message_latencies: List[float] = []
        self._max_latency_samples = 1000
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
    
    async def connect(self) -> bool:
        """Connect to WebSocket and Redis."""
        try:
            # Connect to Redis
            if REDIS_AVAILABLE and not self._redis:
                self._redis = await redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self._redis.ping()
                logger.info("Connected to Redis for WebSocket pub/sub")
            
            # Connect to WebSocket
            self._ws = await websockets.connect(
                f"{self.WS_URL}/ws/market",
                ping_interval=None,  # Manual ping
            )
            
            # Send subscription message
            subscribe_msg = {
                "type": "market",
                "assets_ids": list(self.asset_ids),
                "custom_feature_enabled": self.enable_custom_features,
            }
            await self._ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {len(self.asset_ids)} assets")
            
            self._connected = True
            self._reconnect_attempt = 0
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        """Gracefully disconnect."""
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        # Close connections
        if self._ws:
            await self._ws.close()
            self._ws = None
        
        if self._redis:
            await self._redis.close()
            self._redis = None
        
        self._connected = False
        logger.info("WebSocket client disconnected")
    
    async def run(self) -> None:
        """Main run loop with auto-reconnect."""
        self._running = True
        
        while self._running:
            try:
                # Connect
                if not self._connected:
                    connected = await self.connect()
                    if not connected:
                        await self._handle_reconnect()
                        continue
                
                # Start background tasks
                self._tasks = [
                    asyncio.create_task(self._ping_loop()),
                    asyncio.create_task(self._message_loop()),
                ]
                
                # Wait for tasks
                await asyncio.gather(*self._tasks, return_exceptions=True)
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._connected = False
                await self._handle_reconnect()
    
    async def _message_loop(self) -> None:
        """Process incoming WebSocket messages."""
        try:
            async for message in self._ws:
                if message == "PONG":
                    continue
                
                # Track latency
                receive_time = time.time()
                
                try:
                    data = json.loads(message)
                    await self._handle_message(data, receive_time)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Message handler error: {e}", exc_info=True)
        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self._connected = False
        except Exception as e:
            logger.error(f"Message loop error: {e}")
            self._connected = False
    
    async def _handle_message(self, data: dict, receive_time: float) -> None:
        """
        Handle WebSocket message by type.
        
        Message types:
        - book: Full L2 orderbook snapshot
        - price_change: Incremental orderbook update
        - last_trade_price: Trade event
        - tick_size_change: Tick size update
        - best_bid_ask: Top of book (custom feature)
        - new_market: New market created (custom feature)
        - market_resolved: Market resolved (custom feature)
        """
        event_type = data.get("event_type")
        
        if not event_type:
            return
        
        # Calculate message latency
        msg_timestamp = int(data.get("timestamp", receive_time * 1000))
        latency_ms = (receive_time * 1000) - msg_timestamp
        self._track_latency(latency_ms)
        
        # Route to handler
        if event_type == "book":
            await self._handle_book(data)
        elif event_type == "price_change":
            await self._handle_price_change(data)
        elif event_type == "last_trade_price":
            await self._handle_trade(data)
        elif event_type == "tick_size_change":
            logger.info(f"Tick size changed: {data}")
        elif event_type == "best_bid_ask":
            await self._handle_best_bid_ask(data)
        elif event_type == "new_market":
            logger.info(f"New market created: {data.get('question')}")
        elif event_type == "market_resolved":
            logger.info(f"Market resolved: {data.get('question')} -> {data.get('winning_outcome')}")
        else:
            logger.debug(f"Unknown event type: {event_type}")
    
    async def _handle_book(self, data: dict) -> None:
        """Handle full orderbook snapshot."""
        asset_id = data.get("asset_id")
        market_id = data.get("market")
        
        if not asset_id or not market_id:
            return
        
        # Parse orderbook
        bids = [
            OrderBookLevel(Decimal(level["price"]), Decimal(level["size"]))
            for level in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(Decimal(level["price"]), Decimal(level["size"]))
            for level in data.get("asks", [])
        ]
        
        snapshot = OrderBookSnapshot(
            asset_id=asset_id,
            market_id=market_id,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(timezone.utc),
            hash=data.get("hash"),
        )
        
        # Update local state
        self._orderbooks[asset_id] = snapshot
        
        # Callback
        if self.on_book_update:
            try:
                await self.on_book_update(snapshot)
            except Exception as e:
                logger.error(f"Book update callback error: {e}")
        
        # Publish to Redis
        await self._publish_orderbook(snapshot)
        
        logger.debug(f"Book snapshot {asset_id}: bid={snapshot.best_bid} ask={snapshot.best_ask}")
    
    async def _handle_price_change(self, data: dict) -> None:
        """Handle incremental orderbook update."""
        market_id = data.get("market")
        price_changes = data.get("price_changes", [])
        
        for change in price_changes:
            asset_id = change.get("asset_id")
            
            if asset_id not in self._orderbooks:
                # No snapshot yet, ignore update
                continue
            
            book = self._orderbooks[asset_id]
            price = Decimal(change["price"])
            size = Decimal(change["size"])
            side = change["side"]  # "BUY" or "SELL"
            
            # Update the appropriate side
            levels = book.bids if side == "BUY" else book.asks
            
            # Find and update level
            updated = False
            for i, level in enumerate(levels):
                if level.price == price:
                    if size == 0:
                        # Remove level
                        levels.pop(i)
                    else:
                        # Update size
                        level.size = size
                    updated = True
                    break
            
            # Add new level if not found
            if not updated and size > 0:
                levels.append(OrderBookLevel(price, size))
                # Resort (bids descending, asks ascending)
                if side == "BUY":
                    levels.sort(key=lambda x: x.price, reverse=True)
                else:
                    levels.sort(key=lambda x: x.price)
            
            book.timestamp = datetime.now(timezone.utc)
            
            # Callback
            if self.on_book_update:
                try:
                    await self.on_book_update(book)
                except Exception as e:
                    logger.error(f"Book update callback error: {e}")
            
            # Publish to Redis
            await self._publish_orderbook(book)
    
    async def _handle_trade(self, data: dict) -> None:
        """Handle trade event."""
        asset_id = data.get("asset_id")
        market_id = data.get("market")
        
        if not asset_id or not market_id:
            return
        
        trade = TradeEvent(
            asset_id=asset_id,
            market_id=market_id,
            price=Decimal(data["price"]),
            size=Decimal(data["size"]),
            side=data["side"],
            timestamp=datetime.fromtimestamp(int(data["timestamp"]) / 1000, tz=timezone.utc),
            fee_rate_bps=data.get("fee_rate_bps", "0"),
        )
        
        # Callback
        if self.on_trade:
            try:
                await self.on_trade(trade)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")
        
        # Publish to Redis
        await self._publish_trade(trade)
        
        logger.debug(f"Trade {asset_id}: {trade.side} {trade.size} @ {trade.price}")
    
    async def _handle_best_bid_ask(self, data: dict) -> None:
        """Handle best bid/ask update (custom feature)."""
        asset_id = data.get("asset_id")
        
        if asset_id in self._orderbooks:
            book = self._orderbooks[asset_id]
            # Update top of book if we have full snapshot
            logger.debug(f"BBA {asset_id}: {data.get('best_bid')}/{data.get('best_ask')}")
    
    async def _publish_orderbook(self, snapshot: OrderBookSnapshot) -> None:
        """Publish orderbook to Redis."""
        if not self._redis:
            return
        
        try:
            # Publish to channel
            channel = f"polymarket:orderbook:{snapshot.asset_id}"
            await self._redis.publish(channel, json.dumps(snapshot.to_dict()))
            
            # Also cache in Redis with TTL
            key = f"orderbook:polymarket:{snapshot.asset_id}"
            await self._redis.setex(key, 5, json.dumps(snapshot.to_dict()))
            
        except Exception as e:
            logger.warning(f"Redis publish failed: {e}")
    
    async def _publish_trade(self, trade: TradeEvent) -> None:
        """Publish trade to Redis."""
        if not self._redis:
            return
        
        try:
            channel = f"polymarket:trade:{trade.asset_id}"
            await self._redis.publish(channel, json.dumps(trade.to_dict()))
        except Exception as e:
            logger.warning(f"Redis trade publish failed: {e}")
    
    async def _ping_loop(self) -> None:
        """Send PING messages to keep connection alive."""
        while self._connected:
            try:
                await asyncio.sleep(self.PING_INTERVAL)
                if self._ws and self._connected:
                    await self._ws.send("PING")
            except Exception as e:
                logger.error(f"Ping failed: {e}")
                self._connected = False
                break
    
    async def _handle_reconnect(self) -> None:
        """Handle reconnection with exponential backoff."""
        if self._reconnect_attempt >= len(self.RECONNECT_DELAYS):
            delay = self.RECONNECT_DELAYS[-1]
        else:
            delay = self.RECONNECT_DELAYS[self._reconnect_attempt]
        
        self._reconnect_attempt += 1
        logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_attempt})...")
        await asyncio.sleep(delay)
    
    def _track_latency(self, latency_ms: float) -> None:
        """Track message latency for monitoring."""
        self._message_latencies.append(latency_ms)
        
        # Keep only recent samples
        if len(self._message_latencies) > self._max_latency_samples:
            self._message_latencies = self._message_latencies[-self._max_latency_samples:]
        
        self._last_message_time = time.time()
    
    async def subscribe(self, asset_ids: List[str]) -> None:
        """Dynamically subscribe to additional asset IDs."""
        if not self._ws or not self._connected:
            raise RuntimeError("Not connected")
        
        self.asset_ids.update(asset_ids)
        
        msg = {
            "operation": "subscribe",
            "assets_ids": asset_ids,
            "custom_feature_enabled": self.enable_custom_features,
        }
        await self._ws.send(json.dumps(msg))
        logger.info(f"Subscribed to {len(asset_ids)} additional assets")
    
    async def unsubscribe(self, asset_ids: List[str]) -> None:
        """Dynamically unsubscribe from asset IDs."""
        if not self._ws or not self._connected:
            raise RuntimeError("Not connected")
        
        self.asset_ids.difference_update(asset_ids)
        
        msg = {
            "operation": "unsubscribe",
            "assets_ids": asset_ids,
        }
        await self._ws.send(json.dumps(msg))
        logger.info(f"Unsubscribed from {len(asset_ids)} assets")
    
    def get_orderbook(self, asset_id: str) -> Optional[OrderBookSnapshot]:
        """Get current orderbook snapshot for an asset."""
        return self._orderbooks.get(asset_id)
    
    def get_stats(self) -> dict:
        """Get connection and performance statistics."""
        latencies = self._message_latencies[-100:] if self._message_latencies else []
        
        return {
            "connected": self._connected,
            "subscribed_assets": len(self.asset_ids),
            "orderbooks_cached": len(self._orderbooks),
            "reconnect_attempts": self._reconnect_attempt,
            "latency_avg_ms": sum(latencies) / len(latencies) if latencies else 0,
            "latency_p50_ms": sorted(latencies)[len(latencies)//2] if latencies else 0,
            "latency_p95_ms": sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
            "latency_max_ms": max(latencies) if latencies else 0,
            "last_message_ago_s": time.time() - self._last_message_time,
        }
