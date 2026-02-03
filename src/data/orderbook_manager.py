"""
PR3DICT: Order Book Manager

High-level manager for real-time order book data.
Integrates WebSocket client with Redis cache and provides
convenient APIs for trading strategies.

Features:
- Multi-asset orderbook tracking
- VWAP calculation
- Spread monitoring
- Liquidity analysis
- Update latency tracking
"""
import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from collections import defaultdict

from .websocket_client import (
    PolymarketWebSocketClient,
    OrderBookSnapshot,
    TradeEvent,
)
from .cache import DataCache

logger = logging.getLogger(__name__)


@dataclass
class OrderBookMetrics:
    """Metrics for an orderbook."""
    asset_id: str
    best_bid: Optional[Decimal]
    best_ask: Optional[Decimal]
    spread: Optional[Decimal]
    spread_bps: Optional[int]  # basis points
    mid_price: Optional[Decimal]
    bid_liquidity_100: Decimal  # Total size in top $100 USDC of bids
    ask_liquidity_100: Decimal  # Total size in top $100 USDC of asks
    vwap_buy_100: Optional[Decimal]  # VWAP for $100 buy
    vwap_sell_100: Optional[Decimal]  # VWAP for $100 sell
    update_latency_ms: float
    last_update: datetime
    
    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "best_bid": str(self.best_bid) if self.best_bid else None,
            "best_ask": str(self.best_ask) if self.best_ask else None,
            "spread": str(self.spread) if self.spread else None,
            "spread_bps": self.spread_bps,
            "mid_price": str(self.mid_price) if self.mid_price else None,
            "bid_liquidity_100": str(self.bid_liquidity_100),
            "ask_liquidity_100": str(self.ask_liquidity_100),
            "vwap_buy_100": str(self.vwap_buy_100) if self.vwap_buy_100 else None,
            "vwap_sell_100": str(self.vwap_sell_100) if self.vwap_sell_100 else None,
            "update_latency_ms": self.update_latency_ms,
            "last_update": self.last_update.isoformat(),
        }


class OrderBookManager:
    """
    Manages real-time orderbook data for multiple assets.
    
    Responsibilities:
    - Subscribe to WebSocket feeds
    - Maintain orderbook snapshots
    - Calculate metrics (spread, VWAP, liquidity)
    - Publish to Redis
    - Provide query API
    """
    
    def __init__(
        self,
        asset_ids: Optional[List[str]] = None,
        redis_url: str = "redis://localhost:6379/0",
        enable_custom_features: bool = True,
    ):
        """
        Initialize orderbook manager.
        
        Args:
            asset_ids: List of asset IDs to track
            redis_url: Redis connection URL
            enable_custom_features: Enable Polymarket custom features
        """
        self.asset_ids = asset_ids or []
        self.redis_url = redis_url
        
        # WebSocket client
        self._ws_client = PolymarketWebSocketClient(
            asset_ids=self.asset_ids,
            redis_url=redis_url,
            enable_custom_features=enable_custom_features,
            on_book_update=self._on_book_update,
            on_trade=self._on_trade,
        )
        
        # Cache
        self._cache = DataCache(redis_url=redis_url)
        
        # Metrics tracking
        self._metrics: Dict[str, OrderBookMetrics] = {}
        self._update_times: Dict[str, float] = defaultdict(float)
        
        # Trade tracking
        self._recent_trades: Dict[str, List[TradeEvent]] = defaultdict(list)
        self._max_trade_history = 100
        
        # Callbacks
        self._book_callbacks: List[Callable] = []
        self._trade_callbacks: List[Callable] = []
        
        # Running state
        self._running = False
    
    async def start(self) -> None:
        """Start the orderbook manager."""
        logger.info("Starting OrderBook Manager...")
        
        # Connect cache
        await self._cache.connect()
        
        # Start WebSocket client
        self._running = True
        asyncio.create_task(self._ws_client.run())
        
        logger.info(f"OrderBook Manager started for {len(self.asset_ids)} assets")
    
    async def stop(self) -> None:
        """Stop the orderbook manager."""
        logger.info("Stopping OrderBook Manager...")
        self._running = False
        
        await self._ws_client.disconnect()
        await self._cache.disconnect()
        
        logger.info("OrderBook Manager stopped")
    
    async def _on_book_update(self, snapshot: OrderBookSnapshot) -> None:
        """Handle orderbook update from WebSocket."""
        try:
            # Calculate metrics
            metrics = self._calculate_metrics(snapshot)
            self._metrics[snapshot.asset_id] = metrics
            
            # Update cache
            await self._cache.set_orderbook(
                market_id=snapshot.market_id,
                platform="polymarket",
                orderbook=snapshot.to_dict(),
            )
            
            # Trigger callbacks
            for callback in self._book_callbacks:
                try:
                    await callback(snapshot, metrics)
                except Exception as e:
                    logger.error(f"Book callback error: {e}")
        
        except Exception as e:
            logger.error(f"Error processing book update: {e}", exc_info=True)
    
    async def _on_trade(self, trade: TradeEvent) -> None:
        """Handle trade event from WebSocket."""
        try:
            # Store recent trade
            trades = self._recent_trades[trade.asset_id]
            trades.append(trade)
            
            # Keep only recent trades
            if len(trades) > self._max_trade_history:
                self._recent_trades[trade.asset_id] = trades[-self._max_trade_history:]
            
            # Trigger callbacks
            for callback in self._trade_callbacks:
                try:
                    await callback(trade)
                except Exception as e:
                    logger.error(f"Trade callback error: {e}")
        
        except Exception as e:
            logger.error(f"Error processing trade: {e}", exc_info=True)
    
    def _calculate_metrics(self, snapshot: OrderBookSnapshot) -> OrderBookMetrics:
        """Calculate orderbook metrics."""
        import time
        
        # Track update latency
        now = time.time()
        last_update_time = self._update_times.get(snapshot.asset_id, now)
        latency_ms = (now - last_update_time) * 1000
        self._update_times[snapshot.asset_id] = now
        
        # Calculate spread in basis points
        spread_bps = None
        if snapshot.spread and snapshot.mid_price and snapshot.mid_price > 0:
            spread_bps = int((snapshot.spread / snapshot.mid_price) * 10000)
        
        # Calculate liquidity depth (total size in top $100 USDC)
        bid_liquidity = self._calculate_liquidity_depth(snapshot.bids, Decimal("100"))
        ask_liquidity = self._calculate_liquidity_depth(snapshot.asks, Decimal("100"))
        
        # VWAP calculations
        vwap_buy = snapshot.calculate_vwap("BUY", Decimal("100"))
        vwap_sell = snapshot.calculate_vwap("SELL", Decimal("100"))
        
        return OrderBookMetrics(
            asset_id=snapshot.asset_id,
            best_bid=snapshot.best_bid,
            best_ask=snapshot.best_ask,
            spread=snapshot.spread,
            spread_bps=spread_bps,
            mid_price=snapshot.mid_price,
            bid_liquidity_100=bid_liquidity,
            ask_liquidity_100=ask_liquidity,
            vwap_buy_100=vwap_buy,
            vwap_sell_100=vwap_sell,
            update_latency_ms=latency_ms,
            last_update=snapshot.timestamp,
        )
    
    def _calculate_liquidity_depth(self, levels, depth_usdc: Decimal) -> Decimal:
        """Calculate total size available within a USDC depth."""
        total_size = Decimal("0")
        remaining = depth_usdc
        
        for level in levels:
            level_cost = level.price * level.size
            
            if level_cost <= remaining:
                total_size += level.size
                remaining -= level_cost
            else:
                # Partial fill
                partial_size = remaining / level.price
                total_size += partial_size
                break
        
        return total_size
    
    # === Public API ===
    
    async def subscribe(self, asset_ids: List[str]) -> None:
        """Subscribe to additional assets."""
        self.asset_ids.extend(asset_ids)
        await self._ws_client.subscribe(asset_ids)
    
    async def unsubscribe(self, asset_ids: List[str]) -> None:
        """Unsubscribe from assets."""
        for asset_id in asset_ids:
            if asset_id in self.asset_ids:
                self.asset_ids.remove(asset_id)
        await self._ws_client.unsubscribe(asset_ids)
    
    def get_orderbook(self, asset_id: str) -> Optional[OrderBookSnapshot]:
        """Get current orderbook for an asset."""
        return self._ws_client.get_orderbook(asset_id)
    
    def get_metrics(self, asset_id: str) -> Optional[OrderBookMetrics]:
        """Get calculated metrics for an asset."""
        return self._metrics.get(asset_id)
    
    def get_recent_trades(self, asset_id: str, limit: int = 10) -> List[TradeEvent]:
        """Get recent trades for an asset."""
        trades = self._recent_trades.get(asset_id, [])
        return trades[-limit:]
    
    def get_best_bid_ask(self, asset_id: str) -> tuple[Optional[Decimal], Optional[Decimal]]:
        """Get best bid and ask for an asset."""
        book = self.get_orderbook(asset_id)
        if book:
            return book.best_bid, book.best_ask
        return None, None
    
    def get_mid_price(self, asset_id: str) -> Optional[Decimal]:
        """Get mid price for an asset."""
        book = self.get_orderbook(asset_id)
        return book.mid_price if book else None
    
    def get_spread(self, asset_id: str) -> Optional[Decimal]:
        """Get spread for an asset."""
        book = self.get_orderbook(asset_id)
        return book.spread if book else None
    
    def calculate_vwap(
        self,
        asset_id: str,
        side: str,
        depth_usdc: Decimal = Decimal("100")
    ) -> Optional[Decimal]:
        """Calculate VWAP for a given depth."""
        book = self.get_orderbook(asset_id)
        if book:
            return book.calculate_vwap(side, depth_usdc)
        return None
    
    def register_book_callback(self, callback: Callable) -> None:
        """
        Register a callback for orderbook updates.
        
        Callback signature: async def callback(snapshot: OrderBookSnapshot, metrics: OrderBookMetrics)
        """
        self._book_callbacks.append(callback)
    
    def register_trade_callback(self, callback: Callable) -> None:
        """
        Register a callback for trade events.
        
        Callback signature: async def callback(trade: TradeEvent)
        """
        self._trade_callbacks.append(callback)
    
    def get_stats(self) -> dict:
        """Get comprehensive statistics."""
        ws_stats = self._ws_client.get_stats()
        
        # Calculate metrics stats
        metrics_list = list(self._metrics.values())
        avg_spread_bps = sum(m.spread_bps for m in metrics_list if m.spread_bps) / len(metrics_list) if metrics_list else 0
        avg_latency = sum(m.update_latency_ms for m in metrics_list) / len(metrics_list) if metrics_list else 0
        
        return {
            **ws_stats,
            "tracked_assets": len(self.asset_ids),
            "metrics_cached": len(self._metrics),
            "avg_spread_bps": avg_spread_bps,
            "avg_update_latency_ms": avg_latency,
            "total_trades_tracked": sum(len(t) for t in self._recent_trades.values()),
        }
    
    def get_all_metrics(self) -> Dict[str, OrderBookMetrics]:
        """Get all current metrics."""
        return self._metrics.copy()
