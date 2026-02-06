"""
PR3DICT: Polymarket Platform Integration

Blockchain-based prediction market on Polygon network.
Uses USDC for settlement, py-clob-client for API access.

Enhanced with WebSocket support for <5ms latency (vs 50-100ms REST polling).
"""
import os
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, List
import logging

from .base import (
    PlatformInterface, 
    Market, Order, Position, OrderBook,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)

# Lazy import to avoid dependency issues if not using Polymarket
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, ApiCreds
    POLYMARKET_AVAILABLE = True
except ImportError:
    POLYMARKET_AVAILABLE = False
    logger.warning("py-clob-client not installed. Polymarket integration unavailable.")

# WebSocket support (optional for real-time data)
try:
    from ..data.orderbook_manager import OrderBookManager
    from ..data.websocket_client import OrderBookSnapshot
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.info("WebSocket client not available - using REST only")


class PolymarketPlatform(PlatformInterface):
    """
    Polymarket CLOB API integration with WebSocket support.
    
    Data Sources:
    - REST API: For account data, order placement, market discovery
    - WebSocket: For real-time orderbook and trade data (<5ms latency)
    
    Requires:
    - Polygon wallet private key
    - API credentials (key/secret/passphrase)
    
    WebSocket Features (optional):
    - Real-time L2 orderbook updates
    - Trade stream
    - VWAP calculation
    - <5ms latency vs 50-100ms REST polling
    """
    
    CLOB_URL = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"
    CHAIN_ID = 137  # Polygon Mainnet
    
    def __init__(self,
                 private_key: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 passphrase: Optional[str] = None,
                 use_websocket: bool = True,
                 redis_url: str = "redis://localhost:6379/0"):
        if not POLYMARKET_AVAILABLE:
            raise ImportError("py-clob-client is required for Polymarket. pip install py-clob-client")
        
        self.private_key = private_key or os.getenv("POLYMARKET_PRIVATE_KEY")
        self.api_key = api_key or os.getenv("POLYMARKET_API_KEY")
        self.api_secret = api_secret or os.getenv("POLYMARKET_API_SECRET")
        self.passphrase = passphrase or os.getenv("POLYMARKET_PASSPHRASE")
        
        self._client: Optional[ClobClient] = None
        
        # WebSocket support (optional)
        self._use_websocket = use_websocket and WEBSOCKET_AVAILABLE
        self._orderbook_manager: Optional[OrderBookManager] = None
        self._redis_url = redis_url
        self._tracked_assets: set = set()
        
        if use_websocket and not WEBSOCKET_AVAILABLE:
            logger.warning("WebSocket requested but not available - falling back to REST")
    
    @property
    def name(self) -> str:
        return "polymarket"
    
    async def connect(self) -> bool:
        """Initialize the CLOB client and WebSocket feeds."""
        try:
            # Connect REST API
            creds = ApiCreds(
                api_key=self.api_key,
                api_secret=self.api_secret,
                api_passphrase=self.passphrase
            )
            
            self._client = ClobClient(
                self.CLOB_URL,
                key=self.private_key,
                chain_id=self.CHAIN_ID,
                creds=creds
            )
            
            # Verify connection
            await asyncio.to_thread(self._client.get_markets)
            
            logger.info("Connected to Polymarket CLOB (REST)")
            
            # Start WebSocket feeds (if enabled)
            if self._use_websocket:
                try:
                    self._orderbook_manager = OrderBookManager(
                        asset_ids=list(self._tracked_assets),
                        redis_url=self._redis_url,
                        enable_custom_features=True
                    )
                    await self._orderbook_manager.start()
                    logger.info("Connected to Polymarket WebSocket (real-time data)")
                except Exception as e:
                    logger.warning(f"WebSocket connection failed, using REST only: {e}")
                    self._orderbook_manager = None
            
            return True
            
        except Exception as e:
            logger.error(f"Polymarket connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Clean up client reference and WebSocket connections."""
        if self._orderbook_manager:
            await self._orderbook_manager.stop()
            self._orderbook_manager = None
        
        self._client = None
        logger.info("Disconnected from Polymarket")
    
    # --- Account ---
    
    async def get_balance(self) -> Decimal:
        """Get USDC balance."""
        # Polymarket uses on-chain USDC, balance check requires web3
        # For now, return balance from API if available
        try:
            balance_info = await asyncio.to_thread(
                self._client.get_balance_allowance
            )
            return Decimal(str(balance_info.get("balance", 0)))
        except Exception as e:
            logger.warning(f"Balance check failed: {e}")
            return Decimal("0")
    
    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        try:
            positions_data = await asyncio.to_thread(
                self._client.get_positions
            )
            
            positions = []
            for pos in positions_data:
                positions.append(Position(
                    market_id=pos.get("asset_id", ""),
                    ticker=pos.get("market", {}).get("condition_id", ""),
                    side=OrderSide.YES if pos.get("side") == "YES" else OrderSide.NO,
                    quantity=int(pos.get("size", 0)),
                    avg_price=Decimal(str(pos.get("avg_price", 0))),
                    current_price=Decimal(str(pos.get("cur_price", 0))),
                    unrealized_pnl=Decimal(str(pos.get("pnl", 0))),
                    platform=self.name
                ))
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    # --- Market Data ---
    
    async def get_markets(self,
                          status: str = "open",
                          category: Optional[str] = None,
                          limit: int = 100) -> List[Market]:
        """Fetch markets from Gamma API."""
        try:
            # Use gamma API for market discovery
            import httpx
            async with httpx.AsyncClient() as client:
                params = {"limit": limit, "active": status == "open"}
                if category:
                    params["tag"] = category
                
                response = await client.get(
                    f"{self.GAMMA_URL}/markets",
                    params=params
                )
                response.raise_for_status()
                
                markets = []
                for m in response.json():
                    markets.append(self._parse_market(m))
                return markets
                
        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            return []
    
    async def get_market(self, market_id: str) -> Optional[Market]:
        """Get a single market by condition_id."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.GAMMA_URL}/markets/{market_id}"
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return self._parse_market(response.json())
                
        except Exception as e:
            logger.error(f"Failed to get market {market_id}: {e}")
            return None
    
    def _parse_market(self, m: dict) -> Market:
        """Convert Polymarket market response to Market dataclass."""
        # Polymarket has YES and NO tokens with separate prices
        yes_price = Decimal(str(m.get("outcomePrices", ["0.5", "0.5"])[0]))
        no_price = Decimal(str(m.get("outcomePrices", ["0.5", "0.5"])[1]))
        
        return Market(
            id=m.get("conditionId", m.get("id", "")),
            ticker=m.get("slug", m.get("conditionId", "")),
            title=m.get("question", ""),
            description=m.get("description", ""),
            yes_price=yes_price,
            no_price=no_price,
            volume=Decimal(str(m.get("volume", 0))),
            liquidity=Decimal(str(m.get("liquidity", 0))),
            close_time=datetime.fromisoformat(m.get("endDate", "2030-01-01T00:00:00Z").replace("Z", "+00:00")),
            resolved=m.get("resolved", False),
            platform=self.name
        )
    
    async def get_orderbook(self, market_id: str) -> OrderBook:
        """
        Get order book from CLOB.
        
        Uses WebSocket data if available (<5ms latency), falls back to REST.
        market_id is actually the asset_id (token ID) in Polymarket terminology.
        """
        # Try WebSocket first (real-time, <5ms)
        if self._orderbook_manager:
            ws_book = self._orderbook_manager.get_orderbook(market_id)
            if ws_book:
                # Convert WebSocket snapshot to OrderBook
                bids = [(level.price, int(level.size)) for level in ws_book.bids]
                asks = [(level.price, int(level.size)) for level in ws_book.asks]
                
                return OrderBook(
                    market_id=ws_book.market_id,
                    bids=bids,
                    asks=asks,
                    timestamp=ws_book.timestamp
                )
            else:
                # Not subscribed yet, subscribe now
                await self._subscribe_to_asset(market_id)
        
        # Fallback to REST API (50-100ms latency)
        try:
            book = await asyncio.to_thread(
                self._client.get_order_book,
                market_id
            )
            
            bids = [(Decimal(str(b["price"])), int(b["size"])) for b in book.get("bids", [])]
            asks = [(Decimal(str(a["price"])), int(a["size"])) for a in book.get("asks", [])]
            
            return OrderBook(
                market_id=market_id,
                bids=bids,
                asks=asks,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Failed to get orderbook: {e}")
            return OrderBook(market_id=market_id, bids=[], asks=[], timestamp=datetime.now(timezone.utc))
    
    async def _subscribe_to_asset(self, asset_id: str) -> None:
        """Subscribe to WebSocket feed for an asset."""
        if self._orderbook_manager and asset_id not in self._tracked_assets:
            self._tracked_assets.add(asset_id)
            try:
                await self._orderbook_manager.subscribe([asset_id])
                logger.info(f"Subscribed to WebSocket feed for {asset_id[:12]}...")
            except Exception as e:
                logger.warning(f"Failed to subscribe to WebSocket: {e}")
    
    def get_orderbook_metrics(self, market_id: str) -> Optional[dict]:
        """
        Get real-time orderbook metrics (WebSocket only).
        
        Returns:
            Dict with spread, VWAP, liquidity metrics, or None if WebSocket not available
        """
        if not self._orderbook_manager:
            return None
        
        metrics = self._orderbook_manager.get_metrics(market_id)
        return metrics.to_dict() if metrics else None
    
    def calculate_vwap(self, market_id: str, side: str, depth_usdc: Decimal = Decimal("100")) -> Optional[Decimal]:
        """
        Calculate VWAP from real-time orderbook (WebSocket only).
        
        Args:
            market_id: Asset ID (token ID)
            side: "BUY" or "SELL"
            depth_usdc: Depth in USDC to calculate over
        
        Returns:
            VWAP price or None
        """
        if not self._orderbook_manager:
            return None
        
        return self._orderbook_manager.calculate_vwap(market_id, side, depth_usdc)
    
    def get_websocket_stats(self) -> Optional[dict]:
        """Get WebSocket connection and performance statistics."""
        if not self._orderbook_manager:
            return None
        
        return self._orderbook_manager.get_stats()
    
    # --- Orders ---
    
    async def place_order(self,
                          market_id: str,
                          side: OrderSide,
                          order_type: OrderType,
                          quantity: int,
                          price: Optional[Decimal] = None) -> Order:
        """Place order via CLOB."""
        try:
            order_args = OrderArgs(
                token_id=market_id,
                side="BUY" if side == OrderSide.YES else "SELL",
                size=quantity,
                price=float(price) if price else None
            )
            
            if order_type == OrderType.MARKET:
                result = await asyncio.to_thread(
                    self._client.create_and_post_market_order,
                    order_args
                )
            else:
                result = await asyncio.to_thread(
                    self._client.create_and_post_order,
                    order_args
                )
            
            return Order(
                id=result.get("orderID", ""),
                market_id=market_id,
                side=side,
                order_type=order_type,
                price=price,
                quantity=quantity,
                filled_quantity=0,
                status=OrderStatus.PENDING,
                created_at=datetime.now(timezone.utc),
                platform=self.name
            )
            
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            await asyncio.to_thread(
                self._client.cancel,
                order_id
            )
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False
    
    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders."""
        try:
            orders_data = await asyncio.to_thread(
                self._client.get_orders
            )
            
            orders = []
            for o in orders_data:
                order_status = OrderStatus.OPEN
                if o.get("status") == "MATCHED":
                    order_status = OrderStatus.FILLED
                elif o.get("status") == "CANCELLED":
                    order_status = OrderStatus.CANCELLED
                
                if status and order_status != status:
                    continue
                
                orders.append(Order(
                    id=o.get("id", ""),
                    market_id=o.get("asset_id", ""),
                    side=OrderSide.YES if o.get("side") == "BUY" else OrderSide.NO,
                    order_type=OrderType.LIMIT,
                    price=Decimal(str(o.get("price", 0))),
                    quantity=int(o.get("original_size", 0)),
                    filled_quantity=int(o.get("size_matched", 0)),
                    status=order_status,
                    created_at=datetime.fromisoformat(o.get("created_at", "2030-01-01T00:00:00Z").replace("Z", "+00:00")),
                    platform=self.name
                ))
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
