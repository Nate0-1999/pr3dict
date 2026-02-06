"""
PR3DICT: Kalshi Platform Integration

CFTC-regulated prediction market with REST/WebSocket/FIX APIs.
Dominant in sports betting (75% market share).
"""
import os
import httpx
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


class KalshiPlatform(PlatformInterface):
    """
    Kalshi API integration.
    
    Endpoints:
    - Production: https://trading-api.kalshi.com
    - Sandbox: https://demo-trading-api.kalshi.com
    """
    
    # Updated 2026-02-02: API has moved to api.elections.kalshi.com
    PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
    SANDBOX_URL = "https://api.elections.kalshi.com/trade-api/v2"  # Sandbox may use same URL
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 sandbox: bool = True):
        self.api_key = api_key or os.getenv("KALSHI_API_KEY")
        self.api_secret = api_secret or os.getenv("KALSHI_API_SECRET")
        self.sandbox = sandbox
        self.base_url = self.SANDBOX_URL if sandbox else self.PROD_URL
        
        self._client: Optional[httpx.AsyncClient] = None
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
    
    @property
    def name(self) -> str:
        return "kalshi"
    
    async def connect(self) -> bool:
        """Authenticate and establish HTTP client."""
        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0
            )
            
            # Login to get token
            response = await self._client.post(
                "/login",
                json={
                    "email": self.api_key,
                    "password": self.api_secret
                }
            )
            response.raise_for_status()
            data = response.json()
            
            self._token = data.get("token")
            # Kalshi tokens expire in 30 minutes
            self._token_expiry = datetime.now(timezone.utc)
            
            logger.info(f"Connected to Kalshi ({'sandbox' if self.sandbox else 'production'})")
            return True
            
        except Exception as e:
            logger.error(f"Kalshi connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._token = None
    
    def _headers(self) -> dict:
        """Get auth headers for requests."""
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json"
        }
    
    async def _ensure_token(self) -> None:
        """Refresh token if expired (30 min expiry)."""
        if self._token_expiry:
            elapsed = (datetime.now(timezone.utc) - self._token_expiry).seconds
            if elapsed > 1500:  # Refresh at 25 min
                await self.connect()
    
    # --- Account ---
    
    async def get_balance(self) -> Decimal:
        await self._ensure_token()
        response = await self._client.get("/portfolio/balance", headers=self._headers())
        response.raise_for_status()
        data = response.json()
        # Kalshi returns cents, convert to dollars
        return Decimal(str(data.get("balance", 0))) / 100
    
    async def get_positions(self) -> List[Position]:
        await self._ensure_token()
        response = await self._client.get("/portfolio/positions", headers=self._headers())
        response.raise_for_status()
        data = response.json()
        
        positions = []
        for pos in data.get("market_positions", []):
            positions.append(Position(
                market_id=pos["market_id"],
                ticker=pos.get("ticker", pos["market_id"]),
                side=OrderSide.YES if pos.get("position", 0) > 0 else OrderSide.NO,
                quantity=abs(pos.get("position", 0)),
                avg_price=Decimal(str(pos.get("average_price", 0))) / 100,
                current_price=Decimal(str(pos.get("market_price", 0))) / 100,
                unrealized_pnl=Decimal(str(pos.get("unrealized_pnl", 0))) / 100,
                platform=self.name
            ))
        return positions
    
    # --- Market Data ---
    
    async def get_markets(self,
                          status: str = "open",
                          category: Optional[str] = None,
                          limit: int = 100) -> List[Market]:
        await self._ensure_token()
        
        params = {"status": status, "limit": limit}
        if category:
            params["series_ticker"] = category
        
        response = await self._client.get(
            "/markets",
            params=params,
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        
        markets = []
        for m in data.get("markets", []):
            markets.append(self._parse_market(m))
        return markets
    
    async def get_market(self, market_id: str) -> Optional[Market]:
        await self._ensure_token()
        response = await self._client.get(
            f"/markets/{market_id}",
            headers=self._headers()
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return self._parse_market(response.json().get("market", {}))
    
    def _parse_market(self, m: dict) -> Market:
        """Convert Kalshi market response to Market dataclass."""
        return Market(
            id=m.get("ticker", ""),
            ticker=m.get("ticker", ""),
            title=m.get("title", ""),
            description=m.get("subtitle", ""),
            yes_price=Decimal(str(m.get("yes_bid", 0))) / 100,
            no_price=Decimal(str(m.get("no_bid", 0))) / 100,
            volume=Decimal(str(m.get("volume", 0))),
            liquidity=Decimal(str(m.get("open_interest", 0))),
            close_time=datetime.fromisoformat(m.get("close_time", "2030-01-01T00:00:00Z").replace("Z", "+00:00")),
            resolved=m.get("status") == "settled",
            platform=self.name
        )
    
    async def get_orderbook(self, market_id: str) -> OrderBook:
        await self._ensure_token()
        response = await self._client.get(
            f"/markets/{market_id}/orderbook",
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json().get("orderbook", {})
        
        bids = [(Decimal(str(b[0])) / 100, b[1]) for b in data.get("yes", [])]
        asks = [(Decimal(str(a[0])) / 100, a[1]) for a in data.get("no", [])]
        
        return OrderBook(
            market_id=market_id,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(timezone.utc)
        )
    
    # --- Orders ---
    
    async def place_order(self,
                          market_id: str,
                          side: OrderSide,
                          order_type: OrderType,
                          quantity: int,
                          price: Optional[Decimal] = None) -> Order:
        await self._ensure_token()
        
        payload = {
            "ticker": market_id,
            "action": "buy",
            "side": side.value,
            "count": quantity,
            "type": order_type.value
        }
        
        if price is not None:
            # Kalshi uses cents
            payload["yes_price" if side == OrderSide.YES else "no_price"] = int(price * 100)
        
        response = await self._client.post(
            "/portfolio/orders",
            json=payload,
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json().get("order", {})
        
        return self._parse_order(data)
    
    async def cancel_order(self, order_id: str) -> bool:
        await self._ensure_token()
        response = await self._client.delete(
            f"/portfolio/orders/{order_id}",
            headers=self._headers()
        )
        return response.status_code == 200
    
    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        await self._ensure_token()
        
        params = {}
        if status:
            params["status"] = status.value
        
        response = await self._client.get(
            "/portfolio/orders",
            params=params,
            headers=self._headers()
        )
        response.raise_for_status()
        
        return [self._parse_order(o) for o in response.json().get("orders", [])]
    
    def _parse_order(self, o: dict) -> Order:
        """Convert Kalshi order response to Order dataclass."""
        status_map = {
            "resting": OrderStatus.OPEN,
            "pending": OrderStatus.PENDING,
            "executed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED
        }
        
        return Order(
            id=o.get("order_id", ""),
            market_id=o.get("ticker", ""),
            side=OrderSide.YES if o.get("side") == "yes" else OrderSide.NO,
            order_type=OrderType.LIMIT if o.get("type") == "limit" else OrderType.MARKET,
            price=Decimal(str(o.get("yes_price", o.get("no_price", 0)))) / 100,
            quantity=o.get("count", 0),
            filled_quantity=o.get("filled_count", 0),
            status=status_map.get(o.get("status", ""), OrderStatus.PENDING),
            created_at=datetime.fromisoformat(o.get("created_time", "2030-01-01T00:00:00Z").replace("Z", "+00:00")),
            platform=self.name
        )
