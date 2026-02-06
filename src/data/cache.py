"""
PR3DICT: Market Data Cache

Redis-based caching layer with multi-TTL for different data types:
- Orderbooks: 5 seconds
- Market prices: 30 seconds  
- Market metadata: 5 minutes
- Historical trades: 1 hour
"""
import json
import logging
from typing import Optional, List, Any
from datetime import datetime, timedelta
from decimal import Decimal

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataCache:
    """
    Async Redis cache for market data.
    
    Reduces API load and provides faster access to frequently-used data.
    Falls back to no-cache if Redis is unavailable.
    """
    
    # TTL configs in seconds
    TTL_ORDERBOOK = 5
    TTL_PRICE = 30
    TTL_METADATA = 300
    TTL_TRADES = 3600
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize cache.
        
        Args:
            redis_url: Redis connection string
        """
        self.redis_url = redis_url
        self._client: Optional[redis.Redis] = None
        self._enabled = REDIS_AVAILABLE
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - caching disabled")
    
    async def connect(self) -> bool:
        """Connect to Redis."""
        if not self._enabled:
            return False
        
        try:
            self._client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self._client.ping()
            logger.info("Connected to Redis cache")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e} - caching disabled")
            self._enabled = False
            return False
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
    
    # === Orderbook Caching ===
    
    async def get_orderbook(self, market_id: str, platform: str) -> Optional[dict]:
        """Get cached orderbook."""
        if not self._enabled:
            return None
        
        key = f"orderbook:{platform}:{market_id}"
        data = await self._client.get(key)
        
        if data:
            return json.loads(data)
        return None
    
    async def set_orderbook(self, market_id: str, platform: str, orderbook: dict) -> None:
        """Cache orderbook with 5s TTL."""
        if not self._enabled:
            return
        
        key = f"orderbook:{platform}:{market_id}"
        await self._client.setex(
            key,
            self.TTL_ORDERBOOK,
            json.dumps(orderbook)
        )
    
    # === Price Caching ===
    
    async def get_price(self, market_id: str, platform: str) -> Optional[dict]:
        """Get cached market price."""
        if not self._enabled:
            return None
        
        key = f"price:{platform}:{market_id}"
        data = await self._client.get(key)
        
        if data:
            return json.loads(data)
        return None
    
    async def set_price(self, market_id: str, platform: str, price_data: dict) -> None:
        """Cache price with 30s TTL."""
        if not self._enabled:
            return
        
        key = f"price:{platform}:{market_id}"
        await self._client.setex(
            key,
            self.TTL_PRICE,
            json.dumps(price_data)
        )
    
    # === Market Metadata ===
    
    async def get_market_meta(self, market_id: str, platform: str) -> Optional[dict]:
        """Get cached market metadata (title, description, close time, etc)."""
        if not self._enabled:
            return None
        
        key = f"meta:{platform}:{market_id}"
        data = await self._client.get(key)
        
        if data:
            return json.loads(data)
        return None
    
    async def set_market_meta(self, market_id: str, platform: str, metadata: dict) -> None:
        """Cache metadata with 5min TTL."""
        if not self._enabled:
            return
        
        key = f"meta:{platform}:{market_id}"
        await self._client.setex(
            key,
            self.TTL_METADATA,
            json.dumps(metadata)
        )
    
    # === Market List Caching ===
    
    async def get_market_list(self, platform: str, category: str = "all") -> Optional[List[str]]:
        """Get cached list of active market IDs."""
        if not self._enabled:
            return None
        
        key = f"markets:{platform}:{category}"
        data = await self._client.get(key)
        
        if data:
            return json.loads(data)
        return None
    
    async def set_market_list(self, platform: str, market_ids: List[str], category: str = "all") -> None:
        """Cache market list with 5min TTL."""
        if not self._enabled:
            return
        
        key = f"markets:{platform}:{category}"
        await self._client.setex(
            key,
            self.TTL_METADATA,
            json.dumps(market_ids)
        )
    
    # === Probability Trends (Time Series) ===
    
    async def add_price_point(self, market_id: str, platform: str, yes_price: float, timestamp: Optional[datetime] = None) -> None:
        """
        Add a price point to time series for trend analysis.
        
        Stores as sorted set with timestamp scores.
        Useful for detecting rapid price changes (overreaction).
        """
        if not self._enabled:
            return
        
        key = f"trend:{platform}:{market_id}"
        ts = timestamp or datetime.utcnow()
        score = ts.timestamp()
        
        await self._client.zadd(key, {json.dumps({"price": yes_price, "ts": score}): score})
        
        # Keep only last 24 hours
        cutoff = (datetime.utcnow() - timedelta(hours=24)).timestamp()
        await self._client.zremrangebyscore(key, 0, cutoff)
    
    async def get_price_trend(self, market_id: str, platform: str, hours: int = 1) -> List[dict]:
        """
        Get price trend for the last N hours.
        
        Returns list of {price, ts} dicts sorted by time.
        """
        if not self._enabled:
            return []
        
        key = f"trend:{platform}:{market_id}"
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).timestamp()
        
        data = await self._client.zrangebyscore(key, cutoff, "+inf")
        
        return [json.loads(d) for d in data]
    
    # === Statistics ===
    
    async def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self._enabled or not self._client:
            return {"enabled": False}
        
        info = await self._client.info("stats")
        
        return {
            "enabled": True,
            "total_keys": await self._client.dbsize(),
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1)
        }
