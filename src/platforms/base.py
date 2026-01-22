"""
PR3DICT: Base Platform Interface

Abstract base class defining the contract for all prediction market platforms.
Platforms (Kalshi, Polymarket) must implement these methods.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
from decimal import Decimal
from datetime import datetime


class OrderSide(Enum):
    YES = "yes"
    NO = "no"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class Market:
    """Represents a prediction market contract."""
    id: str
    ticker: str
    title: str
    description: str
    yes_price: Decimal
    no_price: Decimal
    volume: Decimal
    liquidity: Decimal
    close_time: datetime
    resolved: bool
    platform: str
    
    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return abs(Decimal("1.0") - self.yes_price - self.no_price)
    
    @property
    def arbitrage_opportunity(self) -> bool:
        """True if YES + NO < 1.00 (binary complement arb)."""
        return self.yes_price + self.no_price < Decimal("1.0")


@dataclass
class Order:
    """Represents an order on a prediction market."""
    id: str
    market_id: str
    side: OrderSide
    order_type: OrderType
    price: Optional[Decimal]
    quantity: int
    filled_quantity: int
    status: OrderStatus
    created_at: datetime
    platform: str


@dataclass
class Position:
    """Represents a held position in a market."""
    market_id: str
    ticker: str
    side: OrderSide
    quantity: int
    avg_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    platform: str


@dataclass
class OrderBook:
    """Order book for a market."""
    market_id: str
    bids: List[tuple[Decimal, int]]  # (price, size)
    asks: List[tuple[Decimal, int]]
    timestamp: datetime


class PlatformInterface(ABC):
    """
    Abstract base class for prediction market platform integrations.
    
    Implements the same decoupled pattern as ST0CK's BrokerInterface,
    allowing the engine to work with any platform.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Platform name identifier."""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the platform. Returns True on success."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up connections."""
        pass
    
    # --- Account ---
    
    @abstractmethod
    async def get_balance(self) -> Decimal:
        """Get available cash balance."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        pass
    
    # --- Market Data ---
    
    @abstractmethod
    async def get_markets(self, 
                          status: str = "open",
                          category: Optional[str] = None,
                          limit: int = 100) -> List[Market]:
        """Fetch available markets with optional filters."""
        pass
    
    @abstractmethod
    async def get_market(self, market_id: str) -> Optional[Market]:
        """Get a single market by ID."""
        pass
    
    @abstractmethod
    async def get_orderbook(self, market_id: str) -> OrderBook:
        """Get order book depth for a market."""
        pass
    
    # --- Orders ---
    
    @abstractmethod
    async def place_order(self,
                          market_id: str,
                          side: OrderSide,
                          order_type: OrderType,
                          quantity: int,
                          price: Optional[Decimal] = None) -> Order:
        """Place an order. Returns the created order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
        pass
    
    @abstractmethod
    async def get_orders(self, 
                         status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders with optional status filter."""
        pass
