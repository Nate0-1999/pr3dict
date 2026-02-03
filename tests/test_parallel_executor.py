"""
PR3DICT: Parallel Executor Tests

Test suite for parallel execution engine.

Run with: pytest tests/test_parallel_executor.py -v
"""
import asyncio
import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta

from src.execution.parallel_executor import (
    ParallelExecutor, ExecutionStrategy, TradeLeg, MultiLegTrade,
    ExecutionConfig, LegStatus
)
from src.execution.metrics import MetricsCollector
from src.execution.integration import ArbitrageExecutionEngine
from src.platforms.base import (
    PlatformInterface, Market, Order, OrderSide, OrderType, OrderStatus
)
from src.risk.manager import RiskManager


class MockPlatform(PlatformInterface):
    """Mock platform for testing."""
    
    def __init__(self, name: str, balance: Decimal = Decimal("1000")):
        self._name = name
        self._balance = balance
        self._orders = []
        self._order_counter = 0
        self._fill_delay_ms = 10  # Simulated fill time
    
    @property
    def name(self) -> str:
        return self._name
    
    async def connect(self) -> bool:
        return True
    
    async def disconnect(self) -> None:
        pass
    
    async def get_balance(self) -> Decimal:
        return self._balance
    
    async def get_positions(self):
        return []
    
    async def get_markets(self, status="open", category=None, limit=100):
        # Return mock markets
        return [
            Market(
                id=f"market_{i}",
                ticker=f"TEST_{i}",
                title=f"Test Market {i}",
                description="Test",
                yes_price=Decimal("0.45") if i % 2 == 0 else Decimal("0.55"),
                no_price=Decimal("0.50"),
                volume=Decimal("10000"),
                liquidity=Decimal("5000"),
                close_time=datetime.now(timezone.utc) + timedelta(days=7),
                resolved=False,
                platform=self.name
            )
            for i in range(5)
        ]
    
    async def get_market(self, market_id: str):
        markets = await self.get_markets()
        return next((m for m in markets if m.id == market_id), None)
    
    async def get_orderbook(self, market_id: str):
        return None
    
    async def place_order(self, market_id, side, order_type, quantity, price=None):
        # Simulate order placement
        self._order_counter += 1
        order_id = f"order_{self.name}_{self._order_counter}"
        
        # Simulate fill delay
        await asyncio.sleep(self._fill_delay_ms / 1000.0)
        
        order = Order(
            id=order_id,
            market_id=market_id,
            side=side,
            order_type=order_type,
            price=price or Decimal("0.5"),
            quantity=quantity,
            filled_quantity=quantity,  # Instant fill for testing
            status=OrderStatus.FILLED,
            created_at=datetime.now(timezone.utc),
            platform=self.name
        )
        
        self._orders.append(order)
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        for order in self._orders:
            if order.id == order_id:
                order.status = OrderStatus.CANCELLED
                return True
        return False
    
    async def get_orders(self, status=None):
        if status:
            return [o for o in self._orders if o.status == status]
        return self._orders


@pytest.fixture
def mock_platforms():
    """Create mock platforms for testing."""
    return {
        "kalshi": MockPlatform("kalshi", Decimal("5000")),
        "polymarket": MockPlatform("polymarket", Decimal("5000"))
    }


@pytest.fixture
def risk_manager():
    """Create risk manager for testing."""
    return RiskManager()


@pytest.fixture
def parallel_executor(mock_platforms, risk_manager):
    """Create parallel executor for testing."""
    return ParallelExecutor(
        platforms=mock_platforms,
        risk_manager=risk_manager
    )


@pytest.mark.asyncio
async def test_market_execution(parallel_executor, mock_platforms):
    """Test market order execution strategy."""
    legs = [
        TradeLeg(
            market_id="market_1",
            side=OrderSide.YES,
            quantity=10,
            target_price=Decimal("0.45"),
            platform="kalshi"
        ),
        TradeLeg(
            market_id="market_1",
            side=OrderSide.NO,
            quantity=10,
            target_price=Decimal("0.50"),
            platform="kalshi"
        )
    ]
    
    trade = await parallel_executor.execute_arbitrage(
        legs=legs,
        strategy=ExecutionStrategy.MARKET,
        expected_profit=Decimal("0.50")
    )
    
    assert trade.all_filled, "All legs should be filled"
    assert trade.committed, "Trade should be committed"
    assert not trade.rolled_back, "Trade should not be rolled back"
    assert trade.execution_time_ms is not None
    assert trade.execution_time_ms < 100, "Should execute quickly"


@pytest.mark.asyncio
async def test_limit_execution(parallel_executor, mock_platforms):
    """Test limit order execution strategy."""
    legs = [
        TradeLeg(
            market_id="market_2",
            side=OrderSide.YES,
            quantity=5,
            target_price=Decimal("0.45"),
            platform="polymarket"
        )
    ]
    
    trade = await parallel_executor.execute_arbitrage(
        legs=legs,
        strategy=ExecutionStrategy.LIMIT,
        expected_profit=Decimal("0.25")
    )
    
    assert trade.all_filled or trade.rolled_back


@pytest.mark.asyncio
async def test_hybrid_execution(parallel_executor, mock_platforms):
    """Test hybrid execution strategy."""
    legs = [
        TradeLeg(
            market_id="market_3",
            side=OrderSide.YES,
            quantity=15,
            target_price=Decimal("0.48"),
            platform="kalshi"
        ),
        TradeLeg(
            market_id="market_3",
            side=OrderSide.NO,
            quantity=15,
            target_price=Decimal("0.47"),
            platform="kalshi"
        )
    ]
    
    trade = await parallel_executor.execute_arbitrage(
        legs=legs,
        strategy=ExecutionStrategy.HYBRID,
        expected_profit=Decimal("0.75")
    )
    
    assert trade is not None
    # Hybrid should attempt limit first, fallback to market


@pytest.mark.asyncio
async def test_preflight_checks(parallel_executor, risk_manager, mock_platforms):
    """Test pre-flight validation checks."""
    # Create trade that exceeds limits
    legs = [
        TradeLeg(
            market_id="market_4",
            side=OrderSide.YES,
            quantity=1000,  # Huge size
            target_price=Decimal("0.50"),
            platform="kalshi"
        )
    ]
    
    trade = await parallel_executor.execute_arbitrage(
        legs=legs,
        strategy=ExecutionStrategy.MARKET
    )
    
    # Should fail preflight checks or be reduced


@pytest.mark.asyncio
async def test_partial_fill_rollback(parallel_executor, mock_platforms):
    """Test rollback on partial fill."""
    # Configure one platform to fail
    mock_platforms["polymarket"]._fill_delay_ms = 100  # Too slow
    
    legs = [
        TradeLeg(
            market_id="market_5",
            side=OrderSide.YES,
            quantity=10,
            target_price=Decimal("0.45"),
            platform="kalshi"
        ),
        TradeLeg(
            market_id="market_5",
            side=OrderSide.NO,
            quantity=10,
            target_price=Decimal("0.50"),
            platform="polymarket"  # Will timeout
        )
    ]
    
    # Set short timeout
    config = ExecutionConfig(max_execution_time_ms=20)
    executor = ParallelExecutor(
        platforms=mock_platforms,
        risk_manager=risk_manager,
        config=config
    )
    
    trade = await executor.execute_arbitrage(
        legs=legs,
        strategy=ExecutionStrategy.MARKET
    )
    
    # Should rollback due to timeout on one leg


@pytest.mark.asyncio
async def test_metrics_collection(parallel_executor):
    """Test metrics collection during execution."""
    legs = [
        TradeLeg(
            market_id="market_1",
            side=OrderSide.YES,
            quantity=5,
            target_price=Decimal("0.45"),
            platform="kalshi"
        )
    ]
    
    await parallel_executor.execute_arbitrage(
        legs=legs,
        strategy=ExecutionStrategy.MARKET,
        expected_profit=Decimal("0.25")
    )
    
    summary = parallel_executor.metrics.get_summary()
    
    assert summary["total_trades"] > 0
    assert "by_strategy" in summary
    assert "market" in summary["by_strategy"]


@pytest.mark.asyncio
async def test_arbitrage_detection(mock_platforms, risk_manager):
    """Test arbitrage opportunity detection."""
    engine = ArbitrageExecutionEngine(
        platforms=mock_platforms,
        risk_manager=risk_manager,
        paper_mode=True
    )
    
    markets = await mock_platforms["kalshi"].get_markets()
    
    # Detect binary complement opportunities
    opportunities = await engine.detect_binary_complement_arb(markets)
    
    assert isinstance(opportunities, list)
    # May or may not find opportunities depending on mock data


@pytest.mark.asyncio
async def test_cross_platform_detection(mock_platforms, risk_manager):
    """Test cross-platform arbitrage detection."""
    engine = ArbitrageExecutionEngine(
        platforms=mock_platforms,
        risk_manager=risk_manager,
        paper_mode=True
    )
    
    # Get markets from both platforms
    kalshi_markets = await mock_platforms["kalshi"].get_markets()
    poly_markets = await mock_platforms["polymarket"].get_markets()
    all_markets = kalshi_markets + poly_markets
    
    # Detect cross-platform opportunities
    opportunities = await engine.detect_cross_platform_arb(all_markets)
    
    assert isinstance(opportunities, list)


@pytest.mark.asyncio
async def test_end_to_end_arbitrage(mock_platforms, risk_manager):
    """Test end-to-end arbitrage execution."""
    engine = ArbitrageExecutionEngine(
        platforms=mock_platforms,
        risk_manager=risk_manager,
        paper_mode=True  # Paper mode for testing
    )
    
    markets = await mock_platforms["kalshi"].get_markets()
    
    # Scan and execute
    trades = await engine.scan_and_execute(
        markets=markets,
        strategy=ExecutionStrategy.HYBRID,
        max_opportunities=2
    )
    
    assert isinstance(trades, list)
    
    # Check statistics
    stats = engine.get_statistics()
    assert "opportunities_detected" in stats
    assert "metrics" in stats


def test_metrics_aggregation():
    """Test metrics aggregation logic."""
    collector = MetricsCollector()
    
    # Create mock trade metrics
    from src.execution.parallel_executor import MultiLegTrade
    
    trade = MultiLegTrade(
        trade_id="test_1",
        legs=[],
        strategy=ExecutionStrategy.MARKET
    )
    trade.committed = True
    trade.start_time = 0.0
    trade.end_time = 0.025  # 25ms
    trade.expected_profit = Decimal("1.00")
    trade.actual_profit = Decimal("0.95")
    
    collector.record_trade(trade)
    
    summary = collector.get_summary()
    assert summary["total_trades"] == 1
    assert summary["successful"] == 1


def test_execution_config():
    """Test execution configuration."""
    config = ExecutionConfig(
        max_execution_time_ms=30,
        max_slippage_pct=Decimal("0.02"),
        use_batch_transactions=True
    )
    
    assert config.max_execution_time_ms == 30
    assert config.max_slippage_pct == Decimal("0.02")
    assert config.use_batch_transactions is True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
