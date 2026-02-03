"""
PR3DICT: Parallel Execution Example

Quick start example showing how to use the parallel execution engine
for atomic multi-leg arbitrage trades.

Usage:
    python examples/parallel_execution_example.py
"""
import asyncio
import logging
from decimal import Decimal
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_basic_execution():
    """Example 1: Basic parallel execution"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Parallel Execution")
    print("="*60)
    
    from src.execution.parallel_executor import (
        ParallelExecutor, ExecutionStrategy, TradeLeg
    )
    from src.execution.integration import ArbitrageExecutionEngine
    from src.platforms.base import OrderSide
    from src.risk.manager import RiskManager
    
    # Setup (you would use real platforms)
    # For demo, we'll use mock platforms
    from tests.test_parallel_executor import MockPlatform
    
    platforms = {
        "polymarket": MockPlatform("polymarket", Decimal("10000"))
    }
    
    risk_manager = RiskManager()
    
    # Create executor
    executor = ParallelExecutor(
        platforms=platforms,
        risk_manager=risk_manager
    )
    
    # Define arbitrage legs
    # Example: Binary complement arb (YES + NO < $1.00)
    legs = [
        TradeLeg(
            market_id="TRUMP_2024",
            side=OrderSide.YES,
            quantity=100,
            target_price=Decimal("0.42"),
            platform="polymarket"
        ),
        TradeLeg(
            market_id="TRUMP_2024",
            side=OrderSide.NO,
            quantity=100,
            target_price=Decimal("0.53"),
            platform="polymarket"
        )
    ]
    
    # Expected profit: 100 * ($1.00 - $0.42 - $0.53) = $5.00
    expected_profit = Decimal("5.00")
    
    print(f"\n📊 Trade Setup:")
    print(f"  Legs: {len(legs)}")
    print(f"  Strategy: HYBRID (limit → market fallback)")
    print(f"  Expected Profit: ${expected_profit}")
    
    # Execute
    print(f"\n⚡ Executing trade...")
    trade = await executor.execute_arbitrage(
        legs=legs,
        strategy=ExecutionStrategy.HYBRID,
        expected_profit=expected_profit
    )
    
    # Results
    print(f"\n📈 Results:")
    print(f"  Trade ID: {trade.trade_id}")
    print(f"  Status: {'✓ COMMITTED' if trade.committed else '✗ FAILED'}")
    print(f"  Execution Time: {trade.execution_time_ms:.1f}ms")
    print(f"  Within Block: {'✓' if trade.execution_time_ms <= 30 else '✗'}")
    print(f"  Actual Profit: ${trade.actual_profit or 'N/A'}")
    
    if trade.slippage_pct:
        print(f"  Slippage: {trade.slippage_pct:.2%}")
    
    # Per-leg details
    print(f"\n📋 Leg Details:")
    for i, leg in enumerate(trade.legs, 1):
        print(f"  Leg {i}: {leg.status.value} - {leg.side.value.upper()} x{leg.quantity}")
        if leg.execution_time_ms:
            print(f"         Time: {leg.execution_time_ms:.1f}ms")


async def example_arbitrage_detection():
    """Example 2: Automated arbitrage detection and execution"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Automated Arbitrage Detection")
    print("="*60)
    
    from src.execution.integration import ArbitrageExecutionEngine
    from src.execution.parallel_executor import ExecutionStrategy
    from src.risk.manager import RiskManager
    from tests.test_parallel_executor import MockPlatform
    
    # Setup
    platforms = {
        "kalshi": MockPlatform("kalshi", Decimal("5000")),
        "polymarket": MockPlatform("polymarket", Decimal("5000"))
    }
    
    risk_manager = RiskManager()
    
    # Create arbitrage engine
    engine = ArbitrageExecutionEngine(
        platforms=platforms,
        risk_manager=risk_manager,
        paper_mode=True  # Start with paper trading
    )
    
    print(f"\n🔍 Scanning markets for arbitrage opportunities...")
    
    # Get markets
    markets = []
    for platform in platforms.values():
        platform_markets = await platform.get_markets()
        markets.extend(platform_markets)
    
    print(f"  Found {len(markets)} markets to scan")
    
    # Detect opportunities
    bc_opps = await engine.detect_binary_complement_arb(
        markets=markets,
        min_profit_pct=Decimal("0.02")  # 2% minimum
    )
    
    xp_opps = await engine.detect_cross_platform_arb(
        markets=markets,
        min_differential=Decimal("0.03")  # 3% minimum
    )
    
    total_opps = len(bc_opps) + len(xp_opps)
    print(f"\n💡 Opportunities Found:")
    print(f"  Binary Complement: {len(bc_opps)}")
    print(f"  Cross-Platform: {len(xp_opps)}")
    print(f"  Total: {total_opps}")
    
    if total_opps > 0:
        print(f"\n⚡ Executing top opportunities...")
        
        # Execute top opportunities
        trades = await engine.scan_and_execute(
            markets=markets,
            strategy=ExecutionStrategy.HYBRID,
            max_opportunities=3
        )
        
        print(f"\n📈 Execution Summary:")
        print(f"  Trades Executed: {len(trades)}")
        
        successful = sum(1 for t in trades if t.committed)
        print(f"  Successful: {successful}/{len(trades)}")
        
        total_profit = sum(t.actual_profit for t in trades if t.actual_profit)
        print(f"  Total Profit: ${total_profit}")
    else:
        print(f"\n  No opportunities found (this is normal in efficient markets)")


async def example_metrics_monitoring():
    """Example 3: Metrics and monitoring"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Metrics & Monitoring")
    print("="*60)
    
    from src.execution.metrics import MetricsCollector
    from src.execution.parallel_executor import MultiLegTrade, ExecutionStrategy
    
    # Create collector
    collector = MetricsCollector()
    
    # Simulate some trades
    print(f"\n📊 Simulating trade execution...")
    
    for i in range(5):
        trade = MultiLegTrade(
            trade_id=f"trade_{i}",
            legs=[],
            strategy=ExecutionStrategy.HYBRID
        )
        trade.committed = i % 4 != 0  # 75% success rate
        trade.start_time = 0.0
        trade.end_time = 0.020 + (i * 0.005)  # 20-40ms
        trade.expected_profit = Decimal("5.00")
        trade.actual_profit = Decimal("4.50") if trade.committed else None
        
        collector.record_trade(trade)
    
    # Get summary
    summary = collector.get_summary()
    
    print(f"\n📈 Performance Summary:")
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Success Rate: {summary['success_rate_pct']}%")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    
    # Strategy performance
    if summary["by_strategy"]:
        print(f"\n📊 Strategy Performance:")
        for strategy, stats in summary["by_strategy"].items():
            print(f"\n  {strategy.upper()}:")
            print(f"    Trades: {stats['count']}")
            print(f"    Success Rate: {stats['success_rate_pct']}%")
            print(f"    Avg Execution Time: {stats['avg_execution_time_ms']:.1f}ms")
            print(f"    Within Block Rate: {stats['within_block_rate_pct']}%")
    
    # Recent trades
    print(f"\n📋 Recent Trades:")
    recent = collector.get_recent_trades(limit=5)
    for trade in recent:
        status = "✓" if trade["success"] else "✗"
        print(f"  {status} {trade['trade_id']}: {trade['time_ms']}ms - ${trade['profit'] or 'N/A'}")


async def example_strategy_comparison():
    """Example 4: Compare execution strategies"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Strategy Comparison")
    print("="*60)
    
    from src.execution.parallel_executor import (
        ParallelExecutor, ExecutionStrategy, TradeLeg
    )
    from src.platforms.base import OrderSide
    from src.risk.manager import RiskManager
    from tests.test_parallel_executor import MockPlatform
    
    # Setup
    platforms = {
        "polymarket": MockPlatform("polymarket", Decimal("10000"))
    }
    risk_manager = RiskManager()
    
    # Test legs
    legs = [
        TradeLeg(
            market_id="test_market",
            side=OrderSide.YES,
            quantity=50,
            target_price=Decimal("0.45"),
            platform="polymarket"
        )
    ]
    
    strategies = [
        ExecutionStrategy.MARKET,
        ExecutionStrategy.LIMIT,
        ExecutionStrategy.HYBRID
    ]
    
    print(f"\n⚡ Testing each strategy...")
    
    results = {}
    for strategy in strategies:
        executor = ParallelExecutor(platforms, risk_manager)
        
        trade = await executor.execute_arbitrage(
            legs=legs,
            strategy=strategy,
            expected_profit=Decimal("2.50")
        )
        
        results[strategy.value] = {
            "success": trade.committed,
            "time_ms": trade.execution_time_ms,
            "within_block": trade.execution_time_ms <= 30 if trade.execution_time_ms else False
        }
    
    # Compare
    print(f"\n📊 Strategy Comparison:")
    print(f"\n{'Strategy':<15} {'Success':<10} {'Time (ms)':<12} {'Within Block'}")
    print("-" * 55)
    
    for strategy, data in results.items():
        success = "✓" if data["success"] else "✗"
        within = "✓" if data["within_block"] else "✗"
        print(f"{strategy:<15} {success:<10} {data['time_ms']:<12.1f} {within}")
    
    print(f"\n💡 Recommendations:")
    print(f"  • MARKET: Best for urgent arbs, accept higher slippage")
    print(f"  • LIMIT: Best for patient execution, minimal slippage")
    print(f"  • HYBRID: ⭐ Recommended - balances speed and slippage")


async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("PR3DICT: Parallel Execution Engine Examples")
    print("="*60)
    
    try:
        # Run examples
        await example_basic_execution()
        await asyncio.sleep(1)
        
        await example_arbitrage_detection()
        await asyncio.sleep(1)
        
        await example_metrics_monitoring()
        await asyncio.sleep(1)
        
        await example_strategy_comparison()
        
        print("\n" + "="*60)
        print("✅ All examples completed!")
        print("="*60)
        
        print("\n📚 Next Steps:")
        print("  1. Review docs/PARALLEL_EXECUTION.md for detailed docs")
        print("  2. Run tests: pytest tests/test_parallel_executor.py")
        print("  3. Start with paper_mode=True")
        print("  4. Monitor metrics daily")
        print("  5. Gradually increase position sizes")
        
        print("\n🚀 Happy Trading!\n")
        
    except Exception as e:
        logger.error(f"Example error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
