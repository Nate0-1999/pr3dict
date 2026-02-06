"""
PR3DICT: Optimization Solver Demo

Demonstrates how to use the integer programming solver for optimal arbitrage execution.

Usage:
    python examples/optimization_demo.py
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from decimal import Decimal
import logging

from src.optimization.solver import (
    ArbitrageSolver,
    ArbitrageOpportunity,
    SolverBackend
)
from src.optimization.benchmarks import (
    SolverBenchmark,
    create_synthetic_opportunities
)
from src.optimization.integration import (
    OptimizedArbitrageExecutor,
    OpportunityConverter
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_optimization():
    """Demo 1: Basic optimization with single opportunity."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Single-Opportunity Optimization")
    print("=" * 80)
    
    # Create solver
    solver = ArbitrageSolver(
        transaction_fee_rate=Decimal("0.02"),  # 2% fee
        min_profit_threshold=Decimal("5.0")     # $5 minimum
    )
    
    # Create a simple arbitrage opportunity
    # Market trading at $0.55, expected value $1.00 → $0.45 profit per contract
    opportunity = ArbitrageOpportunity(
        market_id="TRUMP_2024",
        outcome_id="YES",
        current_price=Decimal("0.55"),
        expected_value=Decimal("1.00"),
        max_liquidity=1000,
        platform="polymarket",
        ticker="TRUMP-YES"
    )
    
    # Solve with $1000 capital
    capital = Decimal("1000.0")
    result = solver.solve(
        opportunities=[opportunity],
        available_capital=capital,
        backend=SolverBackend.FRANK_WOLFE
    )
    
    # Display results
    print(f"\n📊 Optimization Results:")
    print(f"  Solver: {result.solver_backend}")
    print(f"  Status: {result.solver_status}")
    print(f"  Solve Time: {result.solve_time_ms:.2f}ms")
    print(f"\n💰 Allocation:")
    print(f"  Capital Used: ${result.total_capital_used:.2f}")
    print(f"  Expected Profit: ${result.total_expected_profit:.2f}")
    print(f"  ROI: {result.profit_percentage:.2f}%")
    print(f"  Number of Trades: {result.num_trades}")
    
    if result.allocations:
        for alloc in result.allocations:
            print(f"\n  Trade Details:")
            print(f"    Market: {alloc.market_id}")
            print(f"    Quantity: {alloc.quantity} contracts")
            print(f"    Price: ${alloc.price:.4f}")
            print(f"    Capital Required: ${alloc.capital_required:.2f}")
            print(f"    Expected Profit: ${alloc.expected_profit:.2f}")


def demo_multiple_opportunities():
    """Demo 2: Optimize across multiple arbitrage opportunities."""
    print("\n" + "=" * 80)
    print("DEMO 2: Multi-Opportunity Portfolio Optimization")
    print("=" * 80)
    
    solver = ArbitrageSolver(
        transaction_fee_rate=Decimal("0.02"),
        max_position_fraction=Decimal("0.25")  # Max 25% per market
    )
    
    # Create multiple opportunities with varying profit margins
    opportunities = [
        ArbitrageOpportunity(
            market_id=f"MARKET_{i}",
            outcome_id="YES",
            current_price=Decimal(str(0.40 + i * 0.05)),
            expected_value=Decimal("1.00"),
            max_liquidity=500 + i * 100,
            platform="polymarket",
            ticker=f"MKT{i}"
        )
        for i in range(8)
    ]
    
    capital = Decimal("10000.0")
    
    print(f"\n📈 Optimizing {len(opportunities)} opportunities with ${capital:.2f} capital...")
    
    result = solver.solve(
        opportunities=opportunities,
        available_capital=capital,
        backend=SolverBackend.FRANK_WOLFE
    )
    
    print(f"\n📊 Portfolio Optimization Results:")
    print(f"  Total Capital Allocated: ${result.total_capital_used:.2f} / ${capital:.2f}")
    print(f"  Capital Efficiency: {result.capital_efficiency:.2%}")
    print(f"  Expected Total Profit: ${result.total_expected_profit:.2f}")
    print(f"  Average ROI: {result.profit_percentage:.2f}%")
    print(f"  Number of Selected Trades: {result.num_trades}")
    print(f"  Solve Time: {result.solve_time_ms:.2f}ms")
    
    print(f"\n💼 Trade Allocations:")
    for i, alloc in enumerate(result.allocations, 1):
        roi = (alloc.expected_profit / alloc.capital_required * 100) if alloc.capital_required > 0 else 0
        print(f"  {i}. {alloc.market_id}: {alloc.quantity} contracts @ ${alloc.price:.4f}")
        print(f"     Capital: ${alloc.capital_required:.2f} | Profit: ${alloc.expected_profit:.2f} | ROI: {roi:.2f}%")


def demo_binary_complement_arbitrage():
    """Demo 3: Binary complement arbitrage (YES + NO < $1)."""
    print("\n" + "=" * 80)
    print("DEMO 3: Binary Complement Arbitrage")
    print("=" * 80)
    
    print("\nScenario: Market with YES=$0.45, NO=$0.50 (sum=$0.95 < $1.00)")
    print("Strategy: Buy both YES and NO, guaranteed $0.05 profit at resolution")
    
    solver = ArbitrageSolver(transaction_fee_rate=Decimal("0.02"))
    
    # Create both sides of binary market
    opportunities = [
        ArbitrageOpportunity(
            market_id="BINARY_EVENT",
            outcome_id="YES",
            current_price=Decimal("0.45"),
            expected_value=Decimal("1.00"),
            max_liquidity=2000,
            platform="polymarket",
            ticker="EVENT-YES",
            complement_id="BINARY_EVENT_NO",
            complement_price=Decimal("0.50")
        ),
        ArbitrageOpportunity(
            market_id="BINARY_EVENT",
            outcome_id="NO",
            current_price=Decimal("0.50"),
            expected_value=Decimal("1.00"),
            max_liquidity=2000,
            platform="polymarket",
            ticker="EVENT-NO",
            complement_id="BINARY_EVENT_YES",
            complement_price=Decimal("0.45")
        )
    ]
    
    capital = Decimal("5000.0")
    result = solver.solve(
        opportunities=opportunities,
        available_capital=capital,
        backend=SolverBackend.FRANK_WOLFE
    )
    
    print(f"\n📊 Binary Complement Results:")
    print(f"  Both Sides Purchased: {result.num_trades == 2}")
    print(f"  Total Capital: ${result.total_capital_used:.2f}")
    print(f"  Expected Profit: ${result.total_expected_profit:.2f}")
    print(f"  Guaranteed Return: {result.profit_percentage:.2f}%")
    
    yes_alloc = next((a for a in result.allocations if "YES" in a.outcome_id), None)
    no_alloc = next((a for a in result.allocations if "NO" in a.outcome_id), None)
    
    if yes_alloc and no_alloc:
        print(f"\n  YES Position: {yes_alloc.quantity} contracts @ ${yes_alloc.price:.2f}")
        print(f"  NO Position: {no_alloc.quantity} contracts @ ${no_alloc.price:.2f}")
        print(f"  Net Cost: ${yes_alloc.capital_required + no_alloc.capital_required:.2f}")
        print(f"  Resolution Value: ${(yes_alloc.quantity + no_alloc.quantity) * 1.00:.2f}")
        print(f"  Guaranteed Profit: ${result.total_expected_profit:.2f}")


def demo_backend_comparison():
    """Demo 4: Compare different solver backends."""
    print("\n" + "=" * 80)
    print("DEMO 4: Solver Backend Performance Comparison")
    print("=" * 80)
    
    solver = ArbitrageSolver()
    benchmark = SolverBenchmark(solver)
    
    # Create test opportunities
    opportunities = create_synthetic_opportunities(30)
    capital = Decimal("10000.0")
    
    print(f"\nBenchmarking with {len(opportunities)} opportunities...")
    
    # Test available backends
    backends = [SolverBackend.FRANK_WOLFE]
    
    try:
        import cvxpy
        backends.append(SolverBackend.CVXPY_ECOS)
        print("  ✓ CVXPY available")
    except ImportError:
        print("  ✗ CVXPY not installed (pip install cvxpy)")
    
    try:
        import gurobipy
        backends.append(SolverBackend.GUROBI)
        print("  ✓ Gurobi available")
    except ImportError:
        print("  ✗ Gurobi not available (commercial license required)")
    
    # Run benchmarks
    suite = benchmark.benchmark_backends(
        opportunities=opportunities,
        capital=capital,
        backends=backends,
        runs_per_backend=5
    )
    
    print(suite.compare_backends())


def demo_real_time_performance():
    """Demo 5: Test real-time performance (<50ms requirement)."""
    print("\n" + "=" * 80)
    print("DEMO 5: Real-Time Performance Test (Target: <50ms)")
    print("=" * 80)
    
    solver = ArbitrageSolver()
    benchmark = SolverBenchmark(solver)
    
    # Test with different problem sizes
    for n in [10, 25, 50]:
        opportunities = create_synthetic_opportunities(n)
        capital = Decimal("10000.0")
        
        print(f"\n📊 Testing with {n} opportunities:")
        
        perf = benchmark.benchmark_real_time_performance(
            opportunities=opportunities,
            capital=capital,
            backend=SolverBackend.FRANK_WOLFE,
            target_time_ms=50.0,
            num_runs=20
        )
        
        print(f"  Mean:   {perf['mean_time_ms']:6.2f}ms")
        print(f"  Median: {perf['median_time_ms']:6.2f}ms")
        print(f"  P95:    {perf['p95_time_ms']:6.2f}ms")
        print(f"  P99:    {perf['p99_time_ms']:6.2f}ms")
        print(f"  Max:    {perf['max_time_ms']:6.2f}ms")
        
        status = "✅ PASS" if perf['meets_target_pct'] >= 95 else "⚠️  WARN" if perf['meets_target_pct'] >= 80 else "❌ FAIL"
        print(f"  Success Rate: {perf['meets_target_pct']:5.1f}% under 50ms {status}")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("PR3DICT INTEGER PROGRAMMING OPTIMIZATION DEMO")
    print("=" * 80)
    print("\nDemonstrating optimal arbitrage execution using combinatorial optimization")
    print("(as used by $40M traders, not brute-force enumeration)")
    
    try:
        demo_basic_optimization()
        demo_multiple_opportunities()
        demo_binary_complement_arbitrage()
        demo_backend_comparison()
        demo_real_time_performance()
        
        print("\n" + "=" * 80)
        print("✅ All demos completed successfully!")
        print("=" * 80)
        print("\nNext Steps:")
        print("  1. Install CVXPY: pip install cvxpy")
        print("  2. Run tests: pytest tests/test_optimization.py")
        print("  3. Run benchmarks: python src/optimization/benchmarks.py")
        print("  4. Integrate with live trading: See src/optimization/integration.py")
        print("\nOptional (for best performance):")
        print("  - Get Gurobi license (free for academic, ~$40k/year commercial)")
        print("  - Install: pip install gurobipy")
        print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
