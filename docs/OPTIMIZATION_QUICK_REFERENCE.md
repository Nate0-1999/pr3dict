# Optimization Solver - Quick Reference

## One-Minute Setup

```bash
# Install minimum dependencies
pip install numpy

# Optional: Add CVXPY for more solver options
pip install cvxpy

# Run validation
python3 validate_optimization.py

# See it in action
python3 examples/optimization_demo.py
```

## Basic Usage Pattern

```python
from decimal import Decimal
from src.optimization.solver import ArbitrageSolver, ArbitrageOpportunity, SolverBackend

# 1. Create solver
solver = ArbitrageSolver(transaction_fee_rate=Decimal("0.02"))

# 2. Define opportunities
opps = [
    ArbitrageOpportunity(
        market_id="MARKET_1",
        outcome_id="YES",
        current_price=Decimal("0.55"),
        expected_value=Decimal("1.00"),
        max_liquidity=1000,
        platform="polymarket",
        ticker="MKT1"
    )
]

# 3. Solve
result = solver.solve(
    opportunities=opps,
    available_capital=Decimal("10000.0"),
    backend=SolverBackend.FRANK_WOLFE
)

# 4. Execute trades
for alloc in result.allocations:
    print(f"{alloc.market_id}: Buy {alloc.quantity} @ ${alloc.price}")
```

## Solver Backend Cheat Sheet

| Backend | Speed | Quality | Cost | Use When |
|---------|-------|---------|------|----------|
| `FRANK_WOLFE` | ⚡⚡⚡ Fast (10-50ms) | ~98% optimal | Free | Default choice |
| `CVXPY_ECOS` | ⚡⚡ Medium (50-200ms) | 100% optimal | Free | Need exact optimum |
| `GUROBI` | ⚡⚡⚡ Fastest (10-50ms) | 100% optimal | $$$$ | Production, high-volume |

## Common Parameters

```python
ArbitrageSolver(
    transaction_fee_rate=Decimal("0.02"),  # 2% fee
    gas_fee=Decimal("0.0"),                # Fixed fee per trade
    max_position_fraction=Decimal("0.20"), # Max 20% per market
    min_profit_threshold=Decimal("1.0")    # $1 minimum profit
)
```

## Result Object

```python
result = solver.solve(...)

# Access results
result.allocations              # List[TradeAllocation]
result.total_capital_used       # Decimal
result.total_expected_profit    # Decimal
result.solve_time_ms           # float
result.profit_percentage        # float (ROI)
result.num_trades              # int
```

## Quick Benchmarking

```python
from src.optimization.benchmarks import SolverBenchmark

benchmark = SolverBenchmark(solver)

# Test performance
perf = benchmark.benchmark_real_time_performance(
    opportunities=opps,
    capital=Decimal("10000"),
    backend=SolverBackend.FRANK_WOLFE,
    num_runs=20
)

print(f"P95 time: {perf['p95_time_ms']:.2f}ms")
print(f"Success rate: {perf['meets_target_pct']:.1f}% under 50ms")
```

## Integration with Strategy

```python
from src.optimization.integration import OptimizedArbitrageExecutor

executor = OptimizedArbitrageExecutor(
    solver=solver,
    backend=SolverBackend.FRANK_WOLFE
)

# Get signals from strategy
signals = await strategy.scan_markets(markets)

# Optimize + execute
result = await executor.optimize_and_execute(
    signals=signals,
    markets={m.id: m for m in markets},
    available_capital=capital
)
```

## Binary Complement Arbitrage

When YES + NO < $1.00:

```python
opps = [
    ArbitrageOpportunity(
        market_id="EVENT",
        outcome_id="YES",
        current_price=Decimal("0.45"),
        expected_value=Decimal("1.00"),
        max_liquidity=1000,
        complement_id="EVENT_NO",
        complement_price=Decimal("0.50")  # 0.45 + 0.50 = 0.95 < 1.00!
    ),
    # ... add NO side
]

result = solver.solve(opps, capital)  # Buys both sides
```

## Position Rebalancing

```python
new_positions = solver.bregman_project(
    current_positions={"A": 0.3, "B": 0.5, "C": 0.2},
    target_distribution={"A": 0.4, "B": 0.3, "C": 0.3}
)
# Returns optimal rebalancing minimizing transaction costs
```

## Troubleshooting

### "No module named numpy"
```bash
pip install numpy
```

### "Solve time too slow"
```python
# Use Frank-Wolfe instead of CVXPY
backend=SolverBackend.FRANK_WOLFE
```

### "No profitable trades"
```python
# Lower profit threshold
solver = ArbitrageSolver(min_profit_threshold=Decimal("0.10"))
```

### "Solver status: infeasible"
```python
# Increase capital or reduce position limits
available_capital=Decimal("50000.0")
```

## Performance Tips

1. **Filter opportunities first** - Only pass profitable ones to solver
2. **Use Frank-Wolfe for speed** - 98% optimal, 3x faster than CVXPY
3. **Warm-start** - Reuse previous solutions when opportunities change slowly
4. **Batch updates** - Don't re-solve for every tiny market update

## File Locations

- **Core**: `src/optimization/solver.py`
- **Integration**: `src/optimization/integration.py`
- **Benchmarks**: `src/optimization/benchmarks.py`
- **Tests**: `tests/test_optimization.py`
- **Examples**: `examples/optimization_demo.py`
- **Math docs**: `docs/optimization_formulation.md`
- **Full guide**: `docs/OPTIMIZATION_IMPLEMENTATION.md`

## Further Reading

1. Mathematical formulation: `docs/optimization_formulation.md`
2. Implementation guide: `docs/OPTIMIZATION_IMPLEMENTATION.md`
3. Full summary: `OPTIMIZATION_SUMMARY.md`
4. Run demo: `python examples/optimization_demo.py`
5. Run tests: `pytest tests/test_optimization.py`

---

**Pro Tip**: Start with `FRANK_WOLFE` backend. It's fast, requires no external solvers, and works great for most cases. Only upgrade to Gurobi if you need <50ms with integer programming on large problems (100+ opportunities).
