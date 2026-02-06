# Parallel Execution Engine

## Overview

The Parallel Execution Engine enables atomic multi-leg arbitrage trades on Polygon, ensuring all legs execute within the same block (<30ms window) to lock in profit.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  ArbitrageExecutionEngine                    │
│  (High-level interface for detecting and executing arbs)    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    ParallelExecutor                          │
│  • Submit all orders simultaneously                          │
│  • Track confirmation across legs                            │
│  • Handle partial fills                                      │
│  • Rollback on incomplete arbitrage                          │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┬─────────────────┐
         ▼                       ▼                 ▼
┌─────────────────┐   ┌──────────────────┐   ┌─────────────┐
│ MetricsCollector│   │ PolygonOptimizer │   │RiskManager  │
│  • Fill rates   │   │  • Gas pricing   │   │ • Pre-flight│
│  • Exec times   │   │  • RPC failover  │   │ • Limits    │
│  • Slippage     │   │  • Retry logic   │   │ • Validation│
└─────────────────┘   └──────────────────┘   └─────────────┘
```

## Key Features

### 1. Atomic Execution
- All legs submit simultaneously via `asyncio.gather()`
- <30ms execution window for same-block guarantee
- Automatic rollback if any leg fails

### 2. Execution Strategies

#### Market Orders (Fast, High Slippage)
```python
strategy = ExecutionStrategy.MARKET
# Best for: High-confidence arbs, volatile markets
# Pros: Fastest execution (~10-20ms)
# Cons: Higher slippage (1-3%)
```

#### Limit Orders (Slow, Low Slippage)
```python
strategy = ExecutionStrategy.LIMIT
# Best for: Stable arbs, patient execution
# Pros: Minimal slippage (<0.5%)
# Cons: Slower, may not fill
```

#### Hybrid (Recommended)
```python
strategy = ExecutionStrategy.HYBRID
# Best for: Most arbitrage situations
# How it works:
#   1. Submit as limit orders
#   2. Wait 15ms
#   3. Convert unfilled to market orders
# Pros: Balance of speed and slippage
```

### 3. Polygon Optimizations

- **Batch Transactions**: Group operations to reduce gas
- **Gas Price Management**: Dynamic pricing based on network conditions
- **RPC Failover**: Automatic endpoint switching on failures
- **Retry Logic**: Exponential backoff for transient errors

### 4. Risk Integration

Pre-flight checks before every trade:
- ✅ Risk manager approval
- ✅ Capital availability
- ✅ Position size limits
- ✅ Platform connectivity

## Usage

### Basic Example

```python
from src.execution.integration import ArbitrageExecutionEngine
from src.execution.parallel_executor import ExecutionStrategy

# Initialize
engine = ArbitrageExecutionEngine(
    platforms=platforms,
    risk_manager=risk_manager,
    paper_mode=True  # Start with paper trading
)

# Scan markets for opportunities
markets = await platform.get_markets()

# Detect and execute
trades = await engine.scan_and_execute(
    markets=markets,
    strategy=ExecutionStrategy.HYBRID,
    max_opportunities=5
)

# Check results
for trade in trades:
    if trade.committed:
        print(f"✓ Profit: {trade.actual_profit}")
    else:
        print(f"✗ Failed: {trade.rolled_back}")
```

### Manual Execution

```python
from src.execution.parallel_executor import TradeLeg

# Define legs
legs = [
    TradeLeg(
        market_id="market_1",
        side=OrderSide.YES,
        quantity=100,
        target_price=Decimal("0.45"),
        platform="polymarket"
    ),
    TradeLeg(
        market_id="market_1",
        side=OrderSide.NO,
        quantity=100,
        target_price=Decimal("0.50"),
        platform="polymarket"
    )
]

# Execute
executor = ParallelExecutor(platforms, risk_manager)
trade = await executor.execute_arbitrage(
    legs=legs,
    strategy=ExecutionStrategy.HYBRID,
    expected_profit=Decimal("5.00")
)

# Check execution time
print(f"Execution time: {trade.execution_time_ms:.1f}ms")
print(f"Within block: {trade.execution_time_ms <= 30}")
```

### Monitoring & Metrics

```python
# Get real-time metrics
summary = executor.metrics.get_summary()
print(f"Success rate: {summary['success_rate_pct']}%")
print(f"Avg execution time: {summary['by_strategy']['hybrid']['avg_execution_time_ms']}ms")

# Strategy performance
perf = executor.metrics.get_strategy_performance("hybrid")
print(f"Fill rate: {perf['fill_rate']['avg_pct']}%")
print(f"Avg slippage: {perf['slippage']['avg_pct']}%")

# Recent trades
recent = executor.metrics.get_recent_trades(limit=10)
for trade in recent:
    print(f"{trade['trade_id']}: {trade['success']} - {trade['time_ms']}ms")

# Export for analysis
import json
data = executor.metrics.export_metrics()
with open("metrics.json", "w") as f:
    json.dump(data, f, indent=2)
```

## Integration with Trading Engine

### Modify `engine/core.py`

```python
from src.execution.integration import ArbitrageExecutionEngine
from src.execution.parallel_executor import ExecutionStrategy

class TradingEngine:
    def __init__(self, ...):
        # ... existing code ...
        
        # Add parallel executor
        self.arb_engine = ArbitrageExecutionEngine(
            platforms=self.platforms,
            risk_manager=self.risk,
            paper_mode=self.config.paper_mode
        )
    
    async def _scan_entries(self, markets):
        # ... existing strategy scanning ...
        
        # Add arbitrage scanning
        if "arbitrage" in self.strategies:
            trades = await self.arb_engine.scan_and_execute(
                markets=markets,
                strategy=ExecutionStrategy.HYBRID,
                max_opportunities=3
            )
            
            # Send notifications
            for trade in trades:
                if trade.committed and self.notifications:
                    await self.notifications.send_trade_completed(
                        trade_id=trade.trade_id,
                        profit=float(trade.actual_profit),
                        execution_time_ms=trade.execution_time_ms
                    )
```

## Configuration

### ExecutionConfig

```python
from src.execution.parallel_executor import ExecutionConfig

config = ExecutionConfig(
    # Timing
    max_execution_time_ms=30,  # Polygon block time
    order_submission_delay_ms=0,  # No delay for speed
    
    # Retry
    max_retries=3,
    retry_delay_ms=50,
    
    # Slippage
    max_slippage_pct=Decimal("0.03"),  # 3% max
    slippage_check_enabled=True,
    
    # Gas (Polygon)
    use_batch_transactions=True,
    max_gas_price_gwei=Decimal("500"),
    gas_price_multiplier=Decimal("1.2"),
    
    # Fallback
    hybrid_fallback_timeout_ms=15,
    cancel_on_partial_fill=True,
    
    # RPC endpoints
    primary_rpc="https://polygon-rpc.com",
    fallback_rpcs=[
        "https://rpc-mainnet.matic.network",
        "https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY"
    ]
)

executor = ParallelExecutor(
    platforms=platforms,
    risk_manager=risk_manager,
    config=config
)
```

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/test_parallel_executor.py -v

# Run specific test
pytest tests/test_parallel_executor.py::test_market_execution -v

# Run with coverage
pytest tests/test_parallel_executor.py --cov=src/execution
```

### Paper Trading

Always test in paper mode first:

```python
# Enable paper mode
engine = ArbitrageExecutionEngine(
    platforms=platforms,
    risk_manager=risk_manager,
    paper_mode=True  # ← Simulates execution
)

# Run for several days
# Monitor metrics, check for issues
# Verify strategy performance

# Once confident, switch to live
engine.paper_mode = False
```

## Performance Targets

| Metric | Target | Excellent |
|--------|--------|-----------|
| Execution Time | <30ms | <20ms |
| Fill Rate | >90% | >95% |
| Success Rate | >85% | >90% |
| Slippage (Hybrid) | <2% | <1% |
| Rollback Rate | <10% | <5% |
| Within-Block Rate | >80% | >90% |

## Monitoring Checklist

Daily:
- [ ] Success rate >85%
- [ ] Average execution time <30ms
- [ ] Rollback rate <10%
- [ ] Check error logs

Weekly:
- [ ] Review slippage trends
- [ ] Optimize gas settings
- [ ] Analyze failed trades
- [ ] Update RPC endpoints if needed

Monthly:
- [ ] Compare strategy performance
- [ ] Adjust position sizing
- [ ] Review risk limits
- [ ] Export metrics for deep analysis

## Troubleshooting

### High Rollback Rate

**Symptoms**: >15% of trades rolling back

**Causes**:
1. Execution timeout too tight
2. Poor RPC performance
3. Network congestion
4. Low liquidity markets

**Solutions**:
```python
# Increase timeout
config.max_execution_time_ms = 50

# Use faster RPCs
config.primary_rpc = "premium_endpoint"

# Increase gas for priority
config.gas_price_multiplier = Decimal("1.5")

# Filter low liquidity
min_liquidity = Decimal("5000")
```

### High Slippage

**Symptoms**: Actual profit < 70% of expected

**Causes**:
1. Using market orders in thin markets
2. Large position sizes
3. Volatile market conditions

**Solutions**:
```python
# Switch to hybrid/limit
strategy = ExecutionStrategy.HYBRID

# Reduce position size
risk_config.max_position_size = Decimal("50")

# Tighter slippage limits
config.max_slippage_pct = Decimal("0.02")
```

### Slow Execution

**Symptoms**: >50ms average execution time

**Causes**:
1. RPC latency
2. Network congestion
3. Sequential order submission

**Solutions**:
```python
# Use lower latency RPC
config.primary_rpc = "low_latency_endpoint"

# Ensure parallel submission
# (already implemented, but verify logs)

# Use market orders for speed
strategy = ExecutionStrategy.MARKET
```

## Best Practices

1. **Always test in paper mode first**
2. **Monitor execution time closely** (must be <30ms)
3. **Use hybrid strategy** for best balance
4. **Set conservative slippage limits** (2-3%)
5. **Have fallback RPCs configured**
6. **Review metrics daily**
7. **Start with small position sizes**
8. **Gradually increase as confidence builds**

## Advanced: Custom Strategies

```python
from src.execution.parallel_executor import ParallelExecutor

class CustomExecutor(ParallelExecutor):
    async def _execute_custom(self, trade):
        """Custom execution logic."""
        # Your implementation
        pass

# Use custom executor
executor = CustomExecutor(platforms, risk_manager)
```

## API Reference

### ParallelExecutor

- `execute_arbitrage(legs, strategy, expected_profit)` - Execute multi-leg trade
- `get_active_trades()` - Get currently executing trades
- `get_metrics_summary()` - Get execution metrics

### ArbitrageExecutionEngine

- `detect_binary_complement_arb(markets, min_profit_pct)` - Detect YES+NO<$1 arbs
- `detect_cross_platform_arb(markets, min_differential)` - Detect price differentials
- `execute_opportunity(opportunity, strategy)` - Execute detected opportunity
- `scan_and_execute(markets, strategy, max_opportunities)` - Full scan and execute cycle

### MetricsCollector

- `record_trade(trade)` - Record trade metrics
- `get_summary()` - Get aggregated statistics
- `get_strategy_performance(strategy)` - Get strategy-specific metrics
- `get_recent_trades(limit)` - Get recent trade list
- `export_metrics()` - Export all metrics for analysis

## Support

For issues or questions:
1. Check logs: `logs/execution.log`
2. Review metrics: `executor.metrics.get_summary()`
3. Run tests: `pytest tests/test_parallel_executor.py`
4. Check documentation: This file

## Changelog

### v1.0.0 (2026-02-02)
- Initial release
- Parallel execution engine
- Three execution strategies (market/limit/hybrid)
- Polygon optimizations
- Comprehensive metrics
- Risk integration
- Paper trading mode
