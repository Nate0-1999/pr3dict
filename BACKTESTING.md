# PR3DICT Backtesting Framework

## Overview

The PR3DICT backtesting framework allows you to validate trading strategies against historical market data before deploying them to live trading. It simulates the complete trading lifecycle including order execution, position management, and P&L calculation with realistic commission and slippage modeling.

## Key Features

- **Historical Data Replay**: Chronological replay of market snapshots without look-ahead bias
- **Strategy Simulation**: Test any strategy implementing the `TradingStrategy` interface
- **Realistic Execution**: Commission and slippage modeling for accurate P&L
- **Risk Management**: Optional risk manager integration identical to live trading
- **Performance Metrics**: Industry-standard metrics (Sharpe, Sortino, max drawdown, win rate)
- **Parallel Testing**: Test multiple strategies simultaneously
- **Comprehensive Reports**: Text and JSON reports with trade logs and equity curves

## Architecture

The backtesting framework mirrors the live trading engine architecture:

```
src/backtest/
├── __init__.py          # Module exports
├── engine.py            # BacktestEngine (mirrors TradingEngine)
├── data.py              # HistoricalDataLoader for market snapshots
├── metrics.py           # Performance calculations
├── report.py            # Report generation
└── run.py               # CLI tool
```

### Key Differences from Live Trading

| Live Engine | Backtest Engine |
|-------------|-----------------|
| Async operations | Synchronous replay |
| Real-time market feeds | Historical data snapshots |
| Actual order fills | Simulated fills with slippage |
| Live risk exposure | Simulated P&L tracking |

## Quick Start

### 1. Generate Sample Data

If you don't have historical data yet, generate sample data for testing:

```bash
cd ~/.openclaw/workspace/pr3dict

python -m src.backtest.run \
  --generate-sample \
  --output data/sample_data.csv \
  --sample-markets 5 \
  --sample-days 30
```

This creates realistic synthetic market data with price movements, volume, and spreads.

### 2. Run a Basic Backtest

Test the arbitrage strategy over the generated data:

```bash
python -m src.backtest.run \
  --strategy arbitrage \
  --data data/sample_data.csv \
  --start 2024-01-01 \
  --end 2024-01-31
```

### 3. View Results

The backtest outputs performance metrics to the console and saves detailed reports to `./backtest_reports/`:

```
===========================================
PR3DICT BACKTEST REPORT
===========================================

Period: 2024-01-01 to 2024-01-31 (30 days)

Returns:
  Total Return:           15.34%
  Annualized Return:      224.56%

Risk-Adjusted:
  Sharpe Ratio:             2.34
  Sortino Ratio:            3.21
  Max Drawdown:            -8.45%

Trading:
  Total Trades:               42
  Win Rate:                65.00%
  Profit Factor:            2.14
```

## Usage Examples

### Test All Strategies in Parallel

```bash
python -m src.backtest.run \
  --strategy all \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --balance 50000
```

### Custom Commission and Slippage

```bash
python -m src.backtest.run \
  --strategy market_making \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --commission 0.02 \
  --slippage 10
```

### Load Multiple Data Files

```bash
python -m src.backtest.run \
  --strategy arbitrage \
  --data-dir data/historical/ \
  --platforms kalshi polymarket \
  --start 2024-01-01 \
  --end 2024-12-31
```

### Disable Risk Management

```bash
python -m src.backtest.run \
  --strategy behavioral \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --no-risk-manager
```

## Historical Data Format

The framework expects CSV files with the following columns:

```csv
timestamp,market_id,ticker,title,yes_price,no_price,volume,liquidity,close_time,resolved,platform
2024-01-01T10:00:00,MARKET-1,TICKER1,Will event occur?,0.65,0.34,10000,50000,2024-02-01T00:00:00,false,kalshi
2024-01-01T10:05:00,MARKET-1,TICKER1,Will event occur?,0.66,0.33,11000,50000,2024-02-01T00:00:00,false,kalshi
```

### Column Descriptions

- `timestamp`: ISO 8601 datetime when snapshot was taken
- `market_id`: Unique market identifier
- `ticker`: Short market symbol
- `title`: Market question/description
- `yes_price`: Current YES contract price (0-1)
- `no_price`: Current NO contract price (0-1)
- `volume`: Total trading volume
- `liquidity`: Available liquidity
- `close_time`: When market closes/resolves
- `resolved`: Boolean indicating if market is resolved
- `platform`: Platform name (kalshi, polymarket, etc.)

### Data Collection

To collect real historical data:

1. **Manual CSV Export**: Export market snapshots from platform APIs at regular intervals
2. **Automated Collection**: Use a cron job to poll markets and append to CSV
3. **API Archives**: Some platforms provide historical data APIs

Example collection script:

```python
import asyncio
import csv
from datetime import datetime
from src.platforms.kalshi import KalshiPlatform

async def collect_snapshot():
    platform = KalshiPlatform()
    await platform.connect()
    
    markets = await platform.get_markets(status="open", limit=100)
    
    with open('data/historical/kalshi.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for market in markets:
            writer.writerow([
                datetime.now().isoformat(),
                market.id,
                market.ticker,
                market.title,
                str(market.yes_price),
                str(market.no_price),
                str(market.volume),
                str(market.liquidity),
                market.close_time.isoformat(),
                str(market.resolved).lower(),
                'kalshi'
            ])

# Run every 5 minutes via cron
asyncio.run(collect_snapshot())
```

## Performance Metrics

### Returns

- **Total Return**: `(Final Balance - Initial Balance) / Initial Balance`
- **Annualized Return**: Extrapolated yearly return assuming constant growth

### Risk-Adjusted Returns

- **Sharpe Ratio**: `(Return - Risk-Free Rate) / Volatility`
  - Measures return per unit of total risk
  - >1.0 is good, >2.0 is excellent
  
- **Sortino Ratio**: `(Return - Risk-Free Rate) / Downside Volatility`
  - Like Sharpe but only considers downside risk
  - Better for asymmetric strategies

### Drawdown

- **Max Drawdown**: Largest peak-to-trough decline
- **Max Drawdown Duration**: Time to recover from max drawdown

### Trade Statistics

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss (>1.5 is strong)
- **Avg Win/Loss**: Average profit on wins vs. average loss on losses

## Advanced Features

### Custom Strategy Testing

To backtest your own strategy:

1. **Implement TradingStrategy interface**:

```python
from src.strategies.base import TradingStrategy, Signal
from src.platforms.base import Market, Position, OrderSide

class MyStrategy(TradingStrategy):
    @property
    def name(self) -> str:
        return "my_strategy"
    
    async def scan_markets(self, markets: List[Market]) -> List[Signal]:
        signals = []
        for market in markets:
            if self._meets_criteria(market):
                signals.append(Signal(
                    market_id=market.id,
                    market=market,
                    side=OrderSide.YES,
                    strength=0.8,
                    reason="Custom criteria met"
                ))
        return signals
    
    async def check_exit(self, position: Position, market: Market) -> Optional[Signal]:
        if self._should_exit(position, market):
            return Signal(
                market_id=position.market_id,
                market=market,
                side=OrderSide.NO,
                strength=1.0,
                reason="Exit criteria met"
            )
        return None
```

2. **Add to AVAILABLE_STRATEGIES** in `run.py`:

```python
AVAILABLE_STRATEGIES = {
    'arbitrage': ArbitrageStrategy,
    'market_making': MarketMakingStrategy,
    'behavioral': BehavioralEdgeStrategy,
    'my_strategy': MyStrategy,  # Add here
}
```

3. **Run backtest**:

```bash
python -m src.backtest.run --strategy my_strategy --start 2024-01-01 --end 2024-12-31
```

### Parameter Optimization

Test multiple parameter combinations:

```python
from src.backtest import BacktestEngine, BacktestConfig
from src.backtest.data import HistoricalDataLoader

# Load data once
loader = HistoricalDataLoader()
loader.load_csv(Path('data/historical.csv'))

# Test different thresholds
results = []
for threshold in [0.05, 0.10, 0.15, 0.20]:
    strategy = ArbitrageStrategy(min_spread=threshold)
    
    engine = BacktestEngine(
        data_loader=loader,
        strategies=[strategy],
        config=BacktestConfig(...)
    )
    
    result = engine.run()
    results.append((threshold, result))

# Find best parameters
best = max(results, key=lambda r: r[1]['sharpe_ratio'])
print(f"Optimal threshold: {best[0]}")
```

### Walk-Forward Analysis

Prevent overfitting by testing on out-of-sample data:

```python
# In-sample optimization (2024 Q1)
train_engine = BacktestEngine(
    data_loader=loader,
    strategies=[optimized_strategy],
    config=BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31)
    )
)
train_results = train_engine.run()

# Out-of-sample validation (2024 Q2)
test_engine = BacktestEngine(
    data_loader=loader,
    strategies=[optimized_strategy],
    config=BacktestConfig(
        start_date=datetime(2024, 4, 1),
        end_date=datetime(2024, 6, 30)
    )
)
test_results = test_engine.run()

# Compare performance
if test_results['sharpe_ratio'] > 0.8 * train_results['sharpe_ratio']:
    print("Strategy generalizes well!")
else:
    print("Warning: Performance degradation on out-of-sample data")
```

## Avoiding Common Pitfalls

### 1. Look-Ahead Bias

❌ **Wrong**: Using future information in decisions

```python
# BAD: Using end-of-day close to make decisions during the day
if market.close_price > market.open_price:
    return Signal(...)
```

✅ **Correct**: Only use information available at decision time

```python
# GOOD: Only using current and past data
if market.yes_price > self.moving_average:
    return Signal(...)
```

### 2. Survivorship Bias

❌ **Wrong**: Only testing on markets that resolved successfully

✅ **Correct**: Include all markets, including cancelled/void ones

### 3. Unrealistic Fills

❌ **Wrong**: Assuming instant fills at exact prices

✅ **Correct**: Model slippage and partial fills

The framework automatically adds slippage:
```python
fill_price = market_price + (market_price * slippage_bps / 10000)
```

### 4. Over-Optimization

❌ **Wrong**: Tuning 20 parameters until backtest looks perfect

✅ **Correct**: Use simple strategies, validate out-of-sample

### 5. Ignoring Costs

❌ **Wrong**: Not modeling commission and fees

✅ **Correct**: Set realistic commission rates (1-2% for prediction markets)

## Integration with Live Trading

Once backtest results are satisfactory:

1. **Review performance metrics**:
   - Sharpe ratio >1.0
   - Max drawdown <20%
   - Win rate >55%
   - Profit factor >1.5

2. **Validate on recent data** (last 30 days)

3. **Paper trade** first:
   ```python
   from src.engine.main import main
   
   # Edit config to enable paper mode
   config = EngineConfig(
       paper_mode=True,  # No real orders
       strategies=["my_strategy"]
   )
   
   main()
   ```

4. **Start with small position sizes** in live mode

5. **Monitor closely** for first week

## CLI Reference

```bash
python -m src.backtest.run [OPTIONS]
```

### Required Arguments

- `--strategy`: Strategy to test (`arbitrage`, `market_making`, `behavioral`, `all`)
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)

### Optional Arguments

- `--data PATH`: Path to historical data CSV
- `--data-dir PATH`: Directory with multiple CSV files
- `--balance FLOAT`: Initial balance in USD (default: 10000)
- `--commission FLOAT`: Commission rate (default: 0.01)
- `--slippage FLOAT`: Slippage in basis points (default: 5)
- `--max-positions INT`: Max concurrent positions (default: 10)
- `--platforms LIST`: Platforms to include (default: kalshi)
- `--output PATH`: Output directory for reports
- `--no-risk-manager`: Disable risk management

### Sample Data Generation

- `--generate-sample`: Generate synthetic data
- `--sample-markets INT`: Number of markets (default: 5)
- `--sample-days INT`: Number of days (default: 30)

## Troubleshooting

### "No historical data loaded"

**Cause**: No CSV files found or date range doesn't match data

**Solution**:
1. Verify data file exists: `ls data/historical/*.csv`
2. Check date ranges in CSV match `--start` and `--end`
3. Generate sample data: `--generate-sample`

### "Insufficient balance for trade"

**Cause**: Commission + position cost exceeds available balance

**Solution**:
1. Increase `--balance`
2. Reduce position sizes in strategy
3. Lower `--commission` rate

### "ImportError: No module named 'src'"

**Cause**: Running from wrong directory

**Solution**:
```bash
cd ~/.openclaw/workspace/pr3dict
python -m src.backtest.run ...
```

### Poor Backtest Performance

**Cause**: Strategy not suitable for historical conditions

**Solution**:
1. Review trade log in report
2. Adjust strategy parameters
3. Test on different time periods
4. Consider market regime changes

## Best Practices

1. **Start Simple**: Test basic strategies before complex ones
2. **Validate Assumptions**: Check if strategy logic makes sense
3. **Test Multiple Periods**: Bull markets, bear markets, sideways
4. **Compare to Baseline**: Does strategy beat buy-and-hold?
5. **Document Changes**: Track what parameters were tested
6. **Use Version Control**: Git commit before/after parameter changes
7. **Monitor Correlation**: Test strategies on same data to check correlation
8. **Plan for Failure**: What if win rate drops 10% in live trading?

## Next Steps

- **Collect Real Data**: Start gathering market snapshots
- **Refine Strategies**: Use backtest insights to improve logic
- **Paper Trade**: Validate in real-time without risk
- **Go Live**: Deploy with conservative position sizing

## Support

For issues or questions:
1. Check logs in `backtest_reports/`
2. Review TRADING.md for strategy documentation
3. Examine ARCHITECTURE.md for system design

---

**Remember**: Past performance doesn't guarantee future results. Backtesting shows what *could* have happened, not what *will* happen. Always validate with paper trading before risking real capital.
