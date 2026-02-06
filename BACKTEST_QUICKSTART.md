# Backtesting Quick Start Guide

Get started with PR3DICT backtesting in 5 minutes.

## Step 1: Generate Sample Data (if needed)

```bash
cd ~/.openclaw/workspace/pr3dict

python3 -m src.backtest.run \
  --generate-sample \
  --output data/sample.csv \
  --sample-markets 5 \
  --sample-days 30
```

## Step 2: Run Your First Backtest

```bash
python3 -m src.backtest.run \
  --strategy arbitrage \
  --data data/sample.csv \
  --start 2024-01-01 \
  --end 2024-01-31
```

## Step 3: View Results

Reports are saved to `./backtest_reports/`

Console output shows:
- Total return %
- Sharpe ratio
- Win rate
- Max drawdown
- Trade count

## Common Commands

### Test All Strategies
```bash
python3 -m src.backtest.run \
  --strategy all \
  --data data/sample.csv \
  --start 2024-01-01 \
  --end 2024-12-31
```

### Custom Settings
```bash
python3 -m src.backtest.run \
  --strategy market_making \
  --data data/sample.csv \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --balance 50000 \
  --commission 0.02 \
  --slippage 10
```

### Use Real Historical Data
```bash
python3 -m src.backtest.run \
  --strategy behavioral \
  --data-dir data/historical/ \
  --platforms kalshi polymarket \
  --start 2024-01-01 \
  --end 2024-12-31
```

## Programmatic Usage

```python
from pathlib import Path
from datetime import datetime
from decimal import Decimal

from src.backtest import BacktestEngine, BacktestConfig
from src.backtest.data import HistoricalDataLoader
from src.strategies.arbitrage import ArbitrageStrategy

# Load data
loader = HistoricalDataLoader()
loader.load_csv(Path("data/sample.csv"))

# Configure
config = BacktestConfig(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_balance=Decimal("10000"),
    commission_rate=Decimal("0.01"),
    slippage_bps=Decimal("5")
)

# Run
engine = BacktestEngine(
    data_loader=loader,
    strategies=[ArbitrageStrategy()],
    config=config
)

results = engine.run()

# Analyze
print(f"Final Balance: ${results['final_balance']}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## CLI Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--strategy` | Strategy name or "all" | arbitrage |
| `--start` | Start date (YYYY-MM-DD) | required |
| `--end` | End date (YYYY-MM-DD) | required |
| `--data` | Path to CSV file | auto-load |
| `--balance` | Initial balance USD | 10000 |
| `--commission` | Commission rate (decimal) | 0.01 |
| `--slippage` | Slippage in bps | 5 |
| `--max-positions` | Max concurrent positions | 10 |
| `--output` | Report output directory | ./backtest_reports |

## Interpreting Results

### Good Performance
- Sharpe Ratio: **> 1.5**
- Win Rate: **> 55%**
- Profit Factor: **> 1.5**
- Max Drawdown: **< 20%**

### Warning Signs
- Sharpe < 0.5 (poor risk-adjusted returns)
- Win rate < 45% (more losers than winners)
- Profit factor < 1.2 (barely profitable)
- Max drawdown > 30% (too risky)

## Historical Data Format

CSV with these columns:
```
timestamp,market_id,ticker,title,yes_price,no_price,volume,liquidity,close_time,resolved,platform
```

Example row:
```
2024-01-01T10:00:00,MKT-1,TICKER1,Will event occur?,0.65,0.34,10000,50000,2024-02-01T00:00:00,false,kalshi
```

## Troubleshooting

**"No historical data loaded"**
- Check data file exists
- Verify date range matches data
- Try `--generate-sample` first

**"Insufficient balance"**
- Increase `--balance`
- Reduce position sizes

**"ImportError"**
- Run from pr3dict/ directory
- Use `python3 -m src.backtest.run` format

## Next Steps

1. **Read full docs:** `BACKTESTING.md`
2. **Run test:** `python3 test_backtest.py`
3. **See examples:** `examples/backtest_example.py`
4. **Collect real data** and start optimizing!

---

**Need Help?** Check BACKTESTING.md for detailed documentation.
