# Market Making - Quick Start Guide

## 🚀 5-Minute Setup

### 1. Import & Configure
```python
from src.strategies.market_making import MarketMakingStrategy, MarketMakingConfig
from decimal import Decimal

config = MarketMakingConfig(
    base_spread=Decimal("0.04"),      # 4% spread
    max_inventory=50,                  # Max 50 contracts net
    quote_size=10,                     # 10 contracts per quote
    max_markets=10                     # Quote on 10 markets max
)

strategy = MarketMakingStrategy(config=config)
```

### 2. Register with Engine
```python
from src.engine.core import TradingEngine

engine = TradingEngine()
engine.register_strategy(strategy)
engine.register_platform(kalshi_platform)
```

### 3. Start Trading
```python
await engine.run()
```

---

## 📊 Pre-Configured Profiles

### Conservative (Recommended for Beginners)
```python
config = MarketMakingConfig(
    min_spread=Decimal("0.03"),       # 3% min
    base_spread=Decimal("0.06"),      # 6% base
    max_inventory=30,
    quote_size=5,
    max_markets=5
)
# Expected: $200-400/day, <10% drawdown, Sharpe 3.5+
```

### Balanced (Default)
```python
config = MarketMakingConfig()  # Uses defaults
# Expected: $300-600/day, 10-15% drawdown, Sharpe 3.0
```

### Aggressive (Experienced Traders)
```python
config = MarketMakingConfig(
    min_spread=Decimal("0.015"),      # 1.5% min
    base_spread=Decimal("0.03"),      # 3% base
    max_inventory=75,
    quote_size=20,
    max_markets=15
)
# Expected: $600-1000/day, 20-30% drawdown, Sharpe 2.5
```

---

## 🔍 Monitor Inventory

```python
# Check current positions
status = strategy.get_inventory_status()

for market_id, inv in status.items():
    print(f"{market_id}:")
    print(f"  Net: {inv['net_position']}")
    print(f"  Skew: {inv['skew_ratio']:.1%}")
```

---

## ⚠️ Key Parameters

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `base_spread` | 4% | Your profit margin per trade |
| `max_inventory` | 50 | Max contracts to hold one side |
| `quote_size` | 10 | Contracts per quote |
| `max_markets` | 10 | How many markets to quote |
| `min_liquidity` | $5,000 | Only quote liquid markets |

---

## 🎯 First Week Checklist

- [ ] Start with **Conservative** profile
- [ ] Enable **paper trading mode**
- [ ] Run for 2-3 days, monitor:
  - Fill rate (target: 15-30%)
  - Inventory skew (keep < 20)
  - Spread capture (track P&L)
- [ ] Review logs daily
- [ ] Tune `base_spread` if fill rate too high/low
- [ ] Switch to live with $500-1000 capital
- [ ] Scale up gradually

---

## 🚨 Common Issues & Fixes

### "Low fill rate (<10%)"
➜ **Reduce spreads**: `base_spread = Decimal("0.03")`

### "High inventory skew"
➜ **Lower threshold**: `inventory_skew_threshold = 8`

### "Getting picked off on news"
➜ **Faster quotes**: `max_quote_age_seconds = 15`

### "Too many markets"
➜ **Reduce**: `max_markets = 5`

---

## 📚 Full Documentation

- **Strategy Guide**: `docs/MARKET_MAKING_STRATEGY.md`
- **Code Examples**: `docs/MARKET_MAKING_EXAMPLES.md`
- **Research**: `docs/MARKET_MAKING_RESEARCH.md`
- **Implementation Summary**: `MARKET_MAKING_IMPLEMENTATION_SUMMARY.md`

---

## 💡 Pro Tips

1. **Start small** - Test with $500-1000 before scaling
2. **Monitor closely** - First 2 weeks are learning period
3. **Tune gradually** - Change one parameter at a time
4. **Track metrics** - Fill rate, spread, inventory, P&L
5. **Use alerts** - Set up notifications for skew > 30

---

## 🎓 Next Steps

1. **Week 1**: Paper trading + monitoring
2. **Week 2**: Live with conservative config
3. **Week 3-4**: Tune to balanced config
4. **Month 2+**: Apply to Kalshi MM Program
5. **Month 3+**: Scale to aggressive config

---

**Remember**: Market making is a marathon, not a sprint. Focus on consistent execution and risk management over quick profits.

**Questions?** See full documentation in `docs/` folder.
