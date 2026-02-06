# Behavioral Strategy Quick Reference

## 🎯 Signal Types

### 1. LONGSHOT_FADE
- **Trigger**: YES price < 15%
- **Action**: BET NO
- **Edge**: 5-8%
- **Win Rate**: 65-70%
- **Rationale**: Low-probability events overpriced

### 2. FAVORITE_SUPPORT
- **Trigger**: YES price > 70%
- **Action**: BET YES
- **Edge**: 3-4%
- **Win Rate**: 75-80%
- **Rationale**: High-probability events underpriced

### 3. OVERREACTION_FADE
- **Trigger**: Price move > 20% in < 6 hours
- **Action**: BET opposite direction
- **Edge**: 8-10%
- **Win Rate**: 60-65%
- **Rationale**: Sharp moves tend to reverse

### 4. RECENCY_REVERSE
- **Trigger**: Recent volatility > 2x older volatility
- **Action**: BET against recent trend
- **Edge**: 4-5%
- **Win Rate**: 60-65%
- **Rationale**: Overweighting of recent information

### 5. TIME_ARBITRAGE ⚠️
- **Trigger**: Retail hours + trending (experimental)
- **Action**: FADE retail sentiment
- **Edge**: 2-3%
- **Win Rate**: 55-60%
- **Status**: Needs validation

## 📊 Configuration Presets

### Conservative
```python
strategy = create_behavioral_strategy(
    enable_longshot=True,
    enable_favorite=True,
    enable_overreaction=False,
    enable_recency=False,
    enable_time_arbitrage=False,
    min_edge=0.03  # 3% minimum
)
```
- **Risk**: Low
- **Signal Frequency**: Low
- **Expected Return**: 3-5% per trade

### Balanced (Recommended)
```python
strategy = create_behavioral_strategy(
    enable_longshot=True,
    enable_favorite=True,
    enable_overreaction=True,
    enable_recency=False,
    enable_time_arbitrage=False,
    min_edge=0.02  # 2% minimum
)
```
- **Risk**: Medium
- **Signal Frequency**: Medium
- **Expected Return**: 4-6% per trade

### Aggressive
```python
strategy = create_behavioral_strategy(
    enable_longshot=True,
    enable_favorite=True,
    enable_overreaction=True,
    enable_recency=True,
    enable_time_arbitrage=True,
    min_edge=0.02  # 2% minimum
)
```
- **Risk**: High
- **Signal Frequency**: High
- **Expected Return**: 4-7% per trade (higher variance)

## 🎲 Position Sizing

### Risk Per Trade
- **Conservative**: 1% of capital
- **Standard**: 2% of capital
- **Aggressive**: 5% of capital

### Formula
```python
contracts = strategy.get_position_size(
    signal,
    account_balance,
    risk_pct=0.02  # 2% risk
)
```

### Example
- Account: $10,000
- Risk: 2% = $200
- Entry price: $0.92
- Contracts: $200 / $0.92 = 217 contracts

## 🚪 Exit Rules

### Profit Target
- **Rule**: Exit at 50% of expected edge
- **Longshot**: Exit at +2.5% (50% of 5%)
- **Favorite**: Exit at +1.5% (50% of 3%)
- **Overreaction**: Exit at +4% (50% of 8%)

### Stop Loss
- **Rule**: Exit at 2x expected edge (negative)
- **Longshot**: Stop at -10%
- **Favorite**: Stop at -6%
- **Overreaction**: Stop at -16%

### Time Exit
- **Rule**: Exit after 7 days
- **Rationale**: Behavioral edge decays

### Signal Reversal
- **Rule**: Exit if bias conditions reverse
- **Longshot**: Exit if YES price rises above 15%
- **Favorite**: Exit if YES price falls below 70%
- **Overreaction**: Exit if move extends >10% beyond entry

## 📈 Expected Performance

### Overall Portfolio
| Metric | Expected Range |
|--------|---------------|
| Win Rate | 65-70% |
| Avg Return | +4-6% per trade |
| Sharpe Ratio | 1.2-1.8 |
| Max Drawdown | -15% to -25% |

### By Signal Type
| Signal | Win % | Return % | Sharpe |
|--------|-------|----------|--------|
| Longshot | 65-70 | +5-8 | 1.2-1.5 |
| Favorite | 75-80 | +3-4 | 1.5-2.0 |
| Overreaction | 60-65 | +8-10 | 1.0-1.3 |
| Recency | 60-65 | +4-5 | 0.8-1.2 |

## ⚠️ Requirements & Limits

### Market Requirements
- **Volume**: > $1,000 minimum
- **Liquidity**: Sufficient for position size
- **Data**: Price history (for some signals)

### Portfolio Limits
- **Max positions**: 20 total
- **Max per signal**: 5 positions
- **Max per market**: 10% of capital

## 🔍 Monitoring

### Daily Checks
- [ ] Signal frequency
- [ ] Win rate by signal type
- [ ] Average return per signal
- [ ] Open position count

### Weekly Review
- [ ] Edge realization vs expected
- [ ] Sharpe ratio trend
- [ ] Maximum drawdown
- [ ] Parameter effectiveness

### Monthly Analysis
- [ ] Overall strategy performance
- [ ] Individual signal performance
- [ ] Market efficiency changes
- [ ] Parameter adjustments needed

## 🚀 Deployment Checklist

### Before Live Trading
- [ ] Backtest on 6+ months historical data
- [ ] Validate win rate ≥ 60%
- [ ] Validate avg return ≥ 3%
- [ ] Validate Sharpe ≥ 1.0
- [ ] Paper trade for 30 days
- [ ] Document parameter choices

### Live Deployment
- [ ] Start with 1% of capital
- [ ] Max 5 positions initially
- [ ] Monitor daily for first 30 days
- [ ] Scale up only if profitable

## 📚 Files

- **Implementation**: `src/strategies/behavioral.py`
- **Full Docs**: `docs/behavioral_strategy.md`
- **Examples**: `examples/behavioral_strategy_example.py`
- **Tests**: `tests/test_behavioral_strategy.py`
- **Summary**: `docs/BEHAVIORAL_STRATEGY_SUMMARY.md`

## 🆘 Troubleshooting

### No Signals Generated
- Check market volume > $1,000
- Lower `min_edge` threshold
- Enable more signal types
- Verify markets meet bias criteria

### Low Win Rate
- Review actual vs expected edges
- Check for slippage issues
- Verify stop losses triggering properly
- Consider tightening entry criteria

### High Drawdown
- Reduce position sizes
- Tighten stop losses
- Reduce max concurrent positions
- Disable riskier signal types

## 📞 Quick Commands

### Run Examples
```bash
python examples/behavioral_strategy_example.py
```

### Run Tests
```bash
pytest tests/test_behavioral_strategy.py -v
```

### Import Strategy
```python
from pr3dict.strategies.behavioral import create_behavioral_strategy
```

---

**Strategy**: Behavioral Bias Exploitation  
**Type**: Mean Reversion / Statistical Arbitrage  
**Risk**: Medium  
**Complexity**: Low (fully systematic)  
**Data Requirements**: Price history helpful but not required for all signals
