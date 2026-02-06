# VWAP System Quickstart

Get started with VWAP (Volume-Weighted Average Price) validation in under 5 minutes.

## Why VWAP?

**Problem:** Quoted price ≠ execution price due to slippage  
**Solution:** Validate VWAP before trading to ensure quality fills  
**Result:** No more "profitable signals" that lose money on execution

## 1-Minute Example

```python
from src.data.vwap import quick_vwap_check
from decimal import Decimal

# Your order book (bids and asks)
bids = [(Decimal("0.49"), 300), (Decimal("0.48"), 500)]
asks = [(Decimal("0.51"), 200), (Decimal("0.52"), 400)]

# Check VWAP for buying 150 contracts
result = quick_vwap_check(bids, asks, 150, "buy")

print(f"Quoted: ${asks[0][0]}")           # $0.51 (best ask)
print(f"VWAP: ${result.vwap_price}")      # $0.5133 (actual)
print(f"Slippage: {result.slippage_pct}%") # 0.65%
print(f"Quality: {result.execution_quality}") # EXCELLENT

if result.slippage_pct > Decimal("2.0"):
    print("⚠️ Too much slippage, reduce size!")
```

## Integration Patterns

### Pattern 1: Basic VWAP Check (Before Every Trade)

```python
from src.data.vwap import VWAPCalculator
from decimal import Decimal

calc = VWAPCalculator()

# Before executing...
orderbook = await platform.get_orderbook(market_id)

result = calc.calculate_vwap(
    orders=orderbook.asks,      # For buying
    quantity=500,
    side="buy",
    market_id=market_id,
    quoted_price=market.yes_price
)

if result.slippage_pct > Decimal("2.0"):
    print(f"High slippage: {result.slippage_pct}%, reducing size")
    quantity = quantity // 2  # Cut size in half
```

### Pattern 2: Strategy Integration (Best Practice)

```python
from src.execution.vwap_integration import VWAPTradingGate, StrategyVWAPIntegration
from src.strategies.base import BaseStrategy
from decimal import Decimal

class MyStrategy(BaseStrategy, StrategyVWAPIntegration):
    def __init__(self, platform):
        BaseStrategy.__init__(self, platform)
        
        # Create VWAP gate
        gate = VWAPTradingGate(
            max_slippage_pct=Decimal("2.0"),
            min_liquidity_contracts=500,
            max_spread_bps=300
        )
        StrategyVWAPIntegration.__init__(self, gate)
    
    async def generate_signals(self, markets):
        for market in markets:
            # Your signal logic here...
            if not self.is_signal(market):
                continue
            
            # Get order book
            orderbook = await self.platform.get_orderbook(market.id)
            
            # Enrich signal with VWAP
            enriched = self.enrich_signal_with_vwap(
                market_id=market.id,
                side=OrderSide.YES,
                quantity=500,
                signal_price=market.yes_price,
                orderbook=orderbook
            )
            
            # VWAP validation failed?
            if not enriched:
                continue
            
            # Check if still profitable after slippage
            if not enriched.is_profitable_after_slippage:
                print(f"Not profitable after slippage: {market.ticker}")
                continue
            
            # Signal passed all checks!
            yield enriched
```

### Pattern 3: Risk Management Gate (Recommended)

```python
from src.risk.vwap_checks import VWAPRiskManager, VWAPRiskConfig
from decimal import Decimal

# Create enhanced risk manager
risk_mgr = VWAPRiskManager(
    vwap_config=VWAPRiskConfig(
        max_slippage_pct=Decimal("2.0"),
        reject_on_high_slippage=True,
        enable_auto_adjustment=True
    )
)

# Before every trade...
is_allowed, adjusted_qty, reason = risk_mgr.check_trade_with_vwap(
    market_id=market_id,
    side=OrderSide.YES,
    quantity=500,
    orderbook=orderbook
)

if not is_allowed:
    print(f"Trade blocked: {reason}")
else:
    print(f"Trade approved: {adjusted_qty} contracts")
    # Execute...
```

## Common Scenarios

### Scenario 1: Small Order (Usually Fine)

```python
# Order: 50 contracts
# Book depth: 1000+ contracts
# Expected: EXCELLENT quality, <0.5% slippage
```

### Scenario 2: Medium Order (Check VWAP)

```python
# Order: 500 contracts
# Book depth: 800 contracts
# Expected: GOOD quality, 1-2% slippage
# Action: Validate VWAP before executing
```

### Scenario 3: Large Order (Split It!)

```python
# Order: 2000 contracts
# Book depth: 1000 contracts
# Expected: POOR quality, 5%+ slippage
# Action: Split into chunks or reduce size

from src.data.vwap import VWAPValidator

validator = VWAPValidator(calc)
chunks = validator.suggest_order_split(vwap_result, max_chunks=4)
# Result: [500, 500, 500, 500] instead of [2000]

for chunk in chunks:
    execute(chunk)
    await asyncio.sleep(10)  # Wait between executions
```

### Scenario 4: Thin Market (Avoid!)

```python
# Book depth: <200 contracts
# Spread: >5%
# Expected: INSUFFICIENT_LIQUIDITY
# Action: Skip this market entirely

liquidity = calc.calculate_liquidity_metrics(bids, asks, market_id)

if not liquidity.is_healthy:
    print(f"Unhealthy market: depth={liquidity.bid_depth + liquidity.ask_depth}")
    continue  # Skip
```

## Configuration Presets

### Conservative (Default)
```python
VWAPRiskConfig(
    max_slippage_pct=Decimal("1.0"),      # 1% max
    min_liquidity_contracts=1000,         # High liquidity required
    max_spread_bps=200,                   # 2% max spread
    reject_on_high_slippage=True,
    enable_auto_adjustment=True
)
```

### Moderate
```python
VWAPRiskConfig(
    max_slippage_pct=Decimal("2.0"),      # 2% max
    min_liquidity_contracts=500,
    max_spread_bps=300,                   # 3% max spread
    reject_on_high_slippage=True,
    enable_auto_adjustment=True
)
```

### Aggressive (Use Carefully!)
```python
VWAPRiskConfig(
    max_slippage_pct=Decimal("5.0"),      # 5% max (risky!)
    min_liquidity_contracts=200,
    max_spread_bps=500,                   # 5% spread OK
    reject_on_high_slippage=False,        # Allow bad fills
    enable_auto_adjustment=False
)
```

## Monitoring

### Live Dashboard

```bash
# Start monitoring dashboard
python examples/vwap_monitor_dashboard.py
```

**What it shows:**
- Real-time order book depth
- Live VWAP calculations at different sizes
- Spread tracking
- Execution quality statistics
- Alerts for unhealthy markets

### Get Statistics

```python
from src.data.vwap import VWAPMonitor

monitor = VWAPMonitor(calc)

# After some trades...
stats = monitor.get_execution_stats()

print(f"Total trades: {stats['total_executions']}")
print(f"Avg slippage: {stats['avg_slippage_pct']:.2f}%")
print(f"Quality distribution:")
for quality, count in stats['quality_distribution'].items():
    print(f"  {quality}: {count}")
```

## Testing Your Integration

### Run VWAP Tests
```bash
pytest tests/test_vwap.py -v
```

### Test with Live Data
```bash
# See examples/vwap_strategy_integration.py
python examples/vwap_strategy_integration.py
```

## Troubleshooting

### "Insufficient liquidity" errors?
- **Reduce position size**
- **Enable auto-adjustment:** `enable_auto_adjustment=True`
- **Lower minimum:** `min_liquidity_contracts=200`

### High slippage warnings?
- **Split large orders** into smaller chunks
- **Use limit orders** instead of market orders
- **Wait for better liquidity** (check again later)

### Signals getting blocked?
- **Check slippage threshold:** Maybe 2% is too strict for your markets
- **Review liquidity requirements:** 500 contracts might be too high
- **Increase spread tolerance:** Some markets naturally have wider spreads

### False positives (blocking good trades)?
- **Tune thresholds** based on observed market conditions
- **Use market-specific configs** (different thresholds per market type)
- **Analyze historical fills** to set realistic expectations

## Next Steps

1. **Read full docs:** `docs/VWAP_SYSTEM.md`
2. **See examples:** `examples/vwap_strategy_integration.py`
3. **Review implementation:** `VWAP_IMPLEMENTATION_COMPLETE.md`
4. **Understand the problem:** `xpost-analysis-rohonchain-polymarket-math.md`

## Key Takeaways

✅ Always check VWAP before executing  
✅ Slippage can turn profits into losses  
✅ Adjust position size based on liquidity  
✅ Split large orders to minimize impact  
✅ Monitor execution quality over time  
✅ Avoid thin markets entirely  

**Bottom line:** VWAP validation is not optional. It's the difference between profitable trading and donating money to market makers.
