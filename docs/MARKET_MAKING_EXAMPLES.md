# Market Making Strategy - Usage Examples

## Basic Usage

### 1. Initialize Strategy with Default Config

```python
from src.strategies.market_making import MarketMakingStrategy, MarketMakingConfig

# Use default configuration
strategy = MarketMakingStrategy()

# Or create custom config
config = MarketMakingConfig(
    base_spread=Decimal("0.03"),  # 3% spread
    max_inventory=40,
    quote_size=15
)
strategy = MarketMakingStrategy(config=config)
```

### 2. Register with Trading Engine

```python
from src.engine.core import TradingEngine
from src.platforms.kalshi import KalshiPlatform

# Initialize platform and engine
kalshi = KalshiPlatform(api_key="...", api_secret="...")
engine = TradingEngine()

# Register strategy
engine.register_strategy(strategy)
engine.register_platform(kalshi)

# Start trading
await engine.run()
```

---

## Configuration Examples

### Conservative Configuration (Low Risk)

For beginners or risk-averse traders:

```python
conservative_config = MarketMakingConfig(
    # Wide spreads for safety
    min_spread=Decimal("0.03"),      # 3% minimum
    base_spread=Decimal("0.06"),     # 6% base
    max_spread=Decimal("0.15"),      # 15% max
    
    # Tight inventory limits
    max_inventory=30,
    inventory_skew_threshold=8,
    rebalance_threshold=15,
    
    # Small quotes
    quote_size=5,
    max_position_size=50,
    
    # Conservative market selection
    max_markets=5,
    min_liquidity=Decimal("10000"),  # $10k minimum
    max_time_to_resolution_hours=72  # 3 days max
)

strategy = MarketMakingStrategy(config=conservative_config)
```

**Expected Performance:**
- Lower revenue (~$200-400/day)
- Lower risk (max drawdown <10%)
- Higher Sharpe ratio (3.5+)
- Very stable

### Aggressive Configuration (High Returns)

For experienced traders with good risk management:

```python
aggressive_config = MarketMakingConfig(
    # Tight spreads for volume
    min_spread=Decimal("0.015"),     # 1.5% minimum
    base_spread=Decimal("0.03"),     # 3% base
    max_spread=Decimal("0.08"),      # 8% max
    
    # Larger inventory tolerance
    max_inventory=75,
    inventory_skew_threshold=15,
    rebalance_threshold=30,
    
    # Larger quotes
    quote_size=20,
    max_position_size=150,
    
    # More markets
    max_markets=15,
    min_liquidity=Decimal("3000"),   # $3k minimum
    max_time_to_resolution_hours=168 # 7 days
)

strategy = MarketMakingStrategy(config=aggressive_config)
```

**Expected Performance:**
- Higher revenue (~$600-1000/day)
- Higher risk (max drawdown 20-25%)
- Lower Sharpe ratio (2.0-2.5)
- More volatile

### High-Frequency Configuration

Requires low-latency infrastructure:

```python
hft_config = MarketMakingConfig(
    # Very tight spreads
    min_spread=Decimal("0.01"),      # 1% minimum
    base_spread=Decimal("0.02"),     # 2% base
    
    # Rapid quote updates
    max_quote_age_seconds=5,         # Update every 5s
    price_change_requote_threshold=Decimal("0.01"),  # 1% threshold
    
    # Small quotes, many markets
    quote_size=3,
    max_markets=25,
    
    # Tight inventory control
    max_inventory=20,
    inventory_skew_threshold=5
)

strategy = MarketMakingStrategy(config=hft_config)
```

**Requirements:**
- Co-located server (low latency)
- High-performance infrastructure
- Automated monitoring

---

## Working with Inventory

### Monitoring Inventory

```python
# Get current inventory status
status = strategy.get_inventory_status()

for market_id, inv in status.items():
    print(f"Market: {market_id}")
    print(f"  YES: {inv['yes_contracts']} @ ${inv['avg_yes_price']:.3f}")
    print(f"  NO:  {inv['no_contracts']} @ ${inv['avg_no_price']:.3f}")
    print(f"  Net: {inv['net_position']} (skew: {inv['skew_ratio']:.2f})")
    print()
```

**Example Output:**
```
Market: KALSHI-BTC-100K-DEC24
  YES: 25 @ $0.650
  NO:  15 @ $0.380
  Net: +10 (skew: 0.25)

Market: KALSHI-ELECTION-2024
  YES: 10 @ $0.520
  NO:  30 @ $0.450
  Net: -20 (skew: -0.50)
```

### Manual Inventory Update

```python
# After an order fills, update inventory
strategy.update_inventory(
    market_id="KALSHI-BTC-100K-DEC24",
    side=OrderSide.YES,
    quantity=10,
    price=Decimal("0.65")
)
```

### Forcing Inventory Rebalance

```python
# Check if rebalancing needed
inv = strategy.inventory.get("KALSHI-BTC-100K-DEC24")

if abs(inv.net_position) > strategy.config.rebalance_threshold:
    # Generate exit signal
    exit_signal = await strategy.check_exit(
        position=current_position,
        market=current_market
    )
    
    if exit_signal:
        # Execute exit order
        await platform.place_order(
            market_id=exit_signal.market_id,
            side=exit_signal.side,
            order_type=OrderType.MARKET,
            quantity=abs(inv.net_position)
        )
```

---

## Real-World Scenarios

### Scenario 1: Normal Market Making

**Market**: Bitcoin above $100k by Dec 2024  
**Fair Value**: $0.50  
**Liquidity**: $15,000  
**Time to Close**: 48 hours  
**Current Inventory**: 0 (flat)

**Strategy Calculation:**

```python
# Fair value
fair_value = Decimal("0.50")

# Spread calculation
base_spread = Decimal("0.04")  # 4%
# No adjustments needed (good liquidity, >24h out, no inventory)
spread = base_spread

# Bid/Ask prices
bid = fair_value - (spread / 2) = $0.48
ask = fair_value + (spread / 2) = $0.52

# Quote both sides
# BUY YES @ $0.48 (10 contracts)
# BUY NO @ $0.48  (10 contracts) [equivalent to SELL YES @ $0.52]
```

**Outcome:**
- Both orders fill
- Bought 10 YES @ $0.48
- Bought 10 NO @ $0.48
- Guaranteed profit: $(1.00 - 0.48 - 0.48) × 10 = $0.40
- Return: 8.3% on capital deployed

### Scenario 2: Inventory Skew Management

**Market**: Same as above  
**Current Inventory**: +25 YES contracts (skewed bullish)

**Strategy Calculation:**

```python
# Fair value
fair_value = Decimal("0.50")

# Spread calculation
base_spread = Decimal("0.04")
# Inventory > 10, apply multiplier
spread = base_spread * Decimal("1.3") = Decimal("0.052")  # 5.2%

# Inventory adjustments
# 25 contracts = 2.5 increments of 10
# Adjustment = 2 × 1% = 2%
bid_adjustment = -Decimal("0.02")
ask_adjustment = -Decimal("0.02")

# Final prices
bid = Decimal("0.50") - Decimal("0.026") - Decimal("0.02") = $0.454
ask = Decimal("0.50") + Decimal("0.026") - Decimal("0.02") = $0.506

# Quote primarily rebalancing side
# BUY NO @ $0.494 (to reduce YES exposure)
# May skip or reduce YES bid
```

**Effect:**
- Makes buying NO more attractive (vs normal $0.48)
- Discourages buying more YES (lower bid)
- Helps rebalance inventory toward neutral

### Scenario 3: Approaching Resolution

**Market**: Election market  
**Time to Close**: 2 hours  
**Current Inventory**: +15 YES contracts  
**Fair Value**: $0.55

**Strategy Calculation:**

```python
# Time multiplier (< 6h)
spread = base_spread * Decimal("2.0") = Decimal("0.08")  # 8%

# Check exit condition
if time_to_close < 0.5 and abs(inventory.net_position) > 5:
    # FORCE EXIT
    # Place market order to sell 15 YES (buy 15 NO)
```

**Rationale:**
- 2 hours to resolution = high execution risk
- Can't afford to hold 15 YES if market resolves NO
- Better to take small loss than risk full position

**Exit Execution:**
```python
# Market order: BUY 15 NO
# If NO price is $0.48:
# Cost = 15 × $0.48 = $7.20
# Already invested in YES = 15 × (assume $0.52) = $7.80
# Lock in loss of $0.60, but eliminate directional risk
```

### Scenario 4: Adverse Selection Hit

**Market**: Fed rate decision  
**Initial Setup**:
- Quoted: BUY YES @ $0.48, BUY NO @ $0.48
- Market stable for 30 minutes

**News Breaks** (Fed announces hawkish stance):
- YES price jumps to $0.70 instantly
- Our $0.48 bid gets filled for 10 YES
- Market continues to $0.75

**Adverse Selection Loss:**
```python
# Bought YES @ $0.48
# Current price: $0.75
# Unrealized gain: +$0.27 per contract

# BUT: We bought AFTER information
# Fair value was already $0.70+
# We were adversely selected

# What actually happens:
# Bought 10 YES @ $0.48
# Also bought 10 NO @ $0.48 earlier (before news)
# NO now worth $0.25
# Net: (10 × $0.75 + 10 × $0.25) - $9.60 = +$0.40

# Still profitable due to spread, but gave up edge
```

**Protection Mechanisms:**
```python
# 1. Volume spike detection
if current_volume > avg_volume * 3.0:
    # Pause quoting, cancel orders
    pass

# 2. Price change detection
if abs(current_price - last_price) > Decimal("0.02"):
    # Cancel stale quotes, requote at new levels
    pass

# 3. Maximum quote age
if quote_age_seconds > 30:
    # Force refresh
    pass
```

---

## Integration Patterns

### Pattern 1: Standalone Market Maker

Run market making as the only strategy:

```python
from src.strategies.market_making import MarketMakingStrategy
from src.engine.core import TradingEngine

async def run_mm_only():
    strategy = MarketMakingStrategy()
    engine = TradingEngine()
    
    engine.register_strategy(strategy)
    
    # Main loop
    while True:
        markets = await platform.get_markets(status="open")
        signals = await strategy.scan_markets(markets)
        
        for signal in signals:
            # Place limit orders
            order = await platform.place_order(
                market_id=signal.market_id,
                side=signal.side,
                order_type=OrderType.LIMIT,
                quantity=strategy.config.quote_size,
                price=signal.target_price
            )
            
            # Track for inventory updates
            if order.status == OrderStatus.FILLED:
                strategy.update_inventory(
                    market_id=order.market_id,
                    side=order.side,
                    quantity=order.filled_quantity,
                    price=order.price
                )
        
        await asyncio.sleep(30)  # Refresh every 30s
```

### Pattern 2: Multi-Strategy (MM + Arbitrage)

Combine market making with arbitrage:

```python
from src.strategies.market_making import MarketMakingStrategy
from src.strategies.arbitrage import ArbitrageStrategy

# Initialize both
mm_strategy = MarketMakingStrategy()
arb_strategy = ArbitrageStrategy()

# Register both
engine.register_strategy(mm_strategy)
engine.register_strategy(arb_strategy)

# Engine will:
# 1. Get signals from both strategies
# 2. Prioritize arbitrage (higher strength)
# 3. Execute MM quotes when no arbitrage available
# 4. Track inventory separately for each strategy
```

### Pattern 3: Cross-Platform Market Making

Make markets on both Kalshi and Polymarket:

```python
kalshi_mm = MarketMakingStrategy(
    config=MarketMakingConfig(
        max_markets=8,
        base_spread=Decimal("0.04")
    )
)

polymarket_mm = MarketMakingStrategy(
    config=MarketMakingConfig(
        max_markets=8,
        base_spread=Decimal("0.045")  # Slightly wider for blockchain
    )
)

# Unified inventory tracking across platforms
class CrossPlatformInventory:
    def __init__(self):
        self.kalshi_inv = {}
        self.polymarket_inv = {}
    
    def get_net_exposure(self, event_id):
        """Get total exposure across both platforms for same event."""
        kalshi_net = self.kalshi_inv.get(event_id, 0)
        poly_net = self.polymarket_inv.get(event_id, 0)
        return kalshi_net + poly_net
```

---

## Testing & Validation

### Paper Trading Test

```python
# Enable paper mode
config = MarketMakingConfig()
config.paper_mode = True  # Add this to config

strategy = MarketMakingStrategy(config=config)

# Run for 24 hours, track:
# - Total quotes generated
# - Fill rate
# - Spread capture
# - Max inventory skew
# - Adverse selection events
```

### Backtesting

```python
# Load historical market data
historical_markets = load_historical_data("2024-01-01", "2024-12-31")

# Simulate strategy
total_pnl = Decimal("0")

for timestamp, markets in historical_markets:
    signals = await strategy.scan_markets(markets)
    
    for signal in signals:
        # Simulate fill (assume X% fill rate)
        if random.random() < 0.20:  # 20% fill rate
            # Calculate P&L
            entry_price = signal.target_price
            exit_price = get_exit_price(signal.market_id, timestamp + timedelta(hours=1))
            
            pnl = (exit_price - entry_price) * strategy.config.quote_size
            total_pnl += pnl

print(f"Backtest P&L: ${total_pnl}")
```

### Unit Tests

```python
import pytest
from decimal import Decimal

def test_spread_calculation():
    config = MarketMakingConfig(base_spread=Decimal("0.04"))
    strategy = MarketMakingStrategy(config=config)
    
    # Test normal market
    market = create_test_market(liquidity=Decimal("10000"), hours_to_close=48)
    inventory = InventoryTracker(market_id="TEST", net_position=0)
    
    spread = strategy._calculate_dynamic_spread(market, inventory)
    assert spread == Decimal("0.04")
    
    # Test low liquidity
    market.liquidity = Decimal("6000")
    spread = strategy._calculate_dynamic_spread(market, inventory)
    assert spread == Decimal("0.06")  # +2% for low liquidity

def test_inventory_adjustments():
    strategy = MarketMakingStrategy()
    
    # Test long position
    inventory = InventoryTracker(market_id="TEST")
    inventory.yes_contracts = 25
    inventory.no_contracts = 5
    # net_position = +20
    
    bid_adj, ask_adj = strategy._calculate_inventory_adjustments(inventory)
    
    # Should discourage buying more YES
    assert bid_adj < Decimal("0")
    assert ask_adj < Decimal("0")
```

---

## Troubleshooting

### Issue: Low Fill Rate (<10%)

**Causes:**
- Spreads too wide
- Quote prices not competitive
- Low market activity

**Solutions:**
```python
# Reduce spreads
config.base_spread = Decimal("0.03")  # Down from 0.04
config.min_spread = Decimal("0.015")  # Down from 0.02

# Increase quote size
config.quote_size = 15  # Up from 10
```

### Issue: High Inventory Skew

**Causes:**
- Too many fills on one side
- Market trending strongly
- Not rebalancing fast enough

**Solutions:**
```python
# More aggressive skew adjustments
config.skew_price_adjustment_per_10 = Decimal("0.015")  # 1.5% instead of 1%
config.inventory_skew_threshold = 8  # Down from 10

# Lower rebalance threshold
config.rebalance_threshold = 15  # Down from 20
```

### Issue: Frequent Adverse Selection

**Causes:**
- Quotes too stale
- Not detecting news events
- Spreads too tight

**Solutions:**
```python
# Faster quote updates
config.max_quote_age_seconds = 15  # Down from 30

# Wider spreads
config.base_spread = Decimal("0.05")  # Up from 0.04

# More sensitive price change detection
config.price_change_requote_threshold = Decimal("0.015")  # 1.5% instead of 2%
```

---

## Next Steps

1. **Start with paper trading** using conservative config
2. **Monitor performance** for 1-2 weeks
3. **Tune parameters** based on fill rate and P&L
4. **Gradually increase** to aggressive config as confidence grows
5. **Apply to Kalshi MM Program** after proving consistent performance
6. **Scale capital** once Sharpe > 2.5 for 3+ months

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-02
