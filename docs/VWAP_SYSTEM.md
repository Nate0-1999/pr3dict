## VWAP (Volume-Weighted Average Price) System

**Last Updated:** 2026-02-02

### Overview

The VWAP system prevents the critical "quoted price ≠ execution price" problem in prediction markets. It validates that trading signals will execute at acceptable prices before placing orders, accounting for:

- Order book depth and liquidity
- Price slippage at different volumes
- Bid-ask spreads
- Market impact
- Execution quality

### Problem Statement

**From X Post Analysis:** "The quoted mid-price you see is NOT what you'll get when you execute. You need to check VWAP (volume-weighted average price) against actual order book depth."

**Example Failure Scenario:**
```
Signal: BUY 1000 contracts @ $0.52 (quoted mid)
Reality: 
  - 200 @ $0.52 (best ask)
  - 300 @ $0.54 (+4%)
  - 500 @ $0.58 (+11%)
  
Actual VWAP: $0.556 (7% slippage!)
Result: "Profit" signal becomes a LOSS after execution
```

### Architecture

#### Core Components

**1. VWAPCalculator** (`src/data/vwap.py`)
- Calculates volume-weighted average price from order book
- Accounts for depth at each price level
- Estimates slippage for different order sizes
- Generates price impact curves

**2. VWAPValidator** (`src/data/vwap.py`)
- Validates execution quality against thresholds
- Checks liquidity sufficiency
- Validates spread width
- Suggests order splits for large trades

**3. VWAPMonitor** (`src/data/vwap.py`)
- Tracks execution history
- Records liquidity snapshots
- Generates quality statistics
- Monitors market health

**4. HistoricalVWAPAnalyzer** (`src/data/vwap.py`)
- Analyzes past trades via Alchemy Polygon RPC
- Builds price impact models
- Detects low-liquidity traps
- Learns typical slippage patterns

**5. VWAPTradingGate** (`src/execution/vwap_integration.py`)
- Quality gate for trading signals
- Validates signals before execution
- Auto-adjusts position sizes
- Blocks poor-quality executions

**6. VWAPRiskManager** (`src/risk/vwap_checks.py`)
- Extends base risk manager with VWAP checks
- Enforces slippage limits
- Requires minimum liquidity
- Prevents execution on wide spreads

### Data Models

#### VWAPResult
```python
@dataclass
class VWAPResult:
    market_id: str
    side: str  # "buy" or "sell"
    target_quantity: int
    quoted_price: Decimal          # Reference price
    vwap_price: Decimal           # Actual execution price
    total_cost: Decimal
    slippage_pct: Decimal         # Percentage slippage
    slippage_absolute: Decimal
    price_impact_pct: Decimal
    fills: List[Tuple[Decimal, int]]  # (price, qty) pairs
    depth_used: int               # Order book levels consumed
    liquidity_sufficient: bool
    
    @property
    def execution_quality(self) -> str:
        """EXCELLENT|GOOD|FAIR|POOR|INSUFFICIENT_LIQUIDITY"""
```

#### LiquidityMetrics
```python
@dataclass
class LiquidityMetrics:
    market_id: str
    bid_depth: int                # Total contracts on bid
    ask_depth: int                # Total contracts on ask
    bid_value: Decimal            # Total USDC value
    ask_value: Decimal
    spread_bps: int               # Spread in basis points
    top_of_book_size: int         # Size at best price
    depth_imbalance: Decimal      # bid / (bid + ask)
    
    @property
    def is_healthy(self) -> bool:
        """Healthy if depth > 100, spread < 5%"""
```

### Usage Examples

#### Basic VWAP Calculation

```python
from src.data.vwap import VWAPCalculator
from decimal import Decimal

calc = VWAPCalculator(slippage_warning_threshold=Decimal("2.0"))

# Order book asks (for buying)
asks = [
    (Decimal("0.52"), 200),  # 200 @ $0.52
    (Decimal("0.53"), 300),  # 300 @ $0.53
    (Decimal("0.55"), 500),  # 500 @ $0.55
]

# Calculate VWAP for 400 contract order
result = calc.calculate_vwap(
    orders=asks,
    quantity=400,
    side="buy",
    market_id="MARKET_123",
    quoted_price=Decimal("0.52")
)

print(f"Quoted: ${result.quoted_price:.4f}")
print(f"VWAP: ${result.vwap_price:.4f}")
print(f"Slippage: {result.slippage_pct:.2f}%")
print(f"Quality: {result.execution_quality}")
```

#### Strategy Integration

```python
from src.execution.vwap_integration import VWAPTradingGate, StrategyVWAPIntegration
from src.platforms.base import OrderSide
from decimal import Decimal

# Create VWAP gate
gate = VWAPTradingGate(
    max_slippage_pct=Decimal("2.0"),
    min_liquidity_contracts=500,
    max_spread_bps=300
)

# Validate signal
is_valid, adjusted_qty, reason = gate.validate_signal(
    market_id="MARKET_123",
    side=OrderSide.YES,
    quantity=1000,
    orderbook=orderbook,
    quoted_price=Decimal("0.52")
)

if is_valid:
    print(f"Signal approved: {adjusted_qty} contracts")
else:
    print(f"Signal blocked: {reason}")
```

#### Risk Management Integration

```python
from src.risk.vwap_checks import VWAPRiskManager, VWAPRiskConfig
from decimal import Decimal

# Create VWAP-enhanced risk manager
risk_mgr = VWAPRiskManager(
    vwap_config=VWAPRiskConfig(
        max_slippage_pct=Decimal("2.0"),
        min_liquidity_contracts=500,
        reject_on_high_slippage=True,
        enable_auto_adjustment=True
    )
)

# Comprehensive trade check
is_allowed, adjusted_qty, reason = risk_mgr.check_trade_with_vwap(
    market_id="MARKET_123",
    side=OrderSide.YES,
    quantity=500,
    orderbook=orderbook
)

if not is_allowed:
    print(f"Trade blocked by VWAP risk check: {reason}")
```

### Configuration

#### VWAP Risk Configuration

```python
@dataclass
class VWAPRiskConfig:
    # Slippage limits
    max_slippage_pct: Decimal = Decimal("2.0")
    slippage_warning_threshold: Decimal = Decimal("1.0")
    
    # Liquidity requirements
    min_liquidity_contracts: int = 500
    min_top_of_book_size: int = 50
    
    # Spread limits
    max_spread_bps: int = 300  # 3%
    
    # Price impact
    max_price_impact_pct: Decimal = Decimal("3.0")
    
    # Trade behavior
    reject_on_insufficient_liquidity: bool = True
    reject_on_high_slippage: bool = True
    reject_on_wide_spread: bool = True
    enable_auto_adjustment: bool = True
```

### Monitoring Dashboard

The VWAP monitoring dashboard provides real-time visibility:

```bash
python examples/vwap_monitor_dashboard.py
```

**Features:**
- Live order book depth tracking
- Spread monitoring
- Slippage alerts
- Liquidity health indicators
- Execution quality statistics
- Price impact curves

**Example Output:**
```
MARKET_ABC:
  Liquidity:
    Bid depth: 1,200 contracts ($600.00)
    Ask depth: 1,500 contracts ($780.00)
    Spread: 250 bps
    Top of book: 300 contracts
    Depth imbalance: 44.4%
    Health: ✓ HEALTHY
  
  VWAP Analysis (BUY):
     100 contracts: VWAP=$0.5200, slippage=0.00%, quality=EXCELLENT
     500 contracts: VWAP=$0.5240, slippage=0.77%, quality=GOOD
    1000 contracts: VWAP=$0.5310, slippage=2.12%, quality=FAIR
```

### Historical Analysis

Use Alchemy Polygon RPC to analyze past trade patterns:

```python
from src.data.vwap import HistoricalVWAPAnalyzer

analyzer = HistoricalVWAPAnalyzer(alchemy_api_key="YOUR_KEY")

# Detect low-liquidity markets
low_liq_markets = await analyzer.detect_low_liquidity_traps(
    market_ids=["MARKET_1", "MARKET_2"],
    threshold_volume=Decimal("1000")
)

# Build price impact model
curve = await analyzer.analyze_price_impact(
    market_id="MARKET_1",
    lookback_days=7
)
```

### Testing

Run comprehensive VWAP tests:

```bash
pytest tests/test_vwap.py -v
```

**Test Coverage:**
- ✓ Basic VWAP calculation
- ✓ Insufficient liquidity handling
- ✓ Slippage calculation accuracy
- ✓ Sell-side VWAP
- ✓ Liquidity metrics
- ✓ Price impact curves
- ✓ Validation logic
- ✓ Order split suggestions
- ✓ Monitoring and statistics

### Performance Considerations

1. **Order Book Frequency:** Fetch fresh order books before each trade
2. **Calculation Speed:** VWAP calculation is O(n) where n = depth levels
3. **Caching:** Cache price impact curves but refresh periodically
4. **Monitoring Overhead:** Dashboard refresh every 30s is optimal

### Common Patterns

#### Pattern 1: Pre-Trade Validation
```python
# Always validate VWAP before executing
orderbook = await platform.get_orderbook(market_id)
result = calc.calculate_vwap(orderbook.asks, quantity, "buy", market_id)

if result.slippage_pct > Decimal("2.0"):
    logger.warning("High slippage, reducing position size")
    quantity = quantity // 2  # Halve size
```

#### Pattern 2: Adaptive Position Sizing
```python
# Size position based on available liquidity
optimal_size = gate._find_optimal_size(
    orders=orderbook.asks,
    side="buy",
    market_id=market_id
)

actual_quantity = min(target_quantity, optimal_size)
```

#### Pattern 3: Order Splitting
```python
# Split large orders to minimize impact
if quantity > 500:
    chunks = validator.suggest_order_split(vwap_result, max_chunks=3)
    for chunk in chunks:
        await execute_order(chunk)
        await asyncio.sleep(5)  # Wait between chunks
```

### Troubleshooting

**Issue:** "Insufficient liquidity" errors
- **Solution:** Reduce position size or wait for liquidity to improve

**Issue:** High slippage warnings
- **Solution:** Use limit orders instead of market orders, or split order

**Issue:** Wide spreads blocking trades
- **Solution:** Increase `max_spread_bps` config or avoid low-volume markets

**Issue:** VWAP deviates significantly from quoted price
- **Solution:** This is the system working correctly! It's preventing bad executions

### Integration Checklist

When adding VWAP to a new strategy:

- [ ] Create `VWAPTradingGate` instance
- [ ] Inherit from `StrategyVWAPIntegration` mixin
- [ ] Call `enrich_signal_with_vwap()` for each signal
- [ ] Check `enriched_signal.is_profitable_after_slippage`
- [ ] Use `adjust_position_size_for_liquidity()` for sizing
- [ ] Consider `split_large_order()` for large positions
- [ ] Monitor execution quality via dashboard

### Future Enhancements

**Planned:**
- [ ] Machine learning models for price impact prediction
- [ ] Cross-market VWAP analysis (Polymarket vs Kalshi)
- [ ] Adaptive slippage thresholds based on market conditions
- [ ] Smart order routing (SOR) to minimize impact
- [ ] TWAP (Time-Weighted Average Price) execution
- [ ] Integration with MEV protection strategies

### References

- `src/data/vwap.py` - Core VWAP calculation
- `src/execution/vwap_integration.py` - Strategy integration
- `src/risk/vwap_checks.py` - Risk management
- `examples/vwap_monitor_dashboard.py` - Monitoring dashboard
- `examples/vwap_strategy_integration.py` - Strategy examples
- `tests/test_vwap.py` - Comprehensive tests

### Related Documentation

- [Risk Management](./RISK_MANAGEMENT.md)
- [Execution Engine](./EXECUTION.md)
- [Platform Integrations](./PLATFORMS.md)
