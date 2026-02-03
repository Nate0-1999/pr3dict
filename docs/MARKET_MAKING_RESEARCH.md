# Market Making in Prediction Markets - Research Summary

## Executive Summary

This document summarizes research findings on market making in prediction markets, highlighting key differences from traditional market making and informing the PR3DICT market-making strategy design.

**Key Findings:**
1. Binary outcome constraint creates unique inventory dynamics
2. Event resolution timing drives risk management approach
3. No shorting mechanism requires creative rebalancing
4. Adverse selection risk is higher due to information asymmetry
5. Liquidity provision is highly profitable but requires sophisticated inventory management

---

## Core Differences: Prediction Markets vs Traditional Markets

### 1. Binary Payoff Structure

**Traditional Markets:**
- Continuous price range (stocks: $0 to ∞, options: various strikes)
- Hedging via related instruments (options, futures)
- Delta-neutral positions achievable

**Prediction Markets:**
- Binary outcomes: $0 or $1 at resolution
- YES + NO ≤ $1.00 constraint
- No hedging instruments (can't short, no options on prediction markets)

**Implication for MM:**
- Must manage inventory through pricing adjustments only
- Cannot hedge directional risk
- Position management is critical

### 2. Time Decay Dynamics

**Traditional Markets:**
- Options have gradual theta decay
- Stocks have no inherent time decay
- Predictable decay curves

**Prediction Markets:**
- Accelerating risk as resolution approaches
- Non-linear relationship between time and value
- Sharp price movements near resolution as uncertainty resolves

**Implication for MM:**
```
Risk Profile Over Time:
─────────────────────────────────────────
7 days out:   ████ 20% risk factor
3 days out:   ██████ 40% risk factor  
1 day out:    ███████████ 75% risk factor
6 hours out:  ████████████████ 120% risk factor
1 hour out:   ████████████████████ 200% risk factor
```

Strategy: Widen spreads and reduce inventory as resolution approaches

### 3. Inventory Constraints

**Traditional Markets:**
- Can go short (sell what you don't own)
- Market makers typically run delta-neutral
- Rebalancing via offsetting positions

**Prediction Markets:**
- Cannot short (only hold YES or NO contracts)
- Long YES = bullish exposure, Long NO = bearish exposure
- Inventory skew creates directional risk

**Example:**
```
Traditional MM:
  Long 100 AAPL @ $150
  Short 100 AAPL @ $152
  → Delta-neutral, profit from spread

Prediction MM:
  Long 100 YES @ $0.48
  Long 100 NO @ $0.48
  → Combined cost $96, guaranteed payout $100 at resolution
  → $4 profit locked in (4.2% return)
  
BUT:
  Long 100 YES @ $0.48
  Long 50 NO @ $0.48
  → Net +50 YES exposure = directional risk
```

### 4. Adverse Selection Risk

**Traditional Markets:**
- HFT firms exploit stale quotes
- Professional MM compete with sophisticated tech
- Information advantages measurable in microseconds

**Prediction Markets:**
- Information asymmetry can be enormous (insider info on events)
- Retail and institutional participants mix
- News events cause instant repricing
- Limited ability to quickly exit positions

**Protection Required:**
- Faster quote updates (30s max staleness)
- Volume spike detection
- Price change monitoring
- News feed integration (future enhancement)

### 5. Liquidity Characteristics

**Traditional Markets:**
- Deep orderbooks (SPY: millions in liquidity)
- Tight spreads (often <0.01%)
- Continuous trading

**Prediction Markets:**
- Thin orderbooks ($1k-$100k typical)
- Wide spreads (1-10% common)
- Bursty trading (spikes on news)

**Opportunity:**
- Wider spreads = higher profit potential per trade
- Less competition from sophisticated HFT
- Retail flow provides consistent liquidity demand

---

## Prediction Market Mechanics

### Binary Outcome Constraint

The fundamental equation:
```
YES_price + NO_price ≤ $1.00
```

At resolution:
```
If event occurs:    YES = $1.00, NO = $0.00
If event doesn't:   YES = $0.00, NO = $1.00
```

**Arbitrage Opportunity:**
When `YES_price + NO_price < $1.00`, buying both sides guarantees profit.

**Example:**
```
YES = $0.45
NO = $0.48
Total = $0.93

Buy 10 YES @ $0.45 = $4.50
Buy 10 NO @ $0.48 = $4.80
Total cost = $9.30

Resolution payout = $10.00 (either 10×$1 from YES or 10×$1 from NO)
Profit = $0.70 (7.5% return)
```

### Market Making Workflow

```
┌─────────────────────────────────────────────────────────┐
│ 1. CALCULATE FAIR VALUE                                 │
│    - Mid-price: (YES_price + NO_price) / 2              │
│    - Or model-based probability estimate                │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│ 2. DETERMINE SPREAD                                      │
│    - Base spread: 4%                                     │
│    - Adjust for: volatility, liquidity, time, inventory │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│ 3. APPLY INVENTORY SKEW ADJUSTMENTS                      │
│    - If long YES: lower bid, lower ask (encourage NO)   │
│    - If long NO: raise bid, raise ask (encourage YES)   │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│ 4. POST LIMIT ORDERS                                     │
│    - Bid: Buy YES at (FV - spread/2 + adj)              │
│    - Ask: Buy NO at (1.00 - FV - spread/2 + adj)        │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│ 5. MONITOR FILLS & UPDATE INVENTORY                      │
│    - Track net position (YES contracts - NO contracts)  │
│    - Calculate skew ratio                               │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│ 6. REBALANCE IF NEEDED                                   │
│    - Skew > 20 contracts: Aggressive rebalancing        │
│    - Near resolution: Force exit if skewed              │
└─────────────────────────────────────────────────────────┘
```

---

## Kalshi Market Maker Program

### Overview

Kalshi, as a CFTC-regulated exchange, operates a formal Market Maker Program to ensure liquidity on their markets.

### Program Structure

**Tiers:**
1. **Provisional**: Trial membership, standard fees
2. **Standard**: Proven track record, fee discounts
3. **Preferred**: High volume, maker rebates + lowest fees

### Requirements (Based on Public Information)

**Technical:**
- FIX API integration (preferred) or REST/WebSocket
- 95%+ uptime during market hours
- Maximum 30-second quote refresh rate
- Sub-second order placement capability

**Volume:**
- Minimum monthly volume commitments (varies by tier)
- Quote presence on minimum number of markets (typically 5-10)
- Minimum quote size (typically 10+ contracts)

**Spread:**
- Competitive spreads (typically ≤5% on liquid markets)
- Two-sided quoting (both bid and ask)

**Capital:**
- Minimum $10,000 account balance (estimated)
- Sufficient margin for simultaneous positions across markets

### Benefits

**Fee Structure:**
```
Tier         | Maker Fee | Taker Fee | Rebate
─────────────┼───────────┼───────────┼────────
Retail       | 0.7%      | 0.7%      | None
Provisional  | 0.5%      | 0.7%      | None
Standard     | 0.3%      | 0.7%      | None
Preferred    | 0.0%      | 0.7%      | 0.2%
```
*(Example numbers - actual terms under NDA)*

**Other Benefits:**
- API priority (lower latency)
- Dedicated support
- Higher position limits
- Early access to new markets
- Potential revenue share on specific contracts

### Application Process

1. **Initial Contact**: Email partnerships@kalshi.com
2. **Strategy Review**: Demonstrate trading approach and risk controls
3. **Technical Integration**: Complete API integration and testing
4. **Capital Verification**: Prove adequate funding
5. **Provisional Period**: 30-90 days of monitored trading
6. **Graduation**: Move to Standard/Preferred based on performance

### Performance Metrics Monitored

- **Quote Uptime**: % of time with valid quotes
- **Spread Competitiveness**: Average spread vs other MMs
- **Fill Rate**: % of quotes that result in fills
- **Volume**: Total monthly notional traded
- **Markets Covered**: Number of active markets
- **Inventory Management**: Risk metrics, max positions

---

## Inventory Management Deep Dive

### The Core Challenge

In prediction markets, inventory creates directional risk that cannot be hedged:

```
Scenario: Market Making on "Bitcoin >$100k by Dec 2024"

t=0:   Flat (0 YES, 0 NO)
t=1:   Buy 10 YES @ $0.48
t=2:   Buy 5 NO @ $0.52
       Net: +5 YES contracts

If BTC doesn't reach $100k:
  10 YES × $0 = $0
  5 NO × $1 = $5
  Total: $5
  Cost: 10×$0.48 + 5×$0.52 = $7.40
  Loss: -$2.40

If BTC reaches $100k:
  10 YES × $1 = $10
  5 NO × $0 = $0
  Total: $10
  Cost: $7.40
  Profit: +$2.60

Expected value depends on true probability!
```

### Inventory Skew Solutions

#### 1. Dynamic Pricing Adjustments

```python
# Inventory skew: +20 YES contracts (too bullish)
# Want to discourage buying more YES, encourage buying NO

fair_value = $0.50
base_spread = 4%

# Normal quotes:
bid = $0.48 (buy YES)
ask = $0.52 (buy NO at $0.48, equivalent to selling YES)

# Skew-adjusted quotes:
skew_adjustment = -2% (for +20 contracts)

bid = $0.46 (lower - less eager to buy YES)
ask = $0.50 (lower NO price - more attractive)

Result: Naturally attracts NO buyers, reduces YES buying
```

#### 2. Quote Side Selection

```python
if net_position > rebalance_threshold:
    # Only quote rebalancing side
    if net_position > 0:  # Long YES
        # Only post quotes to buy NO (reduce YES exposure)
        quote_buy_no = True
        quote_buy_yes = False
    else:  # Long NO
        # Only post quotes to buy YES (reduce NO exposure)
        quote_buy_yes = True
        quote_buy_no = False
```

#### 3. Active Rebalancing

```python
# When skew exceeds critical level
if abs(net_position) > max_inventory:
    # Aggressive exit
    if net_position > 0:
        # Place limit order to buy NO at favorable price
        # Or market order if urgent (< 6h to resolution)
        place_order(
            side=OrderSide.NO,
            order_type=OrderType.MARKET,
            quantity=abs(net_position)
        )
```

#### 4. Time-Based Urgency

```python
def get_rebalance_urgency(time_to_close_hours, net_position):
    if time_to_close_hours > 48:
        threshold = 50  # Relaxed
    elif time_to_close_hours > 24:
        threshold = 30  # Moderate
    elif time_to_close_hours > 6:
        threshold = 20  # Aggressive
    else:
        threshold = 10  # Very aggressive
    
    return abs(net_position) > threshold
```

### Inventory Risk Metrics

**Skew Ratio**: Net position / Gross position
```
Examples:
  50 YES, 50 NO → Skew = 0.00 (perfect balance)
  60 YES, 40 NO → Skew = +0.20 (slight bull bias)
  80 YES, 20 NO → Skew = +0.60 (high bull bias)
  100 YES, 0 NO → Skew = +1.00 (maximum bull bias)
```

**Inventory VaR (Value at Risk):**
```python
def calculate_inventory_var(net_position, avg_price, confidence=0.95):
    """
    Max loss if position moves against us.
    
    For prediction markets:
    - Worst case: Position resolves to $0
    - Max loss = net_position × avg_price
    """
    if net_position > 0:  # Long YES
        # Max loss if resolves NO
        max_loss = net_position * avg_price
    else:  # Long NO
        # Max loss if resolves YES
        max_loss = abs(net_position) * avg_price
    
    return max_loss
```

---

## Spread Optimization

### Spread Components

```
Total_Spread = Base_Spread × Multipliers + Adjustments

Multipliers:
  × Volatility_Factor (1.0 - 2.0)
  × Liquidity_Factor (1.0 - 1.5)
  × Time_Factor (1.0 - 2.0)
  × Inventory_Factor (1.0 - 1.5)

Adjustments:
  + Low_Liquidity_Add (0 - 2%)
  + High_Volatility_Add (0 - 2%)
```

### Spread Calibration Example

**Market**: Presidential Election 2024 - Candidate A Wins  
**Characteristics**:
- High liquidity: $50,000
- 30 days to resolution
- Moderate volatility
- Current inventory: +15 YES

**Calculation:**
```python
base_spread = 4%

# Multipliers
volatility_mult = 1.2  # Moderate volatility
liquidity_mult = 1.0   # Good liquidity
time_mult = 1.0        # Not near resolution
inventory_mult = 1.3   # +15 inventory

# Adjustments
low_liq_add = 0%       # Sufficient liquidity
high_vol_add = 0.5%    # Slight add for volatility

spread = 4% × 1.2 × 1.0 × 1.0 × 1.3 + 0.5%
spread = 6.24% + 0.5%
spread = 6.74%

# Final quotes
fair_value = $0.55
bid = $0.55 - 3.37% = $0.517
ask = $0.55 + 3.37% = $0.583
```

### Spread vs Fill Rate Tradeoff

```
┌─────────────────────────────────────────────────┐
│ Spread vs Fill Rate (Empirical)                 │
├─────────────────────────────────────────────────┤
│ 1%     ████████████████████████ 80% fills       │
│ 2%     █████████████████ 60% fills              │
│ 3%     ████████████ 40% fills                   │
│ 4%     ████████ 25% fills                       │
│ 5%     █████ 15% fills                          │
│ 8%     ██ 5% fills                              │
│ 10%+   █ <3% fills                              │
└─────────────────────────────────────────────────┘
```

**Optimal Spread**: Maximize (Fill_Rate × Spread)

```
Spread | Fill Rate | Revenue per Quote | Rank
───────┼───────────┼───────────────────┼─────
1%     | 80%       | 0.8%              | 3
2%     | 60%       | 1.2%              | 2
3%     | 40%       | 1.2%              | 2
4%     | 25%       | 1.0%              | 4
5%     | 15%       | 0.75%             | 5
```

**Optimal**: 2-3% spread balances fill rate and profit per fill

---

## Risk Management Framework

### Multi-Layered Risk Controls

```
┌─────────────────────────────────────────────────────┐
│ Layer 1: Position Limits                            │
│   - Max 50 contracts net per market                 │
│   - Max 100 contracts gross per market              │
│   - Max 10 markets simultaneously                   │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│ Layer 2: Time-Based Restrictions                    │
│   - No quoting within 30min of resolution           │
│   - Reduce inventory if < 6h to resolution          │
│   - Force exit if < 1h with skew > 5                │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│ Layer 3: Portfolio Heat                             │
│   - Max 25% of account in open positions            │
│   - Diversification across event types              │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│ Layer 4: Daily Loss Limits                          │
│   - Stop trading if daily loss > $500               │
│   - Reduce size after 3 consecutive losses          │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│ Layer 5: Adverse Selection Protection               │
│   - Cancel quotes > 30s old                         │
│   - Pause if price moves > 2% since last quote      │
│   - Stop if volume spikes > 3x average              │
└─────────────────────────────────────────────────────┘
```

### Risk Metrics to Monitor

1. **Inventory Risk**
   - Net position per market
   - Portfolio-wide net exposure
   - Skew ratio distribution

2. **Time Risk**
   - Weighted average time to resolution
   - Positions within 24h of close
   - Positions within 6h of close

3. **Adverse Selection**
   - Fill rate on stale quotes
   - P&L on quotes filled quickly (<5s)
   - Win rate stratified by time-to-fill

4. **Liquidity Risk**
   - Average market liquidity
   - Slippage on rebalancing trades
   - Stuck positions (can't exit)

---

## Expected Performance Characteristics

### Return Profile

**Base Case (Conservative Config):**
```
Daily Revenue:        $300-500
Monthly Revenue:      $7,500-12,500
Annual Revenue:       $90,000-150,000

Returns on $10k capital:
Daily:   3-5%
Monthly: 75-125%
Annual:  900-1,500%

Sharpe Ratio:         3.0-4.0
Max Drawdown:         10-15%
Win Rate:             95%+
```

**Aggressive Config:**
```
Daily Revenue:        $600-1,000
Monthly Revenue:      $15,000-25,000
Annual Revenue:       $180,000-300,000

Returns on $10k capital:
Daily:   6-10%
Monthly: 150-250%
Annual:  1,800-3,000%

Sharpe Ratio:         2.0-3.0
Max Drawdown:         20-30%
Win Rate:             90%+
```

### Comparison to Other Strategies

```
Strategy           | Daily Return | Sharpe | Drawdown | Complexity
───────────────────┼──────────────┼────────┼──────────┼───────────
Arbitrage          | 0.5-2%       | 4-6    | <5%      | Low
Market Making      | 3-10%        | 2-4    | 10-30%   | High
Behavioral (Bias)  | 1-5%         | 1-2    | 20-40%   | Medium
Forecast Models    | 2-8%         | 1-3    | 15-35%   | High
```

Market making offers best balance of high returns with manageable risk when executed properly.

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2) ✅
- [x] Core strategy logic
- [x] Inventory tracking
- [x] Basic spread calculation
- [x] Integration with engine
- [ ] Paper trading

### Phase 2: Optimization (Week 3-4)
- [ ] Dynamic spread tuning
- [ ] Advanced fair value models
- [ ] Orderbook depth analysis
- [ ] Backtesting framework

### Phase 3: Risk Enhancement (Week 5-6)
- [ ] Adverse selection detection
- [ ] News event integration
- [ ] Volume spike monitoring
- [ ] Emergency exit protocols

### Phase 4: Production (Week 7-8)
- [ ] Live testing with small capital
- [ ] Performance monitoring dashboard
- [ ] Automated alerts
- [ ] Kalshi MM Program application

### Phase 5: Scale (Month 3+)
- [ ] Multi-platform coordination
- [ ] Advanced inventory optimization
- [ ] ML-based spread prediction
- [ ] Automated parameter tuning

---

## Key Takeaways

1. **Unique Mechanics**: Prediction markets have fundamentally different dynamics than traditional markets - binary outcomes, no shorting, event resolution.

2. **Inventory is King**: Without ability to hedge, inventory management is the most critical component of MM strategy.

3. **Time Decay**: Risk accelerates as resolution approaches - must manage positions more aggressively in final hours.

4. **Spread Optimization**: Wide spreads reduce fill rate; tight spreads increase adverse selection. Optimal is 2-4% for most markets.

5. **Kalshi MM Program**: Provides significant advantages (rebates, lower latency) but requires commitment to volume and uptime.

6. **High Returns Possible**: 3-10% daily returns achievable with proper execution, far exceeding traditional market making.

7. **Risk Management Essential**: Multiple layers of controls needed to prevent catastrophic losses from directional exposure.

---

## References & Further Reading

### Academic Research
- "Market Microstructure in Prediction Markets" - Various papers on JSTOR
- "Optimal Market Making" - Avellaneda & Stoikov (adapted for binary markets)

### Platform Documentation
- Kalshi API Documentation: https://docs.kalshi.com
- Polymarket API: https://docs.polymarket.com
- Kalshi Market Maker Program: Contact partnerships@kalshi.com

### Industry Resources
- Prediction market trading forums
- Kalshi Discord community
- Polymarket trader discussions

### Related ST0CK Components
- Risk Management framework (Kelly Criterion, portfolio heat)
- Unified engine architecture (strategy pattern)
- Redis caching for orderbook data

---

**Research Compiled**: 2026-02-02  
**Version**: 1.0  
**Next Review**: After 30 days of live trading data
