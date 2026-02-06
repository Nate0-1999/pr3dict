# X Post Analysis: @rohonchain Polymarket Trading Math

**Source:** Twitter/X post warning about Polymarket execution issues  
**Author:** @rohonchain  
**Topic:** Why quoted prices ≠ execution prices  
**Relevance:** Critical for pr3dict VWAP implementation

---

## Context

This document captures the key insights from an X post that warned about a critical issue in prediction market trading: **the quoted mid-price you see is NOT what you'll get when you execute**.

## Core Problem

### The Illusion of Profit

When you look at a prediction market and see:
- **YES:** $0.52
- **NO:** $0.48
- **Sum:** $1.00 (looks fair)

You might think: "If I buy both sides for $1.00 total and one pays $1.00, I break even!"

**But this is WRONG.**

### The Reality: Order Book Depth

What you're seeing is the **mid price** or **best bid/offer**. When you actually try to execute:

```
Quoted YES: $0.52
Actual execution for 1,000 contracts:
  - 200 @ $0.52  (best ask)
  - 300 @ $0.54  (+4% slippage)
  - 500 @ $0.58  (+11% slippage)
  
VWAP (what you actually pay): $0.556
Slippage: 7% vs quoted!
```

### The Math

```
Expected (quoted):  YES $0.52 + NO $0.48 = $1.00
Actual (executed):  YES $0.56 + NO $0.51 = $1.07

Expected profit: $0.00 (break even)
Actual result:   -$0.07 (7% LOSS!)
```

## Section 8: Slippage Deep Dive

### What is Slippage?

**Slippage** = The difference between:
1. The price you **expect** (quoted/mid)
2. The price you **actually get** (VWAP after consuming order book)

### Why It Happens

1. **Order book has DEPTH, not just a single price**
   - Best ask: $0.52 for 200 contracts
   - Next level: $0.54 for 300 contracts
   - Next level: $0.58 for 500 contracts

2. **Large orders consume multiple levels**
   - Small order (10 contracts): Gets best price ✅
   - Large order (1000 contracts): Walks up the book ❌

3. **Low liquidity markets are WORST**
   - Popular market: Tight spread, deep book
   - Niche market: Wide spread, thin book
   - **Trading niche = getting wrecked on fills**

### Real Example: Arbitrage Trap

```python
# You see this opportunity:
market_A_yes = 0.48  # Polymarket
market_B_no = 0.50   # Kalshi
# Sum < 1.00, looks like free money!

# You try to execute:
market_A_vwap = 0.52  # Slippage hit you
market_B_vwap = 0.52  # Slippage here too
# Sum = 1.04, you LOST 4%

# The "arbitrage" was a mirage
```

### Slippage Factors

1. **Order Size**
   - Bigger order = more slippage
   - Logarithmic relationship (doubling size ≠ doubling slippage)

2. **Market Liquidity**
   - High volume markets: 0.1-0.5% slippage
   - Medium volume: 1-3% slippage
   - Low volume: 5-20% slippage (!)

3. **Time of Day**
   - US market hours: Better liquidity
   - Off-hours: Spreads widen

4. **Event Proximity**
   - Far from resolution: Thin
   - Near resolution: Can improve OR crater (depends)

5. **Market Maker Presence**
   - Professional MM: Tight spreads
   - Retail-only: Wide spreads

### The VWAP Solution

**VWAP (Volume-Weighted Average Price)** calculates your ACTUAL execution price:

```python
def calculate_vwap(order_book, quantity):
    total_cost = 0
    remaining = quantity
    
    for price, size in order_book:
        if remaining <= 0:
            break
        fill = min(remaining, size)
        total_cost += price * fill
        remaining -= fill
    
    return total_cost / quantity
```

**Key insight:** Check VWAP BEFORE placing order, not after.

### Protection Strategy

1. **Always check VWAP before trading**
   ```python
   quoted_price = 0.52
   vwap = calculate_vwap(order_book, quantity)
   slippage = (vwap - quoted_price) / quoted_price
   
   if slippage > 0.02:  # 2% threshold
       reject_trade()
   ```

2. **Set slippage limits**
   - Conservative: 1% max
   - Normal: 2% max
   - Aggressive: 5% max

3. **Adjust position size**
   ```python
   # Don't force your size
   target_size = 1000
   max_acceptable_slippage = 0.02
   
   optimal_size = find_size_within_slippage(
       order_book,
       max_acceptable_slippage
   )
   
   actual_size = min(target_size, optimal_size)
   ```

4. **Split large orders**
   ```python
   # Instead of: 1000 all at once
   # Do: 200, 200, 200, 200, 200 over time
   
   for chunk in split_order(1000, chunk_size=200):
       execute(chunk)
       wait(delay=30)  # Let book replenish
   ```

5. **Avoid low-liquidity traps**
   ```python
   if total_book_depth < 500:
       skip_market()  # Not worth it
   ```

## Key Takeaways

1. ⚠️ **Quoted price is a LIE for large orders**
2. ✅ **Always calculate VWAP before executing**
3. 📊 **Check order book DEPTH, not just top price**
4. 💸 **Slippage can turn profits into losses**
5. 🎯 **Adjust size based on liquidity**
6. ⏱️ **Split large orders over time**
7. 🚫 **Avoid thin markets entirely**

## Implementation in pr3dict

The VWAP system implements all these lessons:

- ✅ Real-time VWAP calculation
- ✅ Slippage validation gates
- ✅ Automatic position sizing
- ✅ Order splitting suggestions
- ✅ Liquidity health monitoring
- ✅ Pre-trade execution quality checks

See: `src/data/vwap.py` for full implementation

## Additional Resources

- Polymarket CLOB API: Order book snapshots
- Alchemy Polygon RPC: Historical trade data
- py-clob-client: Polymarket Python SDK

---

**Bottom Line:** If you're not checking VWAP, you're gambling, not trading. The pr3dict VWAP system ensures you only execute when you can actually get the price you need.
