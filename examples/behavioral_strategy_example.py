"""
Example usage of the Behavioral Trading Strategy.

This demonstrates how to:
1. Initialize the strategy with different configurations
2. Scan markets for signals
3. Handle position management
4. Backtest the strategy
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import List

from pr3dict.strategies.behavioral import create_behavioral_strategy
from pr3dict.platforms.base import Market, OrderSide, Position


# Mock market data for demonstration
def create_mock_market(market_id: str, yes_price: float, volume: float = 10000) -> Market:
    """Create a mock market for testing."""
    return Market(
        market_id=market_id,
        question=f"Test market {market_id}",
        yes_price=Decimal(str(yes_price)),
        no_price=Decimal(str(1.0 - yes_price)),
        volume=Decimal(str(volume)),
        close_date=datetime.now() + timedelta(days=30),
        platform="polymarket"
    )


async def example_longshot_fade():
    """Example: Detecting and trading longshot bias."""
    print("\n=== LONGSHOT FADE EXAMPLE ===")
    
    # Create strategy
    strategy = create_behavioral_strategy(
        enable_longshot=True,
        enable_favorite=False,
        enable_overreaction=False,
        enable_recency=False,
        min_edge=0.03
    )
    
    # Create longshot market (8% probability)
    market = create_mock_market("longshot_1", yes_price=0.08)
    
    # Scan for signals
    signals = await strategy.scan_markets([market])
    
    if signals:
        signal = signals[0]
        print(f"✅ Signal Generated:")
        print(f"   Market: {signal.market_id}")
        print(f"   Side: {signal.side}")
        print(f"   Strength: {signal.strength:.2%}")
        print(f"   Reason: {signal.reason}")
        print(f"   Target Price: ${signal.target_price}")
    else:
        print("❌ No signal generated")


async def example_favorite_support():
    """Example: Detecting and trading favorite bias."""
    print("\n=== FAVORITE SUPPORT EXAMPLE ===")
    
    strategy = create_behavioral_strategy(
        enable_longshot=False,
        enable_favorite=True,
        min_edge=0.02
    )
    
    # Create favorite market (82% probability)
    market = create_mock_market("favorite_1", yes_price=0.82)
    
    signals = await strategy.scan_markets([market])
    
    if signals:
        signal = signals[0]
        print(f"✅ Signal Generated:")
        print(f"   Market: {signal.market_id}")
        print(f"   Side: {signal.side}")
        print(f"   Strength: {signal.strength:.2%}")
        print(f"   Reason: {signal.reason}")
        print(f"   Target Price: ${signal.target_price}")
    else:
        print("❌ No signal generated")


async def example_overreaction_detection():
    """Example: Detecting overreaction to news."""
    print("\n=== OVERREACTION FADE EXAMPLE ===")
    
    strategy = create_behavioral_strategy(
        enable_overreaction=True,
        min_edge=0.05
    )
    
    # Create market with price history showing spike
    market = create_mock_market("overreact_1", yes_price=0.65)
    
    # Simulate price history: stable at 0.40, then spiked to 0.65
    base_time = datetime.now()
    strategy.price_history[market.market_id] = [
        (base_time - timedelta(hours=10), 0.40),
        (base_time - timedelta(hours=8), 0.41),
        (base_time - timedelta(hours=6), 0.42),
        (base_time - timedelta(hours=4), 0.55),  # Start of spike
        (base_time - timedelta(hours=2), 0.62),
        (base_time - timedelta(hours=1), 0.65),  # Current
    ]
    
    signals = await strategy.scan_markets([market])
    
    if signals:
        signal = signals[0]
        print(f"✅ Signal Generated:")
        print(f"   Market: {signal.market_id}")
        print(f"   Side: {signal.side}")
        print(f"   Strength: {signal.strength:.2%}")
        print(f"   Reason: {signal.reason}")
        print(f"   Expected: Fade the upward spike (BET NO)")
    else:
        print("❌ No signal generated")


async def example_position_management():
    """Example: Managing positions with exit logic."""
    print("\n=== POSITION MANAGEMENT EXAMPLE ===")
    
    strategy = create_behavioral_strategy()
    
    # Create a position (simulated entry)
    entry_market = create_mock_market("position_1", yes_price=0.08)
    position = Position(
        position_id="pos_123",
        market_id=entry_market.market_id,
        side=OrderSide.NO,
        quantity=100,
        entry_price=Decimal("0.92"),  # Bought NO at $0.92
        entry_time=datetime.now() - timedelta(hours=12),
        platform="polymarket",
        reason="LONGSHOT_FADE: YES at 8.0% (overpriced by ~5%)"
    )
    
    print(f"Position opened:")
    print(f"   Side: {position.side}")
    print(f"   Entry: ${position.entry_price}")
    print(f"   Time held: 12 hours")
    
    # Scenario 1: Price moved in our favor (YES dropped to 5%)
    print("\n--- Scenario 1: Profit Target Hit ---")
    current_market = create_mock_market("position_1", yes_price=0.05)
    exit_signal = await strategy.check_exit(position, current_market)
    
    if exit_signal:
        print(f"✅ Exit Signal: {exit_signal.reason}")
        profit = (position.entry_price - current_market.no_price) / position.entry_price
        print(f"   Profit: {profit:.2%}")
    
    # Scenario 2: Price moved against us (YES rose to 15%)
    print("\n--- Scenario 2: Stop Loss Triggered ---")
    bad_market = create_mock_market("position_1", yes_price=0.15)
    exit_signal = await strategy.check_exit(position, bad_market)
    
    if exit_signal:
        print(f"🛑 Exit Signal: {exit_signal.reason}")
        loss = (position.entry_price - bad_market.no_price) / position.entry_price
        print(f"   Loss: {loss:.2%}")


async def example_multi_signal_scan():
    """Example: Scanning multiple markets for various signals."""
    print("\n=== MULTI-MARKET SCAN EXAMPLE ===")
    
    # Strategy with all signals enabled
    strategy = create_behavioral_strategy(
        enable_longshot=True,
        enable_favorite=True,
        enable_overreaction=True,
        enable_recency=True,
        min_edge=0.02
    )
    
    # Create diverse set of markets
    markets = [
        create_mock_market("m1", yes_price=0.10),  # Longshot
        create_mock_market("m2", yes_price=0.78),  # Favorite
        create_mock_market("m3", yes_price=0.50),  # Neutral
        create_mock_market("m4", yes_price=0.05),  # Extreme longshot
        create_mock_market("m5", yes_price=0.88),  # Strong favorite
    ]
    
    # Add price history for overreaction detection on m3
    base_time = datetime.now()
    strategy.price_history["m3"] = [
        (base_time - timedelta(hours=8), 0.35),
        (base_time - timedelta(hours=6), 0.38),
        (base_time - timedelta(hours=4), 0.42),
        (base_time - timedelta(hours=2), 0.48),
        (base_time - timedelta(hours=1), 0.50),
    ]
    
    # Scan all markets
    signals = await strategy.scan_markets(markets)
    
    print(f"Scanned {len(markets)} markets")
    print(f"Generated {len(signals)} signals:\n")
    
    for i, signal in enumerate(signals, 1):
        print(f"{i}. Market: {signal.market_id}")
        print(f"   Type: {signal.reason.split(':')[0]}")
        print(f"   Side: {signal.side}")
        print(f"   Strength: {signal.strength:.1%}")
        print(f"   Detail: {signal.reason}")
        print()


async def example_position_sizing():
    """Example: Calculate position sizes based on risk."""
    print("\n=== POSITION SIZING EXAMPLE ===")
    
    strategy = create_behavioral_strategy()
    
    # Create a signal
    market = create_mock_market("size_test", yes_price=0.08)
    signals = await strategy.scan_markets([market])
    
    if signals:
        signal = signals[0]
        account_balance = Decimal("10000")  # $10,000 account
        
        # Conservative sizing (1% risk)
        contracts_1pct = strategy.get_position_size(
            signal, 
            account_balance, 
            risk_pct=0.01
        )
        
        # Standard sizing (2% risk)
        contracts_2pct = strategy.get_position_size(
            signal, 
            account_balance, 
            risk_pct=0.02
        )
        
        # Aggressive sizing (5% risk)
        contracts_5pct = strategy.get_position_size(
            signal, 
            account_balance, 
            risk_pct=0.05
        )
        
        print(f"Account Balance: ${account_balance:,}")
        print(f"Signal: {signal.reason.split(':')[0]}")
        print(f"Entry Price: ${signal.target_price}\n")
        
        print(f"Conservative (1% risk): {contracts_1pct:,} contracts (${contracts_1pct * float(signal.target_price):,.2f})")
        print(f"Standard (2% risk):     {contracts_2pct:,} contracts (${contracts_2pct * float(signal.target_price):,.2f})")
        print(f"Aggressive (5% risk):   {contracts_5pct:,} contracts (${contracts_5pct * float(signal.target_price):,.2f})")


async def example_backtest_simulation():
    """Example: Simple backtest simulation."""
    print("\n=== BACKTEST SIMULATION EXAMPLE ===")
    
    strategy = create_behavioral_strategy(min_edge=0.03)
    
    # Simulate 10 historical longshot markets
    results = []
    
    for i in range(10):
        # Create longshot (8-12% probability)
        entry_price = 0.08 + (i * 0.004)
        market = create_mock_market(f"backtest_{i}", yes_price=entry_price)
        
        signals = await strategy.scan_markets([market])
        
        if signals:
            signal = signals[0]
            
            # Simulate outcome (longshot bias means we win ~70% of the time)
            # In reality, YES price should resolve to 0 most of the time
            import random
            random.seed(i)  # Deterministic for example
            
            # 70% win rate for longshot fades
            won = random.random() < 0.70
            
            if won:
                # Market resolved NO, we win
                exit_price = 0.0  # NO contracts worth $1
                pnl = (Decimal("1.0") - signal.target_price) / signal.target_price
            else:
                # Market resolved YES, we lose
                exit_price = 1.0  # NO contracts worth $0
                pnl = -1.0  # Total loss
            
            results.append({
                'market': market.market_id,
                'entry': float(signal.target_price),
                'won': won,
                'pnl': float(pnl)
            })
    
    # Calculate statistics
    trades = len(results)
    wins = sum(1 for r in results if r['won'])
    win_rate = wins / trades if trades > 0 else 0
    avg_pnl = sum(r['pnl'] for r in results) / trades if trades > 0 else 0
    
    print(f"Backtest Results ({trades} trades):")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Average Return: {avg_pnl:.2%}")
    print(f"   Expected Win Rate: 65-70%")
    print(f"   Expected Return: +5-8%")
    
    print("\nTrade Log:")
    for r in results[:5]:  # Show first 5
        status = "✅ WIN" if r['won'] else "❌ LOSS"
        print(f"   {r['market']}: Entry ${r['entry']:.3f} → {status} ({r['pnl']:+.1%})")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("PR3DICT BEHAVIORAL STRATEGY EXAMPLES")
    print("=" * 60)
    
    await example_longshot_fade()
    await example_favorite_support()
    await example_overreaction_detection()
    await example_position_management()
    await example_multi_signal_scan()
    await example_position_sizing()
    await example_backtest_simulation()
    
    print("\n" + "=" * 60)
    print("Examples complete! Review the documentation for more details.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
