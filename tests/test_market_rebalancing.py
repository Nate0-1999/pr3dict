"""
Tests for Market Rebalancing Strategy

Validates the highest ROI strategy from the $40M Polymarket analysis.
"""
import pytest
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import List

from src.strategies.market_rebalancing import (
    MarketRebalancingStrategy,
    RebalancingConfig,
    RebalancingOpportunity
)
from src.platforms.base import Market, Position, OrderSide


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Standard configuration for testing."""
    return RebalancingConfig(
        min_deviation=Decimal("0.02"),
        min_outcomes=3,
        max_outcomes=20,
        min_liquidity_per_outcome=Decimal("500"),
        min_total_liquidity=Decimal("2000"),
        enable_vwap_check=True,
        vwap_depth_usd=Decimal("1000"),
        max_position_size_usd=Decimal("5000"),
        min_profit_threshold=Decimal("0.05"),
    )


@pytest.fixture
def strategy(config):
    """Market rebalancing strategy instance."""
    return MarketRebalancingStrategy(config=config)


def create_market(
    market_id: str,
    title: str,
    yes_price: Decimal,
    liquidity: Decimal = Decimal("5000"),
    hours_to_close: int = 48
) -> Market:
    """Helper to create test market."""
    close_time = datetime.now(timezone.utc) + timedelta(hours=hours_to_close)
    no_price = Decimal("1.0") - yes_price
    
    return Market(
        id=market_id,
        ticker=market_id.upper(),
        title=title,
        description=f"Description for {title}",
        yes_price=yes_price,
        no_price=no_price,
        volume=liquidity * Decimal("2"),
        liquidity=liquidity,
        close_time=close_time,
        resolved=False,
        platform="polymarket"
    )


# ============================================================================
# Test: Market Grouping
# ============================================================================

def test_market_grouping_basic(strategy):
    """Test basic market grouping functionality."""
    markets = [
        create_market("m1", "2024 Election - Trump wins", Decimal("0.30")),
        create_market("m2", "2024 Election - Biden wins", Decimal("0.35")),
        create_market("m3", "2024 Election - Other wins", Decimal("0.25")),
        create_market("m4", "Weather - Sunny tomorrow", Decimal("0.70")),  # Different group
    ]
    
    strategy._update_market_groups(markets)
    
    # Should create one group for Election markets
    assert len(strategy.market_groups) >= 1
    
    # Election markets should be grouped together
    election_groups = [
        group for group, ids in strategy.market_groups.items()
        if "2024 Election" in group
    ]
    assert len(election_groups) == 1
    
    # Group should have 3 markets
    election_group_id = election_groups[0]
    assert len(strategy.market_groups[election_group_id]) == 3
    
    # All election markets should map to same group
    assert strategy.market_to_group["m1"] == election_group_id
    assert strategy.market_to_group["m2"] == election_group_id
    assert strategy.market_to_group["m3"] == election_group_id


def test_market_grouping_min_outcomes(strategy):
    """Test that groups require minimum outcomes."""
    markets = [
        create_market("m1", "Binary Market - Yes", Decimal("0.50")),
        create_market("m2", "Binary Market - No", Decimal("0.50")),
    ]
    
    strategy._update_market_groups(markets)
    
    # Binary market should not create a group (min_outcomes=3)
    assert len(strategy.market_groups) == 0


def test_market_grouping_resolved_skip(strategy):
    """Test that resolved markets are skipped."""
    markets = [
        create_market("m1", "Election - Trump", Decimal("0.30")),
        create_market("m2", "Election - Biden", Decimal("0.35")),
    ]
    
    # Mark one as resolved
    markets[0].resolved = True
    
    strategy._update_market_groups(markets)
    
    # Should not create group with resolved market
    assert "m1" not in strategy.market_to_group


# ============================================================================
# Test: Opportunity Detection
# ============================================================================

@pytest.mark.asyncio
async def test_opportunity_detection_buy_all(strategy):
    """Test detection of buy-all opportunity (sum < $1.00)."""
    # Create markets with sum < $1.00
    markets = [
        create_market("m1", "Election - Candidate A", Decimal("0.20")),
        create_market("m2", "Election - Candidate B", Decimal("0.25")),
        create_market("m3", "Election - Candidate C", Decimal("0.30")),
    ]
    # Sum = 0.75 (should be 1.00) → Buy all for $0.75, get $1.00 payout
    
    opportunity = await strategy._detect_opportunity(markets)
    
    assert opportunity is not None
    assert opportunity.direction == "buy_all"
    assert opportunity.total_sum == Decimal("0.75")
    assert opportunity.deviation == Decimal("0.25")
    assert abs(opportunity.deviation_pct - 0.25) < 0.01
    assert opportunity.expected_profit > Decimal("0")
    assert len(opportunity.markets) == 3


@pytest.mark.asyncio
async def test_opportunity_detection_sell_all(strategy):
    """Test detection of sell-all opportunity (sum > $1.00)."""
    # Create markets with sum > $1.00
    markets = [
        create_market("m1", "Election - Candidate A", Decimal("0.35")),
        create_market("m2", "Election - Candidate B", Decimal("0.40")),
        create_market("m3", "Election - Candidate C", Decimal("0.35")),
    ]
    # Sum = 1.10 (should be 1.00) → Sell all (buy NO) for ~$0.90, collect $1.00
    
    opportunity = await strategy._detect_opportunity(markets)
    
    assert opportunity is not None
    assert opportunity.direction == "sell_all"
    assert opportunity.total_sum == Decimal("1.10")
    assert opportunity.deviation == Decimal("0.10")
    assert opportunity.expected_profit > Decimal("0")


@pytest.mark.asyncio
async def test_opportunity_detection_no_deviation(strategy):
    """Test that balanced markets don't trigger opportunity."""
    # Create markets with sum = $1.00 (perfectly balanced)
    markets = [
        create_market("m1", "Election - Candidate A", Decimal("0.33")),
        create_market("m2", "Election - Candidate B", Decimal("0.33")),
        create_market("m3", "Election - Candidate C", Decimal("0.34")),
    ]
    # Sum = 1.00 (no arbitrage)
    
    opportunity = await strategy._detect_opportunity(markets)
    
    # Should not detect opportunity (deviation < threshold)
    assert opportunity is None


@pytest.mark.asyncio
async def test_opportunity_detection_low_liquidity(strategy):
    """Test that low liquidity markets are rejected."""
    markets = [
        create_market("m1", "Election - A", Decimal("0.20"), liquidity=Decimal("100")),
        create_market("m2", "Election - B", Decimal("0.25"), liquidity=Decimal("100")),
        create_market("m3", "Election - C", Decimal("0.30"), liquidity=Decimal("100")),
    ]
    # Sum = 0.75, but liquidity too low
    
    opportunity = await strategy._detect_opportunity(markets)
    
    # Should reject due to low liquidity
    assert opportunity is None


@pytest.mark.asyncio
async def test_opportunity_detection_liquidity_imbalance(strategy):
    """Test that liquidity imbalance is detected."""
    markets = [
        create_market("m1", "Election - A", Decimal("0.20"), liquidity=Decimal("10000")),
        create_market("m2", "Election - B", Decimal("0.25"), liquidity=Decimal("5000")),
        create_market("m3", "Election - C", Decimal("0.30"), liquidity=Decimal("500")),  # Bottleneck
    ]
    # Sum = 0.75, but m3 has only $500 liquidity (ratio = 500/10000 = 0.05 < 0.3)
    
    opportunity = await strategy._detect_opportunity(markets)
    
    # Should reject due to liquidity imbalance
    assert opportunity is None


# ============================================================================
# Test: VWAP Calculation
# ============================================================================

@pytest.mark.asyncio
async def test_vwap_calculation_buy(strategy):
    """Test VWAP calculation for buy orders."""
    market = create_market("m1", "Test Market", Decimal("0.50"))
    
    vwap = await strategy._calculate_vwap(market, "buy_all")
    
    # VWAP should be slightly higher than quoted (slippage)
    assert vwap >= market.yes_price
    # But not too much higher (sanity check)
    assert vwap < market.yes_price * Decimal("1.02")  # <2% slippage


@pytest.mark.asyncio
async def test_vwap_calculation_sell(strategy):
    """Test VWAP calculation for sell orders."""
    market = create_market("m1", "Test Market", Decimal("0.50"))
    
    vwap = await strategy._calculate_vwap(market, "sell_all")
    
    # VWAP should be close to NO price with slippage
    expected_no_price = Decimal("1.0") - market.yes_price
    assert vwap >= expected_no_price
    assert vwap < expected_no_price * Decimal("1.02")


@pytest.mark.asyncio
async def test_vwap_caching(strategy):
    """Test that VWAP results are cached."""
    market = create_market("m1", "Test Market", Decimal("0.50"))
    
    # Calculate VWAP twice
    vwap1 = await strategy._calculate_vwap(market, "buy_all")
    vwap2 = await strategy._calculate_vwap(market, "buy_all")
    
    # Should return same value (from cache)
    assert vwap1 == vwap2
    
    # Cache should have entry
    assert "m1" in strategy._vwap_cache


# ============================================================================
# Test: Bregman Allocation
# ============================================================================

def test_bregman_allocation_basic(strategy):
    """Test basic Bregman allocation calculation."""
    markets = [
        create_market("m1", "Election - A", Decimal("0.20"), liquidity=Decimal("5000")),
        create_market("m2", "Election - B", Decimal("0.25"), liquidity=Decimal("7000")),
        create_market("m3", "Election - C", Decimal("0.30"), liquidity=Decimal("6000")),
    ]
    
    vwap_prices = {
        "m1": Decimal("0.21"),
        "m2": Decimal("0.26"),
        "m3": Decimal("0.31"),
    }
    
    allocation = strategy._calculate_bregman_allocation(markets, "buy_all", vwap_prices)
    
    # Should allocate to all markets
    assert len(allocation) == 3
    assert all(size > 0 for size in allocation.values())
    
    # Total allocation should equal max position size
    total = sum(allocation.values())
    assert abs(total - strategy.config.max_position_size_usd) < Decimal("0.01")


def test_bregman_allocation_liquidity_weighting(strategy):
    """Test that allocation favors more liquid markets."""
    markets = [
        create_market("m1", "Election - A", Decimal("0.30"), liquidity=Decimal("10000")),  # Most liquid
        create_market("m2", "Election - B", Decimal("0.30"), liquidity=Decimal("2000")),   # Less liquid
        create_market("m3", "Election - C", Decimal("0.30"), liquidity=Decimal("5000")),
    ]
    
    vwap_prices = {
        "m1": Decimal("0.31"),
        "m2": Decimal("0.31"),
        "m3": Decimal("0.31"),
    }
    
    allocation = strategy._calculate_bregman_allocation(markets, "buy_all", vwap_prices)
    
    # m1 (most liquid) should get largest allocation
    assert allocation["m1"] > allocation["m2"]
    assert allocation["m1"] > allocation["m3"]


# ============================================================================
# Test: Signal Generation
# ============================================================================

def test_signal_generation_buy_all(strategy):
    """Test signal generation for buy-all opportunity."""
    markets = [
        create_market("m1", "Election - A", Decimal("0.20")),
        create_market("m2", "Election - B", Decimal("0.25")),
        create_market("m3", "Election - C", Decimal("0.30")),
    ]
    
    opportunity = RebalancingOpportunity(
        market_ids=["m1", "m2", "m3"],
        markets=markets,
        direction="buy_all",
        total_sum=Decimal("0.75"),
        deviation=Decimal("0.25"),
        deviation_pct=0.25,
        optimal_allocation={"m1": Decimal("1500"), "m2": Decimal("1500"), "m3": Decimal("2000")},
        expected_profit=Decimal("100.00"),
        expected_profit_pct=0.20,
        bottleneck_market_id="m1",
        bottleneck_liquidity=Decimal("5000"),
        max_executable_size=Decimal("5000"),
        vwap_prices={"m1": Decimal("0.21"), "m2": Decimal("0.26"), "m3": Decimal("0.31")},
        vwap_validated=True,
        estimated_slippage=Decimal("0.005")
    )
    
    signals = strategy._generate_signals(opportunity)
    
    # Should generate signal for each market
    assert len(signals) == 3
    
    # All signals should be YES (buy all)
    assert all(s.side == OrderSide.YES for s in signals)
    
    # Signals should have correct markets
    signal_markets = {s.market_id for s in signals}
    assert signal_markets == {"m1", "m2", "m3"}
    
    # All signals should have target prices
    assert all(s.target_price is not None for s in signals)


def test_signal_generation_sell_all(strategy):
    """Test signal generation for sell-all opportunity."""
    markets = [
        create_market("m1", "Election - A", Decimal("0.35")),
        create_market("m2", "Election - B", Decimal("0.40")),
        create_market("m3", "Election - C", Decimal("0.35")),
    ]
    
    opportunity = RebalancingOpportunity(
        market_ids=["m1", "m2", "m3"],
        markets=markets,
        direction="sell_all",
        total_sum=Decimal("1.10"),
        deviation=Decimal("0.10"),
        deviation_pct=0.10,
        optimal_allocation={"m1": Decimal("1500"), "m2": Decimal("1500"), "m3": Decimal("2000")},
        expected_profit=Decimal("50.00"),
        expected_profit_pct=0.10,
        bottleneck_market_id="m1",
        bottleneck_liquidity=Decimal("5000"),
        max_executable_size=Decimal("5000"),
        vwap_prices={"m1": Decimal("0.36"), "m2": Decimal("0.41"), "m3": Decimal("0.36")},
        vwap_validated=True,
        estimated_slippage=Decimal("0.005")
    )
    
    signals = strategy._generate_signals(opportunity)
    
    # Should generate signal for each market
    assert len(signals) == 3
    
    # All signals should be NO (sell YES = buy NO)
    assert all(s.side == OrderSide.NO for s in signals)


# ============================================================================
# Test: Position Tracking
# ============================================================================

def test_position_tracking_basic(strategy):
    """Test basic position tracking."""
    # Set up market group
    strategy.market_groups["group1"] = ["m1", "m2", "m3"]
    strategy.market_to_group = {"m1": "group1", "m2": "group1", "m3": "group1"}
    
    # Update positions
    strategy.update_position("m1", 100)
    strategy.update_position("m2", 150)
    strategy.update_position("m3", 200)
    
    # Check positions
    assert strategy.positions["group1"]["m1"] == 100
    assert strategy.positions["group1"]["m2"] == 150
    assert strategy.positions["group1"]["m3"] == 200


def test_position_tracking_complete_position(strategy):
    """Test detection of complete multi-leg position."""
    # Set up market group
    strategy.market_groups["group1"] = ["m1", "m2", "m3"]
    strategy.market_to_group = {"m1": "group1", "m2": "group1", "m3": "group1"}
    
    # Fill all legs
    strategy.update_position("m1", 100)
    strategy.update_position("m2", 100)
    strategy.update_position("m3", 100)
    
    # Should mark as executed
    assert "group1" in strategy.executed_opportunities
    assert strategy.opportunities_executed == 1


def test_position_tracking_incremental_fills(strategy):
    """Test incremental position updates (partial fills)."""
    strategy.market_groups["group1"] = ["m1"]
    strategy.market_to_group = {"m1": "group1"}
    
    # Update in increments
    strategy.update_position("m1", 50)
    assert strategy.positions["group1"]["m1"] == 50
    
    strategy.update_position("m1", 30)
    assert strategy.positions["group1"]["m1"] == 80
    
    strategy.update_position("m1", 20)
    assert strategy.positions["group1"]["m1"] == 100


# ============================================================================
# Test: Full Strategy Flow
# ============================================================================

@pytest.mark.asyncio
async def test_full_strategy_scan_and_detect(strategy):
    """Test complete strategy flow: scan markets and detect opportunities."""
    # Create multi-outcome markets with arbitrage
    markets = [
        # Group 1: Arbitrage opportunity (sum = 0.75)
        create_market("m1", "2024 Election - Trump", Decimal("0.20")),
        create_market("m2", "2024 Election - Biden", Decimal("0.25")),
        create_market("m3", "2024 Election - Other", Decimal("0.30")),
        
        # Group 2: No arbitrage (sum = 1.00)
        create_market("m4", "Weather - Sunny", Decimal("0.40")),
        create_market("m5", "Weather - Rainy", Decimal("0.35")),
        create_market("m6", "Weather - Cloudy", Decimal("0.25")),
    ]
    
    signals = await strategy.scan_markets(markets)
    
    # Should detect only Group 1 (arbitrage)
    assert len(strategy.active_opportunities) >= 1
    
    # Should generate signals for Group 1
    assert len(signals) >= 3  # One per market in the arbitrage group
    
    # All signals for same group should be same direction
    if signals:
        first_side = signals[0].side
        assert all(s.side == first_side for s in signals[:3])


@pytest.mark.asyncio
async def test_strategy_skip_resolved_markets(strategy):
    """Test that strategy skips resolved markets."""
    markets = [
        create_market("m1", "Election - A", Decimal("0.20")),
        create_market("m2", "Election - B", Decimal("0.25")),
        create_market("m3", "Election - C", Decimal("0.30")),
    ]
    
    # Mark one as resolved
    markets[0].resolved = True
    
    signals = await strategy.scan_markets(markets)
    
    # Should not generate signals (group incomplete due to resolved market)
    # Group won't form since resolved markets are filtered
    assert len(signals) == 0


@pytest.mark.asyncio
async def test_strategy_timing_constraints(strategy):
    """Test that strategy respects timing constraints."""
    # Create markets too close to resolution
    markets = [
        create_market("m1", "Election - A", Decimal("0.20"), hours_to_close=0.5),  # 30 min
        create_market("m2", "Election - B", Decimal("0.25"), hours_to_close=0.5),
        create_market("m3", "Election - C", Decimal("0.30"), hours_to_close=0.5),
    ]
    
    signals = await strategy.scan_markets(markets)
    
    # Should not generate signals (too close to resolution)
    assert len(signals) == 0


# ============================================================================
# Test: Edge Cases
# ============================================================================

def test_extreme_deviation_rejection(strategy):
    """Test that extreme deviations are rejected (likely data errors)."""
    markets = [
        create_market("m1", "Election - A", Decimal("0.10")),
        create_market("m2", "Election - B", Decimal("0.10")),
        create_market("m3", "Election - C", Decimal("0.10")),
    ]
    # Sum = 0.30 (70% deviation) - likely data error
    
    # This should be caught by max_deviation check
    # Update config for this test
    strategy.config.max_deviation = Decimal("0.50")
    
    opportunity = asyncio.run(strategy._detect_opportunity(markets))
    
    # Should reject extreme deviation
    assert opportunity is None


def test_zero_liquidity_handling(strategy):
    """Test handling of zero liquidity markets."""
    markets = [
        create_market("m1", "Election - A", Decimal("0.20"), liquidity=Decimal("0")),
        create_market("m2", "Election - B", Decimal("0.25"), liquidity=Decimal("5000")),
        create_market("m3", "Election - C", Decimal("0.30"), liquidity=Decimal("5000")),
    ]
    
    opportunity = asyncio.run(strategy._detect_opportunity(markets))
    
    # Should reject due to zero liquidity in one market
    assert opportunity is None


def test_position_size_calculation(strategy):
    """Test position size calculation with account constraints."""
    market = create_market("m1", "Test", Decimal("0.25"))
    
    signal = Signal(
        market_id="m1",
        market=market,
        side=OrderSide.YES,
        strength=0.5,
        reason="Test",
        target_price=Decimal("0.25")
    )
    
    account_balance = Decimal("10000")
    
    size = strategy.get_position_size(signal, account_balance)
    
    # Should return reasonable position size
    assert size > 0
    
    # Should not exceed account constraints
    position_value = size * signal.target_price
    max_allowed = account_balance * Decimal(str(strategy.config.max_capital_per_trade))
    assert position_value <= max_allowed


# ============================================================================
# Test: Performance Metrics
# ============================================================================

def test_performance_stats_tracking(strategy):
    """Test performance statistics tracking."""
    # Simulate detected opportunities
    strategy.opportunities_detected = 10
    strategy.opportunities_executed = 7
    strategy.total_profit = Decimal("500.50")
    
    stats = strategy.get_performance_stats()
    
    assert stats["opportunities_detected"] == 10
    assert stats["opportunities_executed"] == 7
    assert stats["win_rate"] == "70.0%"
    assert stats["total_profit"] == "$500.50"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    # Run a simple integration test
    async def integration_test():
        print("=== Market Rebalancing Strategy Test ===\n")
        
        strategy = MarketRebalancingStrategy()
        
        # Create test markets
        markets = [
            create_market("m1", "2024 Election - Trump", Decimal("0.22")),
            create_market("m2", "2024 Election - Biden", Decimal("0.28")),
            create_market("m3", "2024 Election - RFK Jr", Decimal("0.18")),
            create_market("m4", "2024 Election - Other", Decimal("0.12")),
        ]
        
        print(f"Created {len(markets)} test markets")
        total_sum = sum(m.yes_price for m in markets)
        print(f"Total probability sum: {total_sum:.3f} (should be 1.000)")
        print(f"Deviation: {abs(total_sum - Decimal('1.0')):.3f}\n")
        
        # Scan for opportunities
        signals = await strategy.scan_markets(markets)
        
        print(f"Detected {len(strategy.active_opportunities)} opportunities")
        print(f"Generated {len(signals)} signals\n")
        
        # Print opportunity details
        for group_id, opp in strategy.active_opportunities.items():
            print(f"Opportunity: {group_id}")
            print(f"  Direction: {opp.direction}")
            print(f"  Total Sum: {opp.total_sum:.3f}")
            print(f"  Deviation: {opp.deviation_pct:.2%}")
            print(f"  Expected Profit: ${opp.expected_profit:.2f}")
            print(f"  ROI: {opp.expected_profit_pct:.1%}")
            print(f"  Markets: {len(opp.markets)}")
            print(f"  Max Size: ${opp.max_executable_size:.2f}\n")
        
        # Print signals
        if signals:
            print("Signals generated:")
            for signal in signals:
                print(f"  {signal.market.ticker}: {signal.side.value.upper()} @ {signal.target_price:.3f}")
                print(f"    Reason: {signal.reason}\n")
        
        print("=== Test Complete ===")
    
    asyncio.run(integration_test())
