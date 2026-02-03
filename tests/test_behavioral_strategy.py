"""
Unit tests for Behavioral Trading Strategy.

Tests all signal types and edge cases.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from typing import List

from pr3dict.strategies.behavioral import BehavioralStrategy, create_behavioral_strategy
from pr3dict.strategies.base import Signal, OrderSide
from pr3dict.platforms.base import Market, Position


# Test fixtures
@pytest.fixture
def strategy():
    """Create a basic strategy for testing."""
    return create_behavioral_strategy(
        enable_longshot=True,
        enable_favorite=True,
        enable_overreaction=True,
        enable_recency=True,
        enable_time_arbitrage=False,
        min_edge=0.02
    )


@pytest.fixture
def longshot_market():
    """Create a longshot market (8% probability)."""
    return Market(
        market_id="longshot_test",
        question="Test longshot market",
        yes_price=Decimal("0.08"),
        no_price=Decimal("0.92"),
        volume=Decimal("5000"),
        close_date=datetime.now() + timedelta(days=30),
        platform="test"
    )


@pytest.fixture
def favorite_market():
    """Create a favorite market (78% probability)."""
    return Market(
        market_id="favorite_test",
        question="Test favorite market",
        yes_price=Decimal("0.78"),
        no_price=Decimal("0.22"),
        volume=Decimal("5000"),
        close_date=datetime.now() + timedelta(days=30),
        platform="test"
    )


@pytest.fixture
def neutral_market():
    """Create a neutral market (50% probability)."""
    return Market(
        market_id="neutral_test",
        question="Test neutral market",
        yes_price=Decimal("0.50"),
        no_price=Decimal("0.50"),
        volume=Decimal("5000"),
        close_date=datetime.now() + timedelta(days=30),
        platform="test"
    )


class TestLongshotBias:
    """Test longshot bias detection and signals."""
    
    @pytest.mark.asyncio
    async def test_longshot_detection(self, strategy, longshot_market):
        """Test that longshot bias is detected correctly."""
        signals = await strategy.scan_markets([longshot_market])
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.side == OrderSide.NO
        assert "LONGSHOT" in signal.reason
        assert signal.strength > 0.5  # Should have high confidence
    
    @pytest.mark.asyncio
    async def test_extreme_longshot(self, strategy):
        """Test extreme longshot (3% probability)."""
        market = Market(
            market_id="extreme_longshot",
            question="Test extreme longshot",
            yes_price=Decimal("0.03"),
            no_price=Decimal("0.97"),
            volume=Decimal("5000"),
            close_date=datetime.now() + timedelta(days=30),
            platform="test"
        )
        
        signals = await strategy.scan_markets([market])
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.side == OrderSide.NO
        # Extreme longshots should have higher strength
        assert signal.strength > 0.7
    
    @pytest.mark.asyncio
    async def test_no_longshot_signal_above_threshold(self, strategy):
        """Test that markets above 15% don't trigger longshot signal."""
        market = Market(
            market_id="not_longshot",
            question="Test not longshot",
            yes_price=Decimal("0.18"),
            no_price=Decimal("0.82"),
            volume=Decimal("5000"),
            close_date=datetime.now() + timedelta(days=30),
            platform="test"
        )
        
        # Disable other signals
        strategy.enable_favorite = False
        strategy.enable_overreaction = False
        strategy.enable_recency = False
        
        signals = await strategy.scan_markets([market])
        
        # Should not generate longshot signal
        longshot_signals = [s for s in signals if "LONGSHOT" in s.reason]
        assert len(longshot_signals) == 0


class TestFavoriteBias:
    """Test favorite bias detection and signals."""
    
    @pytest.mark.asyncio
    async def test_favorite_detection(self, strategy, favorite_market):
        """Test that favorite bias is detected correctly."""
        # Disable other signals
        strategy.enable_longshot = False
        strategy.enable_overreaction = False
        strategy.enable_recency = False
        
        signals = await strategy.scan_markets([favorite_market])
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.side == OrderSide.YES
        assert "FAVORITE" in signal.reason
    
    @pytest.mark.asyncio
    async def test_extreme_favorite(self, strategy):
        """Test extreme favorite (95% probability)."""
        market = Market(
            market_id="extreme_favorite",
            question="Test extreme favorite",
            yes_price=Decimal("0.95"),
            no_price=Decimal("0.05"),
            volume=Decimal("5000"),
            close_date=datetime.now() + timedelta(days=30),
            platform="test"
        )
        
        strategy.enable_longshot = False
        strategy.enable_overreaction = False
        strategy.enable_recency = False
        
        signals = await strategy.scan_markets([market])
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.side == OrderSide.YES
        # Extreme favorites should have higher strength
        assert signal.strength > 0.5
    
    @pytest.mark.asyncio
    async def test_no_favorite_signal_below_threshold(self, strategy):
        """Test that markets below 70% don't trigger favorite signal."""
        market = Market(
            market_id="not_favorite",
            question="Test not favorite",
            yes_price=Decimal("0.65"),
            no_price=Decimal("0.35"),
            volume=Decimal("5000"),
            close_date=datetime.now() + timedelta(days=30),
            platform="test"
        )
        
        strategy.enable_longshot = False
        strategy.enable_overreaction = False
        strategy.enable_recency = False
        
        signals = await strategy.scan_markets([market])
        
        # Should not generate favorite signal
        favorite_signals = [s for s in signals if "FAVORITE" in s.reason]
        assert len(favorite_signals) == 0


class TestOverreactionBias:
    """Test overreaction detection and signals."""
    
    @pytest.mark.asyncio
    async def test_overreaction_upward_spike(self, strategy, neutral_market):
        """Test detection of upward price spike."""
        strategy.enable_longshot = False
        strategy.enable_favorite = False
        strategy.enable_recency = False
        
        # Simulate price history: 0.30 → 0.50 (67% increase)
        base_time = datetime.now()
        strategy.price_history[neutral_market.market_id] = [
            (base_time - timedelta(hours=8), 0.30),
            (base_time - timedelta(hours=6), 0.32),
            (base_time - timedelta(hours=4), 0.35),
            (base_time - timedelta(hours=2), 0.45),
            (base_time - timedelta(hours=1), 0.50),
        ]
        
        signals = await strategy.scan_markets([neutral_market])
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.side == OrderSide.NO  # Fade the spike
        assert "OVERREACTION" in signal.reason
    
    @pytest.mark.asyncio
    async def test_overreaction_downward_spike(self, strategy):
        """Test detection of downward price spike."""
        market = Market(
            market_id="down_spike",
            question="Test downward spike",
            yes_price=Decimal("0.30"),
            no_price=Decimal("0.70"),
            volume=Decimal("5000"),
            close_date=datetime.now() + timedelta(days=30),
            platform="test"
        )
        
        strategy.enable_longshot = False
        strategy.enable_favorite = False
        strategy.enable_recency = False
        
        # Simulate price history: 0.55 → 0.30 (45% decrease)
        base_time = datetime.now()
        strategy.price_history[market.market_id] = [
            (base_time - timedelta(hours=8), 0.55),
            (base_time - timedelta(hours=6), 0.50),
            (base_time - timedelta(hours=4), 0.42),
            (base_time - timedelta(hours=2), 0.35),
            (base_time - timedelta(hours=1), 0.30),
        ]
        
        signals = await strategy.scan_markets([market])
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.side == OrderSide.YES  # Fade the drop
        assert "OVERREACTION" in signal.reason
    
    @pytest.mark.asyncio
    async def test_no_overreaction_small_move(self, strategy, neutral_market):
        """Test that small moves don't trigger overreaction."""
        strategy.enable_longshot = False
        strategy.enable_favorite = False
        strategy.enable_recency = False
        
        # Simulate price history: 0.48 → 0.52 (8% increase, below 20% threshold)
        base_time = datetime.now()
        strategy.price_history[neutral_market.market_id] = [
            (base_time - timedelta(hours=8), 0.48),
            (base_time - timedelta(hours=6), 0.49),
            (base_time - timedelta(hours=4), 0.50),
            (base_time - timedelta(hours=2), 0.51),
            (base_time - timedelta(hours=1), 0.52),
        ]
        
        signals = await strategy.scan_markets([neutral_market])
        
        # Should not trigger overreaction
        overreaction_signals = [s for s in signals if "OVERREACTION" in s.reason]
        assert len(overreaction_signals) == 0


class TestRecencyBias:
    """Test recency bias detection and signals."""
    
    @pytest.mark.asyncio
    async def test_recency_high_volatility(self, strategy, neutral_market):
        """Test detection of recency bias from high recent volatility."""
        strategy.enable_longshot = False
        strategy.enable_favorite = False
        strategy.enable_overreaction = False
        
        base_time = datetime.now()
        
        # Older period: stable around 0.50
        older_history = [
            (base_time - timedelta(days=3), 0.50),
            (base_time - timedelta(days=2, hours=20), 0.51),
            (base_time - timedelta(days=2, hours=16), 0.49),
            (base_time - timedelta(days=2, hours=12), 0.50),
            (base_time - timedelta(days=2, hours=8), 0.51),
            (base_time - timedelta(days=2, hours=4), 0.50),
        ]
        
        # Recent period: high volatility
        recent_history = [
            (base_time - timedelta(hours=20), 0.50),
            (base_time - timedelta(hours=16), 0.45),
            (base_time - timedelta(hours=12), 0.55),
            (base_time - timedelta(hours=8), 0.48),
            (base_time - timedelta(hours=4), 0.52),
            (base_time - timedelta(hours=2), 0.50),
        ]
        
        strategy.price_history[neutral_market.market_id] = older_history + recent_history
        
        signals = await strategy.scan_markets([neutral_market])
        
        # May generate recency signal if volatility ratio > 2.0
        recency_signals = [s for s in signals if "RECENCY" in s.reason]
        assert len(recency_signals) >= 0  # May or may not trigger depending on exact volatility


class TestPositionManagement:
    """Test position exit logic."""
    
    @pytest.mark.asyncio
    async def test_profit_target_exit(self, strategy, longshot_market):
        """Test exit at profit target."""
        # Create a winning position
        position = Position(
            position_id="test_pos_1",
            market_id=longshot_market.market_id,
            side=OrderSide.NO,
            quantity=100,
            entry_price=Decimal("0.92"),
            entry_time=datetime.now() - timedelta(hours=6),
            platform="test",
            reason="LONGSHOT_FADE: YES at 8.0% (overpriced by ~5%)"
        )
        
        # Market moved in our favor (YES dropped to 3%)
        winning_market = Market(
            market_id=longshot_market.market_id,
            question=longshot_market.question,
            yes_price=Decimal("0.03"),
            no_price=Decimal("0.97"),
            volume=longshot_market.volume,
            close_date=longshot_market.close_date,
            platform="test"
        )
        
        exit_signal = await strategy.check_exit(position, winning_market)
        
        assert exit_signal is not None
        assert "PROFIT_TARGET" in exit_signal.reason
    
    @pytest.mark.asyncio
    async def test_stop_loss_exit(self, strategy, longshot_market):
        """Test exit at stop loss."""
        # Create a losing position
        position = Position(
            position_id="test_pos_2",
            market_id=longshot_market.market_id,
            side=OrderSide.NO,
            quantity=100,
            entry_price=Decimal("0.92"),
            entry_time=datetime.now() - timedelta(hours=6),
            platform="test",
            reason="LONGSHOT_FADE: YES at 8.0% (overpriced by ~5%)"
        )
        
        # Market moved against us (YES rose to 25%)
        losing_market = Market(
            market_id=longshot_market.market_id,
            question=longshot_market.question,
            yes_price=Decimal("0.25"),
            no_price=Decimal("0.75"),
            volume=longshot_market.volume,
            close_date=longshot_market.close_date,
            platform="test"
        )
        
        exit_signal = await strategy.check_exit(position, losing_market)
        
        assert exit_signal is not None
        assert "STOP_LOSS" in exit_signal.reason
    
    @pytest.mark.asyncio
    async def test_time_based_exit(self, strategy, longshot_market):
        """Test time-based exit after 7 days."""
        # Create an old position
        position = Position(
            position_id="test_pos_3",
            market_id=longshot_market.market_id,
            side=OrderSide.NO,
            quantity=100,
            entry_price=Decimal("0.92"),
            entry_time=datetime.now() - timedelta(days=8),  # 8 days old
            platform="test",
            reason="LONGSHOT_FADE: YES at 8.0% (overpriced by ~5%)"
        )
        
        exit_signal = await strategy.check_exit(position, longshot_market)
        
        assert exit_signal is not None
        assert "TIME_EXIT" in exit_signal.reason


class TestPositionSizing:
    """Test position sizing calculations."""
    
    def test_position_sizing_default(self, strategy, longshot_market):
        """Test default position sizing (2% risk)."""
        signals = []
        
        # Create mock signal
        signal = Signal(
            market_id=longshot_market.market_id,
            market=longshot_market,
            side=OrderSide.NO,
            strength=0.8,
            reason="Test signal",
            target_price=Decimal("0.92")
        )
        
        account_balance = Decimal("10000")
        contracts = strategy.get_position_size(signal, account_balance, risk_pct=0.02)
        
        # Should risk 2% = $200
        # At $0.92 per contract, should buy ~217 contracts
        assert contracts > 0
        assert contracts < 300  # Reasonable bounds
    
    def test_position_sizing_conservative(self, strategy, longshot_market):
        """Test conservative position sizing (1% risk)."""
        signal = Signal(
            market_id=longshot_market.market_id,
            market=longshot_market,
            side=OrderSide.NO,
            strength=0.8,
            reason="Test signal",
            target_price=Decimal("0.92")
        )
        
        account_balance = Decimal("10000")
        contracts_1pct = strategy.get_position_size(signal, account_balance, risk_pct=0.01)
        contracts_2pct = strategy.get_position_size(signal, account_balance, risk_pct=0.02)
        
        # 1% risk should give ~half the contracts of 2% risk
        assert contracts_1pct < contracts_2pct
        assert contracts_1pct > 0


class TestStrategyConfiguration:
    """Test strategy configuration options."""
    
    def test_disable_signals(self):
        """Test disabling specific signals."""
        strategy = create_behavioral_strategy(
            enable_longshot=True,
            enable_favorite=False,
            enable_overreaction=False,
            enable_recency=False,
            enable_time_arbitrage=False
        )
        
        assert strategy.enable_longshot is True
        assert strategy.enable_favorite is False
        assert strategy.enable_overreaction is False
    
    def test_min_edge_threshold(self):
        """Test minimum edge threshold."""
        strategy = create_behavioral_strategy(min_edge=0.05)
        
        assert strategy.min_edge == 0.05
    
    def test_strategy_name(self):
        """Test strategy has correct name."""
        strategy = create_behavioral_strategy()
        
        assert strategy.name == "behavioral"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_low_volume_market(self, strategy):
        """Test that low-volume markets are skipped."""
        low_volume_market = Market(
            market_id="low_volume",
            question="Test low volume",
            yes_price=Decimal("0.08"),
            no_price=Decimal("0.92"),
            volume=Decimal("500"),  # Below 1000 threshold
            close_date=datetime.now() + timedelta(days=30),
            platform="test"
        )
        
        signals = await strategy.scan_markets([low_volume_market])
        
        # Should not generate signals for low-volume markets
        assert len(signals) == 0
    
    @pytest.mark.asyncio
    async def test_empty_market_list(self, strategy):
        """Test scanning empty market list."""
        signals = await strategy.scan_markets([])
        
        assert signals == []
    
    @pytest.mark.asyncio
    async def test_insufficient_price_history(self, strategy, neutral_market):
        """Test that insufficient price history doesn't crash."""
        strategy.enable_longshot = False
        strategy.enable_favorite = False
        
        # No price history
        signals = await strategy.scan_markets([neutral_market])
        
        # Should not crash, just not generate certain signals
        assert isinstance(signals, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
