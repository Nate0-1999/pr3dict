"""
PR3DICT: VWAP Module Tests

Comprehensive testing for VWAP calculation, validation, and integration.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

from src.data.vwap import (
    VWAPCalculator,
    VWAPValidator,
    VWAPMonitor,
    VWAPResult,
    LiquidityMetrics,
    quick_vwap_check
)
from src.platforms.base import OrderBook, OrderSide


class TestVWAPCalculator:
    """Test core VWAP calculation logic."""
    
    def test_simple_vwap_calculation(self):
        """Test basic VWAP calculation with uniform pricing."""
        calc = VWAPCalculator()
        
        # Simple order book: 100 contracts at each price level
        orders = [
            (Decimal("0.50"), 100),
            (Decimal("0.51"), 100),
            (Decimal("0.52"), 100),
        ]
        
        result = calc.calculate_vwap(
            orders=orders,
            quantity=200,
            side="buy",
            market_id="test_market"
        )
        
        # Should execute 100 @ 0.50 + 100 @ 0.51
        expected_vwap = (Decimal("0.50") * 100 + Decimal("0.51") * 100) / 200
        
        assert result.liquidity_sufficient
        assert result.vwap_price == expected_vwap
        assert result.target_quantity == 200
        assert len(result.fills) == 2
    
    def test_insufficient_liquidity(self):
        """Test handling of insufficient liquidity."""
        calc = VWAPCalculator()
        
        orders = [
            (Decimal("0.50"), 50),
            (Decimal("0.51"), 50),
        ]
        
        result = calc.calculate_vwap(
            orders=orders,
            quantity=200,  # More than available
            side="buy",
            market_id="test_market"
        )
        
        assert not result.liquidity_sufficient
        assert result.execution_quality == "INSUFFICIENT_LIQUIDITY"
    
    def test_slippage_calculation(self):
        """Test slippage calculation accuracy."""
        calc = VWAPCalculator()
        
        orders = [
            (Decimal("0.50"), 100),
            (Decimal("0.55"), 100),  # 10% higher
        ]
        
        result = calc.calculate_vwap(
            orders=orders,
            quantity=150,
            side="buy",
            market_id="test_market",
            quoted_price=Decimal("0.50")  # Best price
        )
        
        # VWAP = (0.50 * 100 + 0.55 * 50) / 150 = 0.5167
        # Slippage = (0.5167 - 0.50) / 0.50 = 3.33%
        
        assert result.liquidity_sufficient
        assert result.slippage_pct > Decimal("3.0")
        assert result.slippage_pct < Decimal("4.0")
    
    def test_sell_side_vwap(self):
        """Test VWAP calculation for sell orders."""
        calc = VWAPCalculator()
        
        # For selling, we take bids (descending)
        orders = [
            (Decimal("0.60"), 100),
            (Decimal("0.59"), 100),
            (Decimal("0.58"), 100),
        ]
        
        result = calc.calculate_vwap(
            orders=orders,
            quantity=150,
            side="sell",
            market_id="test_market"
        )
        
        # Should execute 100 @ 0.60 + 50 @ 0.59
        expected_vwap = (Decimal("0.60") * 100 + Decimal("0.59") * 50) / 150
        
        assert result.vwap_price == expected_vwap
        assert result.liquidity_sufficient
    
    def test_liquidity_metrics(self):
        """Test liquidity metrics calculation."""
        calc = VWAPCalculator()
        
        bids = [
            (Decimal("0.49"), 200),
            (Decimal("0.48"), 300),
        ]
        asks = [
            (Decimal("0.51"), 150),
            (Decimal("0.52"), 250),
        ]
        
        metrics = calc.calculate_liquidity_metrics(bids, asks, "test_market")
        
        assert metrics.bid_depth == 500
        assert metrics.ask_depth == 400
        assert metrics.spread_bps > 0  # Has spread
        assert metrics.top_of_book_size == 150  # Min of best bid/ask
    
    def test_price_impact_curve(self):
        """Test price impact curve generation."""
        calc = VWAPCalculator()
        
        orders = [
            (Decimal("0.50"), 100),
            (Decimal("0.51"), 200),
            (Decimal("0.53"), 300),
            (Decimal("0.56"), 400),
        ]
        
        curve = calc.build_price_impact_curve(
            orders=orders,
            side="buy",
            market_id="test_market",
            sample_sizes=[50, 100, 200, 400]
        )
        
        assert len(curve.data_points) > 0
        # Price should increase with size
        prices = [price for _, price in curve.data_points]
        assert prices == sorted(prices)  # Should be ascending


class TestVWAPValidator:
    """Test VWAP validation logic."""
    
    def test_valid_execution(self):
        """Test validation of good execution."""
        calc = VWAPCalculator()
        validator = VWAPValidator(
            calculator=calc,
            max_slippage_pct=Decimal("2.0"),
            min_liquidity_contracts=100
        )
        
        # Good VWAP result
        vwap_result = VWAPResult(
            market_id="test",
            side="buy",
            target_quantity=100,
            quoted_price=Decimal("0.50"),
            vwap_price=Decimal("0.505"),  # 1% slippage
            total_cost=Decimal("50.5"),
            slippage_pct=Decimal("1.0"),
            slippage_absolute=Decimal("0.005"),
            price_impact_pct=Decimal("1.0"),
            fills=[(Decimal("0.50"), 50), (Decimal("0.51"), 50)],
            depth_used=2,
            liquidity_sufficient=True
        )
        
        liquidity = LiquidityMetrics(
            market_id="test",
            bid_depth=500,
            ask_depth=500,
            bid_value=Decimal("250"),
            ask_value=Decimal("255"),
            spread_bps=200,
            top_of_book_size=100,
            depth_imbalance=Decimal("0.5")
        )
        
        is_valid, reason = validator.validate_execution(vwap_result, liquidity)
        
        assert is_valid
        assert reason == "OK"
    
    def test_reject_high_slippage(self):
        """Test rejection of high slippage."""
        calc = VWAPCalculator()
        validator = VWAPValidator(
            calculator=calc,
            max_slippage_pct=Decimal("2.0")
        )
        
        # Bad VWAP result (high slippage)
        vwap_result = VWAPResult(
            market_id="test",
            side="buy",
            target_quantity=100,
            quoted_price=Decimal("0.50"),
            vwap_price=Decimal("0.52"),  # 4% slippage
            total_cost=Decimal("52"),
            slippage_pct=Decimal("4.0"),
            slippage_absolute=Decimal("0.02"),
            price_impact_pct=Decimal("4.0"),
            fills=[],
            depth_used=5,
            liquidity_sufficient=True
        )
        
        liquidity = LiquidityMetrics(
            market_id="test",
            bid_depth=1000,
            ask_depth=1000,
            bid_value=Decimal("500"),
            ask_value=Decimal("500"),
            spread_bps=100,
            top_of_book_size=200,
            depth_imbalance=Decimal("0.5")
        )
        
        is_valid, reason = validator.validate_execution(vwap_result, liquidity)
        
        assert not is_valid
        assert "Slippage" in reason
    
    def test_reject_low_liquidity(self):
        """Test rejection of low liquidity."""
        calc = VWAPCalculator()
        validator = VWAPValidator(
            calculator=calc,
            min_liquidity_contracts=1000
        )
        
        vwap_result = VWAPResult(
            market_id="test",
            side="buy",
            target_quantity=100,
            quoted_price=Decimal("0.50"),
            vwap_price=Decimal("0.505"),
            total_cost=Decimal("50.5"),
            slippage_pct=Decimal("1.0"),
            slippage_absolute=Decimal("0.005"),
            price_impact_pct=Decimal("1.0"),
            fills=[],
            depth_used=2,
            liquidity_sufficient=True
        )
        
        # Low liquidity
        liquidity = LiquidityMetrics(
            market_id="test",
            bid_depth=200,
            ask_depth=300,  # Less than 1000 required
            bid_value=Decimal("100"),
            ask_value=Decimal("150"),
            spread_bps=200,
            top_of_book_size=50,
            depth_imbalance=Decimal("0.4")
        )
        
        is_valid, reason = validator.validate_execution(vwap_result, liquidity)
        
        assert not is_valid
        assert "Depth" in reason
    
    def test_order_split_suggestion(self):
        """Test order split suggestions."""
        calc = VWAPCalculator()
        validator = VWAPValidator(calculator=calc)
        
        # VWAP result with poor quality (should suggest split)
        vwap_result = VWAPResult(
            market_id="test",
            side="buy",
            target_quantity=1000,
            quoted_price=Decimal("0.50"),
            vwap_price=Decimal("0.55"),
            total_cost=Decimal("550"),
            slippage_pct=Decimal("10.0"),
            slippage_absolute=Decimal("0.05"),
            price_impact_pct=Decimal("10.0"),
            fills=[
                (Decimal("0.50"), 200),
                (Decimal("0.52"), 300),
                (Decimal("0.58"), 500),
            ],
            depth_used=3,
            liquidity_sufficient=True
        )
        
        chunks = validator.suggest_order_split(vwap_result, max_chunks=3)
        
        assert len(chunks) > 1  # Should split
        assert sum(chunks) == 1000  # Total should match


class TestVWAPMonitor:
    """Test VWAP monitoring and statistics."""
    
    def test_record_execution(self):
        """Test execution recording."""
        calc = VWAPCalculator()
        monitor = VWAPMonitor(calc)
        
        # Create sample result
        result = VWAPResult(
            market_id="test",
            side="buy",
            target_quantity=100,
            quoted_price=Decimal("0.50"),
            vwap_price=Decimal("0.505"),
            total_cost=Decimal("50.5"),
            slippage_pct=Decimal("1.0"),
            slippage_absolute=Decimal("0.005"),
            price_impact_pct=Decimal("1.0"),
            fills=[],
            depth_used=2,
            liquidity_sufficient=True
        )
        
        monitor.record_execution(result)
        
        assert len(monitor.execution_history) == 1
        
        stats = monitor.get_execution_stats()
        assert stats['total_executions'] == 1
        assert stats['avg_slippage_pct'] == 1.0
    
    def test_execution_statistics(self):
        """Test aggregated statistics."""
        calc = VWAPCalculator()
        monitor = VWAPMonitor(calc)
        
        # Record multiple executions
        for i in range(10):
            slippage = Decimal(str(i * 0.5))  # 0%, 0.5%, 1%, ...
            result = VWAPResult(
                market_id=f"test_{i}",
                side="buy",
                target_quantity=100,
                quoted_price=Decimal("0.50"),
                vwap_price=Decimal("0.50") + slippage / 100,
                total_cost=Decimal("50"),
                slippage_pct=slippage,
                slippage_absolute=slippage / 100,
                price_impact_pct=slippage,
                fills=[],
                depth_used=1,
                liquidity_sufficient=True
            )
            monitor.record_execution(result)
        
        stats = monitor.get_execution_stats()
        
        assert stats['total_executions'] == 10
        assert stats['min_slippage_pct'] == 0.0
        assert stats['max_slippage_pct'] == 4.5


class TestQuickVWAPCheck:
    """Test convenience function."""
    
    def test_quick_check(self):
        """Test quick VWAP check function."""
        bids = [(Decimal("0.49"), 100)]
        asks = [(Decimal("0.51"), 100)]
        
        result = quick_vwap_check(
            bids=bids,
            asks=asks,
            quantity=50,
            side="buy",
            market_id="test"
        )
        
        assert result.liquidity_sufficient
        assert result.vwap_price == Decimal("0.51")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
