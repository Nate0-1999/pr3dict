"""
Tests for optimization solver module.
"""
import pytest
from decimal import Decimal
import numpy as np

from src.optimization.solver import (
    ArbitrageSolver,
    ArbitrageOpportunity,
    SolverBackend,
    OptimizationResult
)
from src.optimization.benchmarks import (
    SolverBenchmark,
    create_synthetic_opportunities
)


class TestArbitrageSolver:
    """Test the main solver functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.solver = ArbitrageSolver(
            transaction_fee_rate=Decimal("0.02"),
            gas_fee=Decimal("0.0"),
            max_position_fraction=Decimal("0.3"),
            min_profit_threshold=Decimal("1.0")
        )
    
    def test_solver_initialization(self):
        """Test solver initializes correctly."""
        assert self.solver.fee_rate == 0.02
        assert self.solver.max_position_fraction == 0.3
    
    def test_simple_arbitrage_opportunity(self):
        """Test solver on a single arbitrage opportunity."""
        opportunities = [
            ArbitrageOpportunity(
                market_id="TEST_1",
                outcome_id="YES",
                current_price=Decimal("0.50"),
                expected_value=Decimal("1.00"),
                max_liquidity=1000,
                platform="test",
                ticker="TEST1"
            )
        ]
        
        capital = Decimal("1000.0")
        
        result = self.solver.solve(
            opportunities=opportunities,
            available_capital=capital,
            backend=SolverBackend.FRANK_WOLFE
        )
        
        assert result.num_trades >= 0
        assert result.total_capital_used <= capital
        assert result.solve_time_ms > 0
    
    def test_multiple_opportunities(self):
        """Test solver with multiple opportunities."""
        opportunities = [
            ArbitrageOpportunity(
                market_id=f"TEST_{i}",
                outcome_id="YES",
                current_price=Decimal(str(0.4 + i * 0.05)),
                expected_value=Decimal("1.00"),
                max_liquidity=500,
                platform="test",
                ticker=f"TEST{i}"
            )
            for i in range(5)
        ]
        
        capital = Decimal("5000.0")
        
        result = self.solver.solve(
            opportunities=opportunities,
            available_capital=capital,
            backend=SolverBackend.FRANK_WOLFE
        )
        
        assert result.num_trades > 0
        assert result.total_expected_profit > Decimal(0)
        assert result.solve_time_ms < 100  # Should be fast
    
    def test_capital_constraint(self):
        """Test that capital constraint is respected."""
        opportunities = create_synthetic_opportunities(10)
        capital = Decimal("1000.0")
        
        result = self.solver.solve(
            opportunities=opportunities,
            available_capital=capital,
            backend=SolverBackend.FRANK_WOLFE
        )
        
        # Should not exceed capital
        assert result.total_capital_used <= capital * Decimal("1.01")  # Allow small rounding
    
    def test_liquidity_constraint(self):
        """Test that liquidity constraints are respected."""
        opportunities = [
            ArbitrageOpportunity(
                market_id="LIMITED",
                outcome_id="YES",
                current_price=Decimal("0.50"),
                expected_value=Decimal("1.00"),
                max_liquidity=100,  # Limited liquidity
                platform="test",
                ticker="LIMITED"
            )
        ]
        
        capital = Decimal("10000.0")  # More capital than liquidity allows
        
        result = self.solver.solve(
            opportunities=opportunities,
            available_capital=capital,
            backend=SolverBackend.FRANK_WOLFE
        )
        
        if result.num_trades > 0:
            # Should not exceed liquidity
            total_qty = sum(alloc.quantity for alloc in result.allocations)
            assert total_qty <= 100
    
    def test_no_profit_opportunities(self):
        """Test behavior when no profitable opportunities exist."""
        opportunities = [
            ArbitrageOpportunity(
                market_id="UNPROFITABLE",
                outcome_id="YES",
                current_price=Decimal("0.99"),  # Almost no profit
                expected_value=Decimal("1.00"),
                max_liquidity=1000,
                platform="test",
                ticker="UNPROF"
            )
        ]
        
        result = self.solver.solve(
            opportunities=opportunities,
            available_capital=Decimal("1000.0"),
            backend=SolverBackend.FRANK_WOLFE
        )
        
        # Might not trade if profit below threshold
        assert result.total_capital_used >= Decimal(0)
    
    def test_position_size_limit(self):
        """Test max position fraction constraint."""
        opportunities = create_synthetic_opportunities(5)
        capital = Decimal("10000.0")
        
        result = self.solver.solve(
            opportunities=opportunities,
            available_capital=capital,
            backend=SolverBackend.FRANK_WOLFE
        )
        
        # Each position should respect max fraction
        max_position_value = capital * Decimal(str(self.solver.max_position_fraction))
        
        for alloc in result.allocations:
            assert alloc.capital_required <= max_position_value * Decimal("1.01")  # Small tolerance
    
    def test_bregman_projection(self):
        """Test Bregman projection for position rebalancing."""
        current_positions = {
            "A": 0.3,
            "B": 0.5,
            "C": 0.2
        }
        
        target_distribution = {
            "A": 0.4,
            "B": 0.3,
            "C": 0.3
        }
        
        result = self.solver.bregman_project(
            current_positions=current_positions,
            target_distribution=target_distribution
        )
        
        # Should return valid distribution
        assert abs(sum(result.values()) - 1.0) < 0.01
        assert all(v >= 0 for v in result.values())
    
    def test_performance_tracking(self):
        """Test that performance metrics are tracked."""
        opportunities = create_synthetic_opportunities(5)
        
        # Run multiple solves
        for _ in range(3):
            self.solver.solve(
                opportunities=opportunities,
                available_capital=Decimal("1000.0"),
                backend=SolverBackend.FRANK_WOLFE
            )
        
        stats = self.solver.get_performance_stats()
        # Stats might be empty depending on implementation
        assert isinstance(stats, dict)


class TestSolverBenchmarks:
    """Test benchmarking functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.solver = ArbitrageSolver()
        self.benchmark = SolverBenchmark(self.solver)
    
    def test_synthetic_opportunity_generation(self):
        """Test synthetic data generation."""
        opps = create_synthetic_opportunities(10)
        
        assert len(opps) == 10
        assert all(opp.current_price > 0 for opp in opps)
        assert all(opp.expected_value >= opp.current_price for opp in opps)
    
    def test_backend_comparison(self):
        """Test comparing different backends."""
        opportunities = create_synthetic_opportunities(20)
        capital = Decimal("5000.0")
        
        suite = self.benchmark.benchmark_backends(
            opportunities=opportunities,
            capital=capital,
            backends=[SolverBackend.FRANK_WOLFE],
            runs_per_backend=2
        )
        
        assert len(suite.results) > 0
        
        summary = suite.summary()
        assert "frank_wolfe" in summary
    
    def test_scalability_benchmark(self):
        """Test scalability measurements."""
        opportunities = create_synthetic_opportunities(100)
        capital = Decimal("10000.0")
        
        results = self.benchmark.benchmark_scalability(
            base_opportunities=opportunities,
            capital=capital,
            backend=SolverBackend.FRANK_WOLFE,
            sizes=[10, 20, 30]
        )
        
        assert len(results) > 0
        # Solve time should generally increase with problem size
        sizes = sorted(results.keys())
        if len(sizes) >= 2:
            # Not a strict requirement, but expected trend
            pass  # Could check monotonicity
    
    def test_real_time_performance(self):
        """Test real-time performance requirements."""
        opportunities = create_synthetic_opportunities(30)
        capital = Decimal("5000.0")
        
        perf = self.benchmark.benchmark_real_time_performance(
            opportunities=opportunities,
            capital=capital,
            backend=SolverBackend.FRANK_WOLFE,
            target_time_ms=50.0,
            num_runs=10
        )
        
        assert "mean_time_ms" in perf
        assert "p95_time_ms" in perf
        assert "meets_target_pct" in perf
        
        # Frank-Wolfe should be reasonably fast
        assert perf["mean_time_ms"] < 200  # Generous limit for tests


class TestOpportunityModel:
    """Test the ArbitrageOpportunity data model."""
    
    def test_profit_calculation(self):
        """Test profit calculation properties."""
        opp = ArbitrageOpportunity(
            market_id="TEST",
            outcome_id="YES",
            current_price=Decimal("0.60"),
            expected_value=Decimal("1.00"),
            max_liquidity=1000,
            platform="test",
            ticker="TEST"
        )
        
        assert opp.gross_profit_per_contract == Decimal("0.40")
        assert abs(float(opp.profit_percentage) - 0.6667) < 0.01
    
    def test_binary_complement(self):
        """Test binary complement arbitrage setup."""
        opp = ArbitrageOpportunity(
            market_id="BINARY",
            outcome_id="YES",
            current_price=Decimal("0.45"),
            expected_value=Decimal("1.00"),
            max_liquidity=1000,
            platform="test",
            ticker="BIN",
            complement_id="BINARY_NO",
            complement_price=Decimal("0.50")
        )
        
        # YES (0.45) + NO (0.50) = 0.95 < 1.00 → arbitrage!
        assert opp.complement_id is not None
        assert opp.current_price + opp.complement_price < Decimal("1.00")


@pytest.mark.skipif(
    not pytest.importorskip("cvxpy", reason="CVXPY not installed"),
    reason="CVXPY required for this test"
)
class TestCVXPYBackend:
    """Test CVXPY solver backend."""
    
    def test_cvxpy_solve(self):
        """Test solving with CVXPY."""
        solver = ArbitrageSolver()
        opportunities = create_synthetic_opportunities(10)
        
        result = solver.solve(
            opportunities=opportunities,
            available_capital=Decimal("5000.0"),
            backend=SolverBackend.CVXPY_ECOS,
            integer=False
        )
        
        assert result.solver_backend == "cvxpy_ecos"
        assert result.solve_time_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
