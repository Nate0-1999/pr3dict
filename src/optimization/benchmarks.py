"""
PR3DICT: Optimization Solver Benchmarking

Compares performance across different solver backends:
- Solution quality (objective value)
- Solve time
- Scalability (problem size vs. time)
- Accuracy (LP relaxation vs. IP solution)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from decimal import Decimal
import time
import logging
import numpy as np
from collections import defaultdict

from .solver import (
    ArbitrageSolver,
    ArbitrageOpportunity,
    SolverBackend,
    OptimizationResult
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    backend: str
    problem_size: int           # Number of opportunities
    solve_time_ms: float
    objective_value: float
    num_trades: int
    capital_used: Decimal
    expected_profit: Decimal
    solver_status: str
    
    # Quality metrics
    optimality_gap: Optional[float] = None  # vs. best known solution
    lp_relaxation_gap: Optional[float] = None  # IP vs LP bound


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Summarize results by backend."""
        by_backend = defaultdict(list)
        
        for r in self.results:
            by_backend[r.backend].append(r)
        
        summary = {}
        for backend, results in by_backend.items():
            summary[backend] = {
                "avg_solve_time_ms": np.mean([r.solve_time_ms for r in results]),
                "max_solve_time_ms": np.max([r.solve_time_ms for r in results]),
                "min_solve_time_ms": np.min([r.solve_time_ms for r in results]),
                "avg_objective": np.mean([r.objective_value for r in results]),
                "total_runs": len(results),
                "success_rate": len([r for r in results if r.solver_status == "optimal"]) / len(results),
            }
        
        return summary
    
    def compare_backends(self) -> str:
        """Generate comparison report."""
        summary = self.summary()
        
        report = ["=" * 80]
        report.append("SOLVER BACKEND COMPARISON")
        report.append("=" * 80)
        
        for backend, stats in sorted(summary.items()):
            report.append(f"\n{backend.upper()}:")
            report.append(f"  Avg Solve Time: {stats['avg_solve_time_ms']:.2f}ms")
            report.append(f"  Max Solve Time: {stats['max_solve_time_ms']:.2f}ms")
            report.append(f"  Avg Objective:  ${stats['avg_objective']:.2f}")
            report.append(f"  Success Rate:   {stats['success_rate']:.1%}")
            report.append(f"  Total Runs:     {stats['total_runs']}")
        
        # Find fastest
        fastest = min(summary.items(), key=lambda x: x[1]['avg_solve_time_ms'])
        report.append(f"\n🏆 Fastest Backend: {fastest[0]} ({fastest[1]['avg_solve_time_ms']:.2f}ms avg)")
        
        # Find best quality
        best_quality = max(summary.items(), key=lambda x: x[1]['avg_objective'])
        report.append(f"💎 Best Quality: {best_quality[0]} (${best_quality[1]['avg_objective']:.2f} avg)")
        
        report.append("=" * 80)
        
        return "\n".join(report)


class SolverBenchmark:
    """
    Benchmark different solver backends and algorithms.
    """
    
    def __init__(self, solver: ArbitrageSolver):
        self.solver = solver
        self.suite = BenchmarkSuite()
    
    def benchmark_backends(self,
                          opportunities: List[ArbitrageOpportunity],
                          capital: Decimal,
                          backends: Optional[List[SolverBackend]] = None,
                          runs_per_backend: int = 10) -> BenchmarkSuite:
        """
        Compare different solver backends on the same problem.
        
        Args:
            opportunities: List of arbitrage opportunities
            capital: Available capital
            backends: List of backends to test (None = all available)
            runs_per_backend: Number of runs per backend for averaging
        
        Returns:
            BenchmarkSuite with results
        """
        if backends is None:
            backends = [
                SolverBackend.FRANK_WOLFE,  # Always available
                SolverBackend.CVXPY_ECOS,
                SolverBackend.CVXPY_OSQP,
                SolverBackend.GUROBI,
            ]
        
        logger.info(f"Benchmarking {len(backends)} backends with {len(opportunities)} opportunities")
        
        for backend in backends:
            for run in range(runs_per_backend):
                try:
                    result = self.solver.solve(
                        opportunities=opportunities,
                        available_capital=capital,
                        backend=backend,
                        integer=False,  # LP for fair comparison
                        time_limit_ms=1000  # 1 second max
                    )
                    
                    bench_result = BenchmarkResult(
                        backend=backend.value,
                        problem_size=len(opportunities),
                        solve_time_ms=result.solve_time_ms,
                        objective_value=result.objective_value,
                        num_trades=result.num_trades,
                        capital_used=result.total_capital_used,
                        expected_profit=result.total_expected_profit,
                        solver_status=result.solver_status
                    )
                    
                    self.suite.add_result(bench_result)
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {backend.value}: {e}")
                    continue
        
        return self.suite
    
    def benchmark_integer_gap(self,
                             opportunities: List[ArbitrageOpportunity],
                             capital: Decimal,
                             backend: SolverBackend = SolverBackend.CVXPY_ECOS) -> Dict[str, float]:
        """
        Measure the integrality gap (difference between LP relaxation and IP solution).
        
        The gap indicates how much discretization costs vs. continuous allocation.
        
        Returns:
            Dictionary with gap metrics
        """
        # Solve LP relaxation (continuous)
        lp_result = self.solver.solve(
            opportunities=opportunities,
            available_capital=capital,
            backend=backend,
            integer=False
        )
        
        # Solve IP (integer)
        ip_result = self.solver.solve(
            opportunities=opportunities,
            available_capital=capital,
            backend=backend,
            integer=True
        )
        
        if lp_result.objective_value == 0:
            return {"error": "LP failed"}
        
        gap = (lp_result.objective_value - ip_result.objective_value) / lp_result.objective_value
        
        return {
            "lp_objective": lp_result.objective_value,
            "ip_objective": ip_result.objective_value,
            "absolute_gap": lp_result.objective_value - ip_result.objective_value,
            "relative_gap_pct": gap * 100,
            "lp_solve_time_ms": lp_result.solve_time_ms,
            "ip_solve_time_ms": ip_result.solve_time_ms,
            "time_ratio": ip_result.solve_time_ms / lp_result.solve_time_ms if lp_result.solve_time_ms > 0 else 0,
        }
    
    def benchmark_scalability(self,
                             base_opportunities: List[ArbitrageOpportunity],
                             capital: Decimal,
                             backend: SolverBackend = SolverBackend.FRANK_WOLFE,
                             sizes: List[int] = [10, 25, 50, 100, 200]) -> Dict[int, float]:
        """
        Test how solve time scales with problem size.
        
        Args:
            base_opportunities: Pool of opportunities to sample from
            capital: Available capital
            backend: Backend to test
            sizes: Problem sizes to test
        
        Returns:
            Dictionary mapping size -> avg solve time (ms)
        """
        results = {}
        
        for size in sizes:
            if size > len(base_opportunities):
                logger.warning(f"Skipping size {size} (only {len(base_opportunities)} opportunities available)")
                continue
            
            # Sample opportunities
            sample = base_opportunities[:size]
            
            # Run multiple times and average
            times = []
            for _ in range(5):
                result = self.solver.solve(
                    opportunities=sample,
                    available_capital=capital,
                    backend=backend,
                    integer=False
                )
                times.append(result.solve_time_ms)
            
            results[size] = np.mean(times)
            logger.info(f"Size {size}: {results[size]:.2f}ms avg")
        
        return results
    
    def benchmark_real_time_performance(self,
                                       opportunities: List[ArbitrageOpportunity],
                                       capital: Decimal,
                                       backend: SolverBackend,
                                       target_time_ms: float = 50.0,
                                       num_runs: int = 100) -> Dict[str, any]:
        """
        Test if solver meets real-time requirements (<50ms).
        
        Returns:
            Performance metrics including percentiles
        """
        solve_times = []
        objectives = []
        
        for _ in range(num_runs):
            result = self.solver.solve(
                opportunities=opportunities,
                available_capital=capital,
                backend=backend,
                integer=False,
                time_limit_ms=int(target_time_ms)
            )
            
            solve_times.append(result.solve_time_ms)
            objectives.append(result.objective_value)
        
        times_array = np.array(solve_times)
        
        return {
            "mean_time_ms": np.mean(times_array),
            "median_time_ms": np.median(times_array),
            "p95_time_ms": np.percentile(times_array, 95),
            "p99_time_ms": np.percentile(times_array, 99),
            "max_time_ms": np.max(times_array),
            "meets_target_pct": np.sum(times_array <= target_time_ms) / len(times_array) * 100,
            "mean_objective": np.mean(objectives),
            "std_objective": np.std(objectives),
            "total_runs": num_runs,
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        return self.suite.compare_backends()


def create_synthetic_opportunities(n: int,
                                   price_range: tuple = (0.1, 0.9),
                                   spread_range: tuple = (0.02, 0.10),
                                   liquidity_range: tuple = (100, 10000)) -> List[ArbitrageOpportunity]:
    """
    Create synthetic arbitrage opportunities for benchmarking.
    
    Args:
        n: Number of opportunities
        price_range: Min/max current price
        spread_range: Min/max profit spread
        liquidity_range: Min/max liquidity
    
    Returns:
        List of synthetic opportunities
    """
    np.random.seed(42)  # Reproducible
    
    opportunities = []
    
    for i in range(n):
        current_price = Decimal(str(np.random.uniform(*price_range)))
        spread = Decimal(str(np.random.uniform(*spread_range)))
        expected_value = current_price + spread
        liquidity = int(np.random.uniform(*liquidity_range))
        
        # Ensure valid prices
        if expected_value > Decimal("1.0"):
            expected_value = Decimal("1.0")
        
        opportunities.append(ArbitrageOpportunity(
            market_id=f"SYNTH_{i}",
            outcome_id=f"YES_{i}",
            current_price=current_price,
            expected_value=expected_value,
            max_liquidity=liquidity,
            platform="synthetic",
            ticker=f"SYNTH{i}"
        ))
    
    return opportunities


if __name__ == "__main__":
    """Run benchmarks if executed directly."""
    logging.basicConfig(level=logging.INFO)
    
    # Create solver
    solver = ArbitrageSolver(
        transaction_fee_rate=Decimal("0.02"),
        min_profit_threshold=Decimal("5.0")
    )
    
    # Create benchmark suite
    benchmark = SolverBenchmark(solver)
    
    # Generate synthetic data
    opportunities = create_synthetic_opportunities(50)
    capital = Decimal("10000.0")
    
    print("\n" + "=" * 80)
    print("PR3DICT OPTIMIZATION SOLVER BENCHMARKS")
    print("=" * 80)
    
    # 1. Backend comparison
    print("\n[1] Comparing solver backends...")
    suite = benchmark.benchmark_backends(
        opportunities=opportunities,
        capital=capital,
        backends=[SolverBackend.FRANK_WOLFE, SolverBackend.CVXPY_ECOS],
        runs_per_backend=5
    )
    print(suite.compare_backends())
    
    # 2. Integer gap
    print("\n[2] Testing LP vs IP gap...")
    gap_results = benchmark.benchmark_integer_gap(
        opportunities=opportunities[:20],  # Smaller for IP
        capital=capital
    )
    print(f"LP Objective:        ${gap_results.get('lp_objective', 0):.2f}")
    print(f"IP Objective:        ${gap_results.get('ip_objective', 0):.2f}")
    print(f"Integrality Gap:     {gap_results.get('relative_gap_pct', 0):.2f}%")
    print(f"Time Ratio (IP/LP):  {gap_results.get('time_ratio', 0):.2f}x")
    
    # 3. Scalability
    print("\n[3] Testing scalability...")
    scale_results = benchmark.benchmark_scalability(
        base_opportunities=opportunities,
        capital=capital,
        backend=SolverBackend.FRANK_WOLFE,
        sizes=[10, 25, 50]
    )
    for size, time_ms in scale_results.items():
        print(f"  Size {size:3d}: {time_ms:6.2f}ms")
    
    # 4. Real-time performance
    print("\n[4] Testing real-time performance (target: <50ms)...")
    rt_results = benchmark.benchmark_real_time_performance(
        opportunities=opportunities[:30],
        capital=capital,
        backend=SolverBackend.FRANK_WOLFE,
        num_runs=20
    )
    print(f"  Mean:   {rt_results['mean_time_ms']:.2f}ms")
    print(f"  Median: {rt_results['median_time_ms']:.2f}ms")
    print(f"  P95:    {rt_results['p95_time_ms']:.2f}ms")
    print(f"  P99:    {rt_results['p99_time_ms']:.2f}ms")
    print(f"  Success Rate: {rt_results['meets_target_pct']:.1f}% under 50ms")
    
    print("\n" + "=" * 80)
    print("✅ Benchmarking complete!")
    print("=" * 80)
