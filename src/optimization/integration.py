"""
PR3DICT: Optimization Integration with Trading Strategies

Integrates the optimization solver with arbitrage and market-making strategies.
Converts between strategy signals and optimization opportunities.
"""
from typing import List, Dict
from decimal import Decimal
import logging

from .solver import (
    ArbitrageSolver,
    ArbitrageOpportunity,
    OptimizationResult,
    SolverBackend
)
from ..strategies.base import Signal
from ..platforms.base import Market, OrderSide

logger = logging.getLogger(__name__)


class OpportunityConverter:
    """
    Converts between strategy signals and optimization opportunities.
    """
    
    @staticmethod
    def signal_to_opportunity(signal: Signal,
                             market: Market,
                             fee_rate: Decimal = Decimal("0.02")) -> ArbitrageOpportunity:
        """
        Convert a trading signal to an optimization opportunity.
        
        Args:
            signal: Trading signal from strategy
            market: Market data
            fee_rate: Transaction fee rate
        
        Returns:
            ArbitrageOpportunity for optimization
        """
        # Determine current price and expected value based on side
        if signal.side == OrderSide.YES:
            current_price = market.yes_price
            expected_value = Decimal("1.0")  # YES resolves to $1
        else:
            current_price = market.no_price
            expected_value = Decimal("1.0")  # NO resolves to $1
        
        # For binary complement arb, both sides can be included
        complement_id = None
        complement_price = None
        if market.arbitrage_opportunity:
            if signal.side == OrderSide.YES:
                complement_id = f"{market.id}_NO"
                complement_price = market.no_price
            else:
                complement_id = f"{market.id}_YES"
                complement_price = market.yes_price
        
        # Estimate liquidity (use market liquidity as proxy)
        max_liquidity = int(market.liquidity / current_price) if current_price > 0 else 0
        
        return ArbitrageOpportunity(
            market_id=market.id,
            outcome_id=f"{market.id}_{signal.side.value}",
            current_price=current_price,
            expected_value=expected_value,
            max_liquidity=max_liquidity,
            platform=market.platform,
            ticker=market.ticker,
            complement_id=complement_id,
            complement_price=complement_price
        )
    
    @staticmethod
    def signals_to_opportunities(signals: List[Signal],
                                markets: Dict[str, Market]) -> List[ArbitrageOpportunity]:
        """
        Convert multiple signals to opportunities.
        
        Args:
            signals: List of trading signals
            markets: Dictionary of market_id -> Market
        
        Returns:
            List of optimization opportunities
        """
        opportunities = []
        
        for signal in signals:
            market = markets.get(signal.market_id)
            if not market:
                logger.warning(f"Market {signal.market_id} not found for signal")
                continue
            
            try:
                opp = OpportunityConverter.signal_to_opportunity(signal, market)
                opportunities.append(opp)
            except Exception as e:
                logger.error(f"Failed to convert signal to opportunity: {e}")
                continue
        
        return opportunities


class OptimizedArbitrageExecutor:
    """
    Executes arbitrage using optimization solver.
    
    Workflow:
    1. Receive signals from strategy
    2. Convert to optimization opportunities
    3. Solve for optimal allocation
    4. Execute trades in parallel
    """
    
    def __init__(self,
                 solver: ArbitrageSolver,
                 backend: SolverBackend = SolverBackend.FRANK_WOLFE,
                 use_integer: bool = False,
                 parallel_execution: bool = True):
        """
        Initialize executor.
        
        Args:
            solver: Arbitrage solver instance
            backend: Preferred solver backend
            use_integer: Use integer programming for discrete contracts
            parallel_execution: Execute trades in parallel
        """
        self.solver = solver
        self.backend = backend
        self.use_integer = use_integer
        self.parallel_execution = parallel_execution
        
        self._execution_history: List[OptimizationResult] = []
    
    async def optimize_and_execute(self,
                                   signals: List[Signal],
                                   markets: Dict[str, Market],
                                   available_capital: Decimal) -> OptimizationResult:
        """
        Optimize trade allocation and execute.
        
        Args:
            signals: Trading signals from strategy
            markets: Market data dictionary
            available_capital: Available capital for trading
        
        Returns:
            OptimizationResult with execution details
        """
        # Convert signals to opportunities
        opportunities = OpportunityConverter.signals_to_opportunities(signals, markets)
        
        if not opportunities:
            logger.warning("No valid opportunities to optimize")
            return self._empty_result()
        
        # Solve optimization problem
        logger.info(f"Optimizing {len(opportunities)} opportunities with ${available_capital}")
        result = self.solver.solve(
            opportunities=opportunities,
            available_capital=available_capital,
            backend=self.backend,
            integer=self.use_integer,
            time_limit_ms=50  # Real-time requirement
        )
        
        # Log results
        logger.info(
            f"Optimization complete: {result.num_trades} trades, "
            f"${result.total_expected_profit:.2f} expected profit, "
            f"{result.solve_time_ms:.2f}ms solve time"
        )
        
        # Store for analytics
        self._execution_history.append(result)
        
        # TODO: Execute trades via platform interface
        # This would integrate with the execution engine
        # For now, return the optimization result
        
        return result
    
    def get_execution_stats(self) -> Dict[str, float]:
        """Get statistics on optimization/execution performance."""
        if not self._execution_history:
            return {}
        
        return {
            "total_optimizations": len(self._execution_history),
            "avg_solve_time_ms": sum(r.solve_time_ms for r in self._execution_history) / len(self._execution_history),
            "total_expected_profit": float(sum(r.total_expected_profit for r in self._execution_history)),
            "total_capital_deployed": float(sum(r.total_capital_used for r in self._execution_history)),
            "avg_trades_per_optimization": sum(r.num_trades for r in self._execution_history) / len(self._execution_history),
            "avg_roi_pct": sum(r.profit_percentage for r in self._execution_history) / len(self._execution_history),
        }
    
    def _empty_result(self) -> OptimizationResult:
        """Return empty result when no opportunities available."""
        return OptimizationResult(
            allocations=[],
            total_capital_used=Decimal(0),
            total_expected_profit=Decimal(0),
            solve_time_ms=0.0,
            solver_backend="none",
            solver_status="no_opportunities",
            objective_value=0.0,
            profit_percentage=0.0,
            capital_efficiency=0.0,
            num_trades=0
        )


class PortfolioOptimizer:
    """
    Portfolio-level optimization for multi-market arbitrage.
    
    Considers:
    - Correlation between markets
    - Risk diversification
    - Capital allocation across strategies
    """
    
    def __init__(self, solver: ArbitrageSolver):
        self.solver = solver
    
    def optimize_portfolio(self,
                          arbitrage_opportunities: List[ArbitrageOpportunity],
                          capital: Decimal,
                          max_positions: int = 10,
                          max_correlation: float = 0.7) -> OptimizationResult:
        """
        Optimize portfolio considering correlations and diversification.
        
        Args:
            arbitrage_opportunities: Available opportunities
            capital: Total capital to allocate
            max_positions: Maximum number of simultaneous positions
            max_correlation: Maximum correlation between positions
        
        Returns:
            Optimized portfolio allocation
        """
        # TODO: Implement correlation-aware optimization
        # This would require:
        # 1. Market correlation matrix
        # 2. Risk-adjusted objective function
        # 3. Diversification constraints
        
        # For now, use standard optimization
        result = self.solver.solve(
            opportunities=arbitrage_opportunities,
            available_capital=capital,
            backend=SolverBackend.FRANK_WOLFE
        )
        
        # Filter to max positions
        if len(result.allocations) > max_positions:
            # Keep highest profit trades
            sorted_allocs = sorted(
                result.allocations,
                key=lambda x: x.expected_profit,
                reverse=True
            )
            result.allocations = sorted_allocs[:max_positions]
            result.num_trades = len(result.allocations)
            
            # Recalculate totals
            result.total_capital_used = sum(a.capital_required for a in result.allocations)
            result.total_expected_profit = sum(a.expected_profit for a in result.allocations)
        
        return result
