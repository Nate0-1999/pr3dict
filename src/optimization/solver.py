"""
PR3DICT: Integer Programming Solver for Optimal Arbitrage Execution

Implements multiple optimization approaches:
1. Linear Programming (LP) - Continuous allocation
2. Integer Programming (IP) - Discrete contract allocation
3. Frank-Wolfe algorithm - Non-convex objectives
4. Bregman projection - Position rebalancing

Mathematical formulation: docs/optimization_formulation.md
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Literal
from decimal import Decimal
from enum import Enum
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports - fail gracefully if not available
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    logger.warning("CVXPY not installed. Install with: pip install cvxpy")

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    logger.info("Gurobi not available (commercial license required)")


class SolverBackend(Enum):
    """Available optimization backends."""
    CVXPY_ECOS = "cvxpy_ecos"      # Open source, good for LP
    CVXPY_OSQP = "cvxpy_osqp"      # Open source, good for QP
    CVXPY_SCIP = "cvxpy_scip"      # Open source, IP capable
    GUROBI = "gurobi"              # Commercial, fastest for IP
    FRANK_WOLFE = "frank_wolfe"    # Custom implementation


@dataclass
class ArbitrageOpportunity:
    """Represents a single arbitrage opportunity."""
    market_id: str
    outcome_id: str
    current_price: Decimal      # p_i: Current market price
    expected_value: Decimal     # v_i: Expected value at resolution
    max_liquidity: int          # L_i: Max contracts available
    platform: str
    ticker: str
    
    # Optional: for binary complement arb
    complement_id: Optional[str] = None
    complement_price: Optional[Decimal] = None
    
    @property
    def gross_profit_per_contract(self) -> Decimal:
        """Expected profit per contract before fees."""
        return self.expected_value - self.current_price
    
    @property
    def profit_percentage(self) -> Decimal:
        """Profit as percentage of investment."""
        if self.current_price == 0:
            return Decimal(0)
        return self.gross_profit_per_contract / self.current_price


@dataclass
class TradeAllocation:
    """Optimal trade allocation from solver."""
    market_id: str
    outcome_id: str
    quantity: int                # Number of contracts to buy
    capital_required: Decimal    # Total capital needed
    expected_profit: Decimal     # Net profit after fees
    price: Decimal               # Execution price
    platform: str


@dataclass
class OptimizationResult:
    """Result from optimization solver."""
    allocations: List[TradeAllocation]
    total_capital_used: Decimal
    total_expected_profit: Decimal
    solve_time_ms: float
    solver_backend: str
    solver_status: str
    objective_value: float
    
    # Performance metrics
    profit_percentage: float     # ROI
    capital_efficiency: float    # Profit per dollar invested
    num_trades: int


class ArbitrageSolver:
    """
    Optimization solver for arbitrage execution.
    
    Supports multiple backends and algorithms:
    - LP: Fast continuous allocation (CVXPY/Gurobi)
    - IP: Discrete contract allocation (Gurobi preferred)
    - Frank-Wolfe: Custom algorithm for non-convex cases
    """
    
    def __init__(self,
                 transaction_fee_rate: Decimal = Decimal("0.02"),  # 2% fee
                 gas_fee: Decimal = Decimal("0.0"),                # Fixed fee per trade
                 max_position_fraction: Decimal = Decimal("0.2"),  # 20% per market
                 min_profit_threshold: Decimal = Decimal("1.0")):  # $1 min profit
        """
        Initialize solver with cost parameters.
        
        Args:
            transaction_fee_rate: Percentage fee on trades (e.g., 0.02 = 2%)
            gas_fee: Fixed fee per trade execution
            max_position_fraction: Max % of capital per market
            min_profit_threshold: Minimum profit to consider opportunity
        """
        self.fee_rate = float(transaction_fee_rate)
        self.gas_fee = float(gas_fee)
        self.max_position_fraction = float(max_position_fraction)
        self.min_profit_threshold = float(min_profit_threshold)
        
        # Performance tracking
        self._solve_times: List[float] = []
        self._solution_quality: List[float] = []
    
    def solve(self,
              opportunities: List[ArbitrageOpportunity],
              available_capital: Decimal,
              backend: SolverBackend = SolverBackend.CVXPY_ECOS,
              integer: bool = False,
              time_limit_ms: int = 50) -> OptimizationResult:
        """
        Solve for optimal arbitrage allocation.
        
        Args:
            opportunities: List of available arbitrage opportunities
            available_capital: Total capital available to deploy
            backend: Which solver backend to use
            integer: True for integer programming (discrete contracts)
            time_limit_ms: Maximum solve time in milliseconds
        
        Returns:
            OptimizationResult with optimal allocations
        """
        start_time = time.perf_counter()
        
        # Filter low-profit opportunities
        filtered = [
            opp for opp in opportunities
            if opp.gross_profit_per_contract * opp.max_liquidity >= self.min_profit_threshold
        ]
        
        if not filtered:
            return self._empty_result("No profitable opportunities", start_time)
        
        logger.info(f"Optimizing {len(filtered)} opportunities with ${available_capital:.2f} capital")
        
        # Route to appropriate solver
        if backend == SolverBackend.FRANK_WOLFE:
            return self._solve_frank_wolfe(filtered, available_capital, time_limit_ms)
        elif backend == SolverBackend.GUROBI and HAS_GUROBI:
            return self._solve_gurobi(filtered, available_capital, integer, time_limit_ms)
        elif backend.name.startswith("CVXPY") and HAS_CVXPY:
            return self._solve_cvxpy(filtered, available_capital, integer, backend, time_limit_ms)
        else:
            # Fallback to Frank-Wolfe if requested backend unavailable
            logger.warning(f"Backend {backend} not available, using Frank-Wolfe")
            return self._solve_frank_wolfe(filtered, available_capital, time_limit_ms)
    
    def _solve_cvxpy(self,
                     opportunities: List[ArbitrageOpportunity],
                     capital: Decimal,
                     integer: bool,
                     backend: SolverBackend,
                     time_limit_ms: int) -> OptimizationResult:
        """Solve using CVXPY backend."""
        start_time = time.perf_counter()
        n = len(opportunities)
        
        # Decision variables
        if integer:
            x = cp.Variable(n, integer=True)  # Discrete contracts
        else:
            x = cp.Variable(n)  # Continuous allocation
        
        # Parameters
        prices = np.array([float(opp.current_price) for opp in opportunities])
        expected_values = np.array([float(opp.expected_value) for opp in opportunities])
        max_liquidity = np.array([float(opp.max_liquidity) for opp in opportunities])
        
        # Objective: maximize profit - transaction costs
        gross_profit = (expected_values - prices) @ x
        transaction_costs = self.fee_rate * (prices @ x)
        objective = cp.Maximize(gross_profit - transaction_costs)
        
        # Constraints
        constraints = [
            x >= 0,                              # Non-negative positions
            x <= max_liquidity,                  # Liquidity limits
            prices @ x <= float(capital),        # Capital constraint
        ]
        
        # Position size limits (max % of capital per market)
        max_per_position = float(capital) * self.max_position_fraction
        for i, opp in enumerate(opportunities):
            if prices[i] > 0:
                constraints.append(x[i] <= max_per_position / prices[i])
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        # Select solver based on backend
        solver_map = {
            SolverBackend.CVXPY_ECOS: cp.ECOS,
            SolverBackend.CVXPY_OSQP: cp.OSQP,
            SolverBackend.CVXPY_SCIP: cp.SCIP,
        }
        solver = solver_map.get(backend, cp.ECOS)
        
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception as e:
            logger.error(f"CVXPY solver failed: {e}")
            return self._empty_result(f"Solver error: {e}", start_time)
        
        solve_time = (time.perf_counter() - start_time) * 1000  # ms
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"Solver status: {problem.status}")
            return self._empty_result(f"No optimal solution: {problem.status}", start_time)
        
        # Extract allocations
        allocations = []
        total_capital_used = Decimal(0)
        total_profit = Decimal(0)
        
        for i, opp in enumerate(opportunities):
            qty = int(np.round(x.value[i])) if x.value is not None else 0
            if qty > 0:
                capital_req = Decimal(str(qty)) * opp.current_price
                gross = Decimal(str(qty)) * (opp.expected_value - opp.current_price)
                fees = capital_req * Decimal(str(self.fee_rate))
                net_profit = gross - fees
                
                allocations.append(TradeAllocation(
                    market_id=opp.market_id,
                    outcome_id=opp.outcome_id,
                    quantity=qty,
                    capital_required=capital_req,
                    expected_profit=net_profit,
                    price=opp.current_price,
                    platform=opp.platform
                ))
                
                total_capital_used += capital_req
                total_profit += net_profit
        
        return OptimizationResult(
            allocations=allocations,
            total_capital_used=total_capital_used,
            total_expected_profit=total_profit,
            solve_time_ms=solve_time,
            solver_backend=backend.value,
            solver_status=problem.status,
            objective_value=problem.value if problem.value else 0.0,
            profit_percentage=float(total_profit / total_capital_used * 100) if total_capital_used > 0 else 0.0,
            capital_efficiency=float(total_profit / capital) if capital > 0 else 0.0,
            num_trades=len(allocations)
        )
    
    def _solve_gurobi(self,
                      opportunities: List[ArbitrageOpportunity],
                      capital: Decimal,
                      integer: bool,
                      time_limit_ms: int) -> OptimizationResult:
        """Solve using Gurobi (commercial solver)."""
        start_time = time.perf_counter()
        
        try:
            model = gp.Model("arbitrage")
            model.setParam('OutputFlag', 0)  # Suppress output
            model.setParam('TimeLimit', time_limit_ms / 1000.0)  # Convert to seconds
            
            n = len(opportunities)
            
            # Decision variables
            if integer:
                x = model.addVars(n, vtype=GRB.INTEGER, lb=0, name="x")
            else:
                x = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="x")
            
            # Objective: maximize profit - fees
            objective = gp.quicksum(
                (float(opp.expected_value) - float(opp.current_price) - 
                 self.fee_rate * float(opp.current_price)) * x[i]
                for i, opp in enumerate(opportunities)
            )
            model.setObjective(objective, GRB.MAXIMIZE)
            
            # Constraints
            # 1. Capital constraint
            model.addConstr(
                gp.quicksum(float(opp.current_price) * x[i] 
                           for i, opp in enumerate(opportunities)) <= float(capital),
                "capital"
            )
            
            # 2. Liquidity constraints
            for i, opp in enumerate(opportunities):
                model.addConstr(x[i] <= opp.max_liquidity, f"liquidity_{i}")
            
            # 3. Position size limits
            max_per_position = float(capital) * self.max_position_fraction
            for i, opp in enumerate(opportunities):
                if float(opp.current_price) > 0:
                    model.addConstr(
                        x[i] <= max_per_position / float(opp.current_price),
                        f"position_limit_{i}"
                    )
            
            # Solve
            model.optimize()
            
            solve_time = (time.perf_counter() - start_time) * 1000  # ms
            
            if model.status != GRB.OPTIMAL:
                status_map = {
                    GRB.INFEASIBLE: "infeasible",
                    GRB.UNBOUNDED: "unbounded",
                    GRB.TIME_LIMIT: "time_limit",
                }
                status = status_map.get(model.status, f"status_{model.status}")
                return self._empty_result(f"Gurobi: {status}", start_time)
            
            # Extract solution
            allocations = []
            total_capital_used = Decimal(0)
            total_profit = Decimal(0)
            
            for i, opp in enumerate(opportunities):
                qty = int(round(x[i].X))
                if qty > 0:
                    capital_req = Decimal(str(qty)) * opp.current_price
                    gross = Decimal(str(qty)) * (opp.expected_value - opp.current_price)
                    fees = capital_req * Decimal(str(self.fee_rate))
                    net_profit = gross - fees
                    
                    allocations.append(TradeAllocation(
                        market_id=opp.market_id,
                        outcome_id=opp.outcome_id,
                        quantity=qty,
                        capital_required=capital_req,
                        expected_profit=net_profit,
                        price=opp.current_price,
                        platform=opp.platform
                    ))
                    
                    total_capital_used += capital_req
                    total_profit += net_profit
            
            return OptimizationResult(
                allocations=allocations,
                total_capital_used=total_capital_used,
                total_expected_profit=total_profit,
                solve_time_ms=solve_time,
                solver_backend="gurobi",
                solver_status="optimal",
                objective_value=model.ObjVal,
                profit_percentage=float(total_profit / total_capital_used * 100) if total_capital_used > 0 else 0.0,
                capital_efficiency=float(total_profit / capital) if capital > 0 else 0.0,
                num_trades=len(allocations)
            )
            
        except Exception as e:
            logger.error(f"Gurobi solver error: {e}")
            return self._empty_result(f"Gurobi error: {e}", start_time)
    
    def _solve_frank_wolfe(self,
                          opportunities: List[ArbitrageOpportunity],
                          capital: Decimal,
                          time_limit_ms: int,
                          max_iterations: int = 100,
                          tolerance: float = 1e-4) -> OptimizationResult:
        """
        Frank-Wolfe algorithm for non-convex optimization.
        
        Useful when:
        - Transaction costs create non-convexities
        - Market impact is non-linear
        - Need projection-free optimization
        """
        start_time = time.perf_counter()
        n = len(opportunities)
        
        # Initialize feasible point (proportional to profit)
        prices = np.array([float(opp.current_price) for opp in opportunities])
        expected_values = np.array([float(opp.expected_value) for opp in opportunities])
        max_liquidity = np.array([float(opp.max_liquidity) for opp in opportunities])
        
        profits = expected_values - prices
        x = np.zeros(n)
        
        # Initialize with greedy allocation
        cap = float(capital)
        for i in np.argsort(-profits):  # Descending profit
            if cap <= 0:
                break
            max_qty = min(max_liquidity[i], cap / prices[i] if prices[i] > 0 else 0)
            x[i] = max_qty
            cap -= x[i] * prices[i]
        
        def objective(x_vec):
            """Objective function: profit - transaction costs."""
            gross = np.dot(profits, x_vec)
            costs = self.fee_rate * np.dot(prices, x_vec)
            return gross - costs
        
        def gradient(x_vec):
            """Gradient of objective."""
            return profits - self.fee_rate * prices
        
        # Frank-Wolfe iterations
        for iteration in range(max_iterations):
            # Check time limit
            if (time.perf_counter() - start_time) * 1000 > time_limit_ms:
                logger.warning(f"Frank-Wolfe time limit reached at iteration {iteration}")
                break
            
            # Compute gradient
            grad = gradient(x)
            
            # Solve linear subproblem: min <grad, s>
            # This is equivalent to putting all weight on most negative gradient component
            s = np.zeros(n)
            
            # Greedy: allocate to outcome with best gradient-adjusted profit
            cap_remaining = float(capital)
            for i in np.argsort(-grad):  # Descending gradient
                if cap_remaining <= 0:
                    break
                max_qty = min(max_liquidity[i], cap_remaining / prices[i] if prices[i] > 0 else 0)
                s[i] = max_qty
                cap_remaining -= s[i] * prices[i]
            
            # Line search for step size
            # Try different step sizes and pick best
            best_alpha = 0.0
            best_obj = objective(x)
            
            for alpha in [0.1, 0.2, 0.5, 1.0]:
                x_new = x + alpha * (s - x)
                obj = objective(x_new)
                if obj > best_obj:
                    best_obj = obj
                    best_alpha = alpha
            
            # Update
            x_new = x + best_alpha * (s - x)
            
            # Check convergence
            if np.linalg.norm(x_new - x) < tolerance:
                logger.info(f"Frank-Wolfe converged at iteration {iteration}")
                break
            
            x = x_new
        
        solve_time = (time.perf_counter() - start_time) * 1000
        
        # Extract allocations
        allocations = []
        total_capital_used = Decimal(0)
        total_profit = Decimal(0)
        
        for i, opp in enumerate(opportunities):
            qty = int(np.round(x[i]))
            if qty > 0:
                capital_req = Decimal(str(qty)) * opp.current_price
                gross = Decimal(str(qty)) * (opp.expected_value - opp.current_price)
                fees = capital_req * Decimal(str(self.fee_rate))
                net_profit = gross - fees
                
                allocations.append(TradeAllocation(
                    market_id=opp.market_id,
                    outcome_id=opp.outcome_id,
                    quantity=qty,
                    capital_required=capital_req,
                    expected_profit=net_profit,
                    price=opp.current_price,
                    platform=opp.platform
                ))
                
                total_capital_used += capital_req
                total_profit += net_profit
        
        return OptimizationResult(
            allocations=allocations,
            total_capital_used=total_capital_used,
            total_expected_profit=total_profit,
            solve_time_ms=solve_time,
            solver_backend="frank_wolfe",
            solver_status="converged",
            objective_value=float(objective(x)),
            profit_percentage=float(total_profit / total_capital_used * 100) if total_capital_used > 0 else 0.0,
            capital_efficiency=float(total_profit / capital) if capital > 0 else 0.0,
            num_trades=len(allocations)
        )
    
    def bregman_project(self,
                       current_positions: Dict[str, float],
                       target_distribution: Dict[str, float],
                       constraints: Optional[List[Tuple[List[str], float]]] = None,
                       max_iterations: int = 50,
                       tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Bregman projection for position rebalancing.
        
        Finds optimal rebalancing that:
        - Minimizes KL divergence from current positions
        - Satisfies marginal polytope constraints
        - Maintains valid probability distribution
        
        Args:
            current_positions: Current position weights {outcome_id: weight}
            target_distribution: Target probability distribution
            constraints: List of (outcome_ids, total_weight) constraints
            max_iterations: Max projection iterations
            tolerance: Convergence tolerance
        
        Returns:
            Optimal position weights minimizing transaction costs
        """
        # Convert to numpy arrays
        outcomes = sorted(current_positions.keys())
        n = len(outcomes)
        
        x = np.array([current_positions[o] for o in outcomes])
        target = np.array([target_distribution.get(o, 0.0) for o in outcomes])
        
        # Normalize
        x = x / (np.sum(x) + 1e-10)
        target = target / (np.sum(target) + 1e-10)
        
        # Iterative Bregman projection
        for iteration in range(max_iterations):
            x_old = x.copy()
            
            # Project onto simplex (probability constraint)
            x = np.maximum(x, 1e-10)  # Avoid log(0)
            x = x / np.sum(x)
            
            # Apply custom constraints if provided
            if constraints:
                for outcome_ids, total_weight in constraints:
                    indices = [outcomes.index(oid) for oid in outcome_ids if oid in outcomes]
                    if indices:
                        current_sum = np.sum(x[indices])
                        if current_sum > 0:
                            # Scale to match constraint
                            scale = total_weight / current_sum
                            x[indices] *= scale
            
            # Gradient step toward target (KL divergence)
            # This is a simplified version; full algorithm uses Lagrange multipliers
            alpha = 0.1  # Step size
            x = x * np.exp(alpha * np.log(target + 1e-10) - alpha * np.log(x + 1e-10))
            x = x / np.sum(x)
            
            # Check convergence
            if np.linalg.norm(x - x_old) < tolerance:
                logger.debug(f"Bregman projection converged at iteration {iteration}")
                break
        
        return {outcomes[i]: float(x[i]) for i in range(n)}
    
    def _empty_result(self, reason: str, start_time: float) -> OptimizationResult:
        """Return empty result when optimization fails."""
        solve_time = (time.perf_counter() - start_time) * 1000
        logger.warning(f"Optimization failed: {reason}")
        
        return OptimizationResult(
            allocations=[],
            total_capital_used=Decimal(0),
            total_expected_profit=Decimal(0),
            solve_time_ms=solve_time,
            solver_backend="none",
            solver_status=reason,
            objective_value=0.0,
            profit_percentage=0.0,
            capital_efficiency=0.0,
            num_trades=0
        )
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get solver performance statistics."""
        if not self._solve_times:
            return {}
        
        return {
            "avg_solve_time_ms": np.mean(self._solve_times),
            "max_solve_time_ms": np.max(self._solve_times),
            "min_solve_time_ms": np.min(self._solve_times),
            "total_solves": len(self._solve_times),
            "avg_solution_quality": np.mean(self._solution_quality) if self._solution_quality else 0.0,
        }
