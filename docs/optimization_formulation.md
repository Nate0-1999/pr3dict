# Mathematical Formulation for Arbitrage Optimization

## Overview

This document describes the mathematical optimization framework for optimal arbitrage execution in prediction markets. The formulation is inspired by the $40M trader's approach using combinatorial optimization rather than brute-force enumeration.

## 1. Problem Statement

Given a set of arbitrage opportunities across prediction markets, find the optimal allocation of capital to maximize profit while respecting:
- Capital constraints
- Position size limits
- Market liquidity constraints
- Transaction costs
- Risk limits

## 2. Variables

### Decision Variables

**Continuous Variables (Linear Programming):**
- `x_i` ∈ [0, ∞): Trade quantity for outcome `i` (in contracts)
- `c_i` ≥ 0: Capital allocated to trade `i`

**Integer Variables (Integer Programming):**
- `n_i` ∈ ℤ≥0: Number of contracts for outcome `i` (discrete shares)
- `z_i` ∈ {0,1}: Binary indicator if trade `i` is executed

### Parameters

- `p_i`: Current market price for outcome `i` ($/contract)
- `v_i`: Expected value at resolution for outcome `i`
- `L_i`: Maximum liquidity available for outcome `i`
- `C`: Total available capital
- `f_t`: Transaction fee rate (e.g., 0.02 for 2%)
- `f_g`: Gas/network fee (for blockchain-based markets)

## 3. Objective Function

### 3.1 Linear Programming (Continuous)

Maximize expected profit minus transaction costs:

```
maximize:  Σ_i [(v_i - p_i) * x_i - f_t * p_i * x_i - f_g * z_i]

where:
  - (v_i - p_i) * x_i: Gross profit from price differential
  - f_t * p_i * x_i: Proportional transaction costs
  - f_g * z_i: Fixed costs (gas fees, etc.)
```

### 3.2 Integer Programming (Discrete)

For discrete contract sizes:

```
maximize:  Σ_i [(v_i - p_i) * n_i - f_t * p_i * n_i] - Σ_i f_g * z_i

subject to:
  n_i ≥ 0, integer
  z_i ∈ {0, 1}
  n_i ≤ M * z_i  (M = large constant, forces z_i = 1 if n_i > 0)
```

## 4. Constraints

### 4.1 Capital Constraint

Total capital allocated cannot exceed available balance:

```
Σ_i p_i * x_i ≤ C

or for integer:
Σ_i p_i * n_i ≤ C
```

### 4.2 Liquidity Constraints

Cannot trade more than available market depth:

```
x_i ≤ L_i  ∀i

or:
n_i ≤ L_i  ∀i
```

### 4.3 Position Size Limits

Limit exposure to any single market (risk management):

```
x_i ≤ α * C / p_i  ∀i

where α is the max position fraction (e.g., 0.2 for 20%)
```

### 4.4 Binary Complement Constraint (for same-event arbitrage)

For binary markets with YES/NO outcomes on the same event:

```
x_yes + x_no ≤ min(L_yes, L_no)

This ensures balanced execution for complement arbitrage.
```

### 4.5 Marginal Polytope Constraints

For multi-outcome markets where outcomes are mutually exclusive, ensure probabilities sum to 1:

```
Σ_j∈outcomes(m) x_j * p_j = K  (where K is total capital in market m)

This maintains valid probability distributions.
```

## 5. Frank-Wolfe Algorithm for Non-Convex Cases

When transaction costs create non-convexities or when dealing with market impact, use the Frank-Wolfe algorithm:

### Algorithm:
```
1. Initialize: x⁰ feasible
2. For t = 0, 1, 2, ...
   a. Compute gradient: ∇f(x^t)
   b. Solve linear subproblem:
      s^t = argmin_s <∇f(x^t), s>  subject to s ∈ feasible set
   c. Line search: find step size γ^t ∈ [0,1]
   d. Update: x^(t+1) = x^t + γ^t (s^t - x^t)
3. Terminate when ||x^(t+1) - x^t|| < ε
```

### Advantages:
- Handles non-convex objectives
- Maintains feasibility at each iteration
- Suitable for large-scale problems
- No need for projection step

### Application to Arbitrage:
The Frank-Wolfe algorithm is particularly useful when:
1. Market impact is non-linear (large trades move prices)
2. Transaction costs have fixed components
3. Need to balance multiple correlated arbitrage opportunities

## 6. Bregman Projection for Position Sizing

Use Bregman projection to find optimal position sizes that respect the marginal polytope:

### Definition:
```
Bregman divergence: D_φ(x, y) = φ(x) - φ(y) - <∇φ(y), x - y>

For entropy function φ(x) = Σ x_i log(x_i):
D_KL(x, y) = Σ x_i log(x_i / y_i)
```

### Projection:
```
x* = argmin_x D_φ(x, x⁰)  subject to Ax = b, x ≥ 0

where:
- x⁰: Current position
- A: Constraint matrix (marginal polytope)
- b: Target probabilities
```

### Iterative Algorithm:
```
1. Start with current position distribution x⁰
2. For each constraint:
   a. Compute KL divergence to target
   b. Project onto constraint hyperplane
   c. Update position weights
3. Normalize to maintain probability distribution
```

### Application:
When rebalancing multi-outcome positions, Bregman projection ensures:
- Minimal deviation from current positions (reduces transaction costs)
- Maintains valid probability distributions
- Efficient convergence (typically <10 iterations)

## 7. Multi-Stage Optimization

For complex arbitrage involving multiple markets:

### Stage 1: Opportunity Identification
```
Filter markets where:
  profit_i = (v_i - p_i) * L_i - costs_i > threshold
```

### Stage 2: Portfolio Optimization
```
Solve IP/LP with all filtered opportunities
Consider correlations between markets
```

### Stage 3: Execution Optimization
```
For each selected trade:
  - Determine order splitting strategy
  - Optimize execution timing
  - Minimize market impact
```

## 8. Performance Requirements

### Computational Constraints:
- **Solve time**: < 50ms for real-time execution
- **Problem size**: Handle 100+ simultaneous opportunities
- **Update frequency**: Re-optimize every market update

### Optimization:
1. **Warm starting**: Use previous solution as initial point
2. **Incremental updates**: Only re-solve when opportunities change significantly
3. **Parallelization**: Solve independent market groups in parallel
4. **Approximation**: Use LP relaxation for quick estimates, IP for final allocation

## 9. Solver Backend Comparison

### CVXPY (Open Source)
- **Pros**: Free, Python-native, multiple backend support
- **Cons**: Slower for large IP problems (~100-500ms)
- **Best for**: LP, small-scale IP, prototyping

### Gurobi (Commercial)
- **Pros**: Fastest solver (~10-50ms), excellent IP performance, warm-start support
- **Cons**: Expensive (~$40k/year commercial, free for academic)
- **Best for**: Production systems, large-scale IP

### COIN-OR CBC (Open Source)
- **Pros**: Free, decent IP performance
- **Cons**: Slower than Gurobi (~50-200ms)
- **Best for**: Cost-sensitive deployments

### Recommendation:
Start with CVXPY + ECOS/OSQP for LP, evaluate Gurobi if solve times exceed requirements.

## 10. Implementation Notes

### Code Structure:
```python
class ArbitrageSolver:
    def solve_lp(opportunities, capital, backend='cvxpy')
    def solve_ip(opportunities, capital, backend='gurobi')
    def frank_wolfe(opportunities, capital, max_iters=100)
    def bregman_project(positions, constraints)
```

### Integration with Strategy:
```python
# In arbitrage.py
opportunities = await self.scan_markets(markets)
allocations = solver.solve_ip(opportunities, available_capital)
trades = await self.execute_allocations(allocations)
```

## References

1. Frank-Wolfe algorithm: Jaggi (2013) - "Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization"
2. Bregman projection: Bregman (1967) - "The relaxation method of finding the common point of convex sets"
3. Marginal polytope: Wainwright & Jordan (2008) - "Graphical Models, Exponential Families, and Variational Inference"
4. Integer Programming: Wolsey (1998) - "Integer Programming"

---

*Last Updated: 2026-02-02*
*Status: Mathematical foundation complete, ready for implementation*
