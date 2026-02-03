# Marginal Polytope Theory

**Date:** February 2, 2026  
**Source:** arXiv:1606.02825, arXiv:2508.03474  
**Difficulty:** Advanced

---

## Table of Contents

1. [Intuition](#intuition)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Formal Definition](#formal-definition)
4. [Why It Matters for Arbitrage](#why-it-matters-for-arbitrage)
5. [Worked Examples](#worked-examples)
6. [Computational Complexity](#computational-complexity)
7. [Practical Implementation](#practical-implementation)

---

## Intuition

### The Problem

Imagine you're at a prediction market with two related bets:
- **Market A:** "Will Trump win Pennsylvania?" (YES/NO)
- **Market B:** "Will Republicans win PA by 5+ points?" (YES/NO)

At first glance, these seem independent. You could have:
1. Market A = YES, Market B = YES ✓ Valid
2. Market A = YES, Market B = NO  ✓ Valid
3. Market A = NO, Market B = YES  ✗ **IMPOSSIBLE**
4. Market A = NO, Market B = NO   ✓ Valid

**The key insight:** If Market B = YES (Republicans win by 5+), then Market A = YES (Trump wins) **must be true**. This is a **logical dependency** that eliminates some outcome combinations.

### Naive vs Actual Outcome Space

**Naive counting:** 2 markets × 2 outcomes each = 4 possible combinations

**Actual valid outcomes:** Only 3 are logically possible

This gap creates **arbitrage opportunities**.

---

## Mathematical Foundation

### Outcome Space Ω

Let Ω be the set of all possible "states of the world" (complete outcomes).

**Example:** NCAA Basketball Tournament with 63 games
- Each game has 2 possible winners
- Naive outcome space: 2^63 = 9,223,372,036,854,775,808 combinations
- **But:** If Team A loses in Round 1, they can't win in Round 2!
- Actual valid outcomes: Far fewer (though still exponentially large)

### Indicator Functions φ

For each **condition** (tradable market outcome) i, define an indicator function:

```
φᵢ(ω) = {
  1  if condition i is true in outcome ω
  0  otherwise
}
```

**Example:** 
- Condition 1: "Duke wins their first game"
- Condition 2: "Duke wins championship"

```
If ω = "Duke loses first game":
  φ₁(ω) = 0
  φ₂(ω) = 0  (must be 0 because they can't win championship if they lost first game)
```

### Payoff Vector Space Z

Define the set of valid payoff vectors:

```
Z = {φ(ω) : ω ∈ Ω}
```

Where φ(ω) = [φ₁(ω), φ₂(ω), ..., φₙ(ω)] is the vector of all condition outcomes.

**Key property:** |Z| ≤ |Ω| because multiple world states may produce the same condition outcomes.

---

## Formal Definition

### The Marginal Polytope M

The **marginal polytope** is the **convex hull** of the valid payoff vectors:

```
M = conv(Z) = conv({φ(ω) : ω ∈ Ω})
```

In other words, M is the smallest convex set containing all valid payoff vectors.

### Mathematical Properties

1. **Convexity:** For any μ₁, μ₂ ∈ M and λ ∈ [0,1]:
   ```
   λμ₁ + (1-λ)μ₂ ∈ M
   ```

2. **Vertices:** The extreme points of M correspond to valid outcomes in Z

3. **Dimension:** M ⊆ ℝⁿ where n is the number of tradable conditions

4. **Constraints:** M can be represented as:
   ```
   M = {μ ∈ ℝⁿ : Aμ ≥ b, Σμᵢ = 1, μᵢ ≥ 0}
   ```
   Where A encodes the logical dependencies between conditions

### Arbitrage-Free Pricing Theorem

**Theorem:** A price vector θ = (θ₁, θ₂, ..., θₙ) is **arbitrage-free** if and only if θ ∈ M.

**Proof Sketch:**
1. If θ ∉ M, then there exists a betting strategy that guarantees profit regardless of outcome
2. If θ ∈ M, then θ is a convex combination of valid outcomes, so no guaranteed profit exists

---

## Why It Matters for Arbitrage

### Detection

If current market prices θ lie **outside** the marginal polytope M, arbitrage exists:

```
θ ∉ M  ⟹  Arbitrage opportunity exists
```

### Profit Calculation

The maximum extractable profit is the distance from θ to M:

```
Profit = D(μ*, θ)
```

Where:
- μ* is the projection of θ onto M (closest point in M)
- D is the Bregman divergence (distance metric)

---

## Worked Examples

### Example 1: Simple Two-Market Dependency

**Setup:**
- Market A: "Team wins first game" (YES/NO)
- Market B: "Team wins championship" (YES/NO)

**Dependency:** B = YES ⟹ A = YES (can't win championship without winning first game)

**Valid Outcomes:**
```
Ω = {
  ω₁: (A=NO, B=NO)  → φ(ω₁) = [0, 0]
  ω₂: (A=YES, B=NO) → φ(ω₂) = [1, 0]
  ω₃: (A=YES, B=YES) → φ(ω₃) = [1, 1]
}
```

**Marginal Polytope:**
```
Z = {[0,0], [1,0], [1,1]}
M = conv(Z) = convex hull of these 3 points
```

**Constraint Representation:**
```
M = {(μ₁, μ₂) : 
  μ₁ + μ₂ ≤ 1,  (can't both be fully likely)
  μ₂ ≤ μ₁,      (B implies A)
  μ₁, μ₂ ≥ 0
}
```

**Arbitrage Detection:**

Current prices: θ = (0.4, 0.7)
- Market A YES: $0.40
- Market B YES: $0.70

Check if θ ∈ M:
- θ₂ ≤ θ₁? → 0.7 ≤ 0.4? → **FALSE**

**Arbitrage exists!**

Strategy:
1. Sell Market B YES at $0.70
2. Buy Market A YES at $0.40
3. Guaranteed profit: $0.30 in scenario ω₃ (both YES)

---

### Example 2: Three-Team Tournament

**Setup:**
- Three teams: A, B, C
- Single-elimination: One team wins

**Conditions:**
1. Team A wins: φ₁
2. Team B wins: φ₂
3. Team C wins: φ₃

**Valid Outcomes:**
```
Ω = {ωₐ, ωᵦ, ωᴄ}
Z = {[1,0,0], [0,1,0], [0,0,1]}
```

**Marginal Polytope:**
```
M = {(μ₁, μ₂, μ₃) : 
  μ₁ + μ₂ + μ₃ = 1,
  μᵢ ≥ 0 for all i
}
```

This is a **probability simplex** in 3D.

**Arbitrage Check:**

Current prices: θ = (0.35, 0.40, 0.35) → sum = 1.10

Since Σθᵢ ≠ 1, θ ∉ M

**Arbitrage Strategy:**
Sell all three outcomes, collect $1.10, pay out $1.00 → **$0.10 guaranteed profit**

---

### Example 3: NCAA 2010 Tournament (Real-World Scale)

**Setup:**
- 63 games
- Each game: 2 possible winners
- Dependencies: Winner of Game 2 must be winner of either Game 0 or Game 1

**Naive Outcome Space:**
```
|Ω_naive| = 2^63 = 9,223,372,036,854,775,808
```

**Actual Valid Outcomes:**
```
|Ω_actual| = 2^63 (still huge, but constrained by tournament structure)
```

**Key Insight:** We don't enumerate! We use **linear constraints** instead.

**Constraint Example:**
```
φ₆₃(ω) ≤ φ₃₂(ω) + φ₃₃(ω)
```
(Championship winner must have won one of the final two games)

**Result from Paper:**
- Successfully computed arbitrage using Frank-Wolfe algorithm
- Converged in 50-150 iterations
- Each iteration: <30 seconds IP solve
- Total: tractable despite 2^63 outcome space!

---

## Computational Complexity

### Naive Approach: Enumerate All Outcomes

**Time Complexity:** O(2^n) where n = number of conditions

**Problem:** 
- 63 conditions → 9.2 quintillion checks
- Even at 1 billion checks/second: 292 years to complete

### Constraint-Based Approach: Integer Programming

**Key Theorem:** The marginal polytope M can be represented as:

```
M = conv({z ∈ {0,1}^n : A^T z ≥ b})
```

Where A and b encode the logical dependencies as **linear inequalities**.

**Time Complexity:** 
- Constraint generation: O(n²) for pairwise dependencies
- IP solve per iteration: O(n³) with commercial solvers (Gurobi)
- Frank-Wolfe iterations: O(k) where k ≈ 50-150

**Total Complexity:** O(k × n³) → Polynomial in practice!

### Real-World Performance (from arXiv:1606.02825)

**NCAA 2010 Tournament (63 games):**
- Constraint count: ~200 linear inequalities (not 2^63!)
- IP solve time: 1-30 seconds per iteration
- Total convergence: 50-150 iterations
- Total time: **Minutes instead of centuries**

---

## Practical Implementation

### Step 1: Define Valid Outcome Space

```python
# Example: Two dependent markets
def generate_valid_outcomes():
    """
    Market A: Trump wins PA
    Market B: Republicans win PA by 5+
    """
    return [
        [0, 0],  # A=NO, B=NO (valid)
        [1, 0],  # A=YES, B=NO (valid)
        [1, 1],  # A=YES, B=YES (valid)
        # [0, 1] is INVALID (B implies A)
    ]
```

### Step 2: Generate Linear Constraints

```python
import numpy as np

def generate_constraints():
    """
    Constraint: B=YES ⟹ A=YES
    Equivalent: μ_B ≤ μ_A
    Or: -μ_A + μ_B ≤ 0
    """
    # A matrix: -μ_A + μ_B ≤ 0
    A = np.array([
        [-1, 1],  # Dependency constraint
    ])
    b = np.array([0])
    
    # Additional constraints
    # μ_A + μ_B ≤ 1 (probabilities sum)
    # μ_A, μ_B ≥ 0 (non-negative)
    
    return A, b
```

### Step 3: Check Membership

```python
from scipy.optimize import linprog

def is_in_marginal_polytope(theta, A, b):
    """
    Check if price vector theta is in marginal polytope M
    """
    # Check if A^T θ ≥ b
    if not all(A.T @ theta >= b - 1e-6):  # Allow small numerical error
        return False
    
    # Check if Σθᵢ ≈ 1
    if abs(sum(theta) - 1.0) > 1e-6:
        return False
    
    # Check non-negativity
    if not all(theta >= -1e-6):
        return False
    
    return True
```

### Step 4: Detect Arbitrage

```python
def detect_arbitrage(market_a_price, market_b_price):
    """
    Detect arbitrage between dependent markets
    """
    theta = np.array([market_a_price, market_b_price])
    A, b = generate_constraints()
    
    if is_in_marginal_polytope(theta, A, b):
        return None  # No arbitrage
    
    # Arbitrage exists!
    return {
        "type": "dependency_violation",
        "current_prices": theta,
        "violation": "B implies A, but P(B) > P(A)"
    }
```

---

## Key Takeaways

1. **Marginal polytope M** = set of all arbitrage-free price vectors
2. **Checking θ ∈ M** = checking for arbitrage
3. **Exponential outcome space** → **Polynomial constraint representation**
4. **Linear inequalities** replace brute-force enumeration
5. **Real-world tractability** proven on 2^63 outcome spaces

---

## Next Reading

- [Bregman Projection](02-bregman-projection.md) - How to calculate optimal trades
- [Frank-Wolfe Algorithm](03-frank-wolfe.md) - How to build M iteratively
- [Integer Programming](04-integer-programming.md) - How to solve the constraints

---

## References

1. Dudík, M., et al. (2016). "Arbitrage-Free Combinatorial Market Making via Integer Programming." arXiv:1606.02825
2. Saguillo, O., et al. (2025). "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets." arXiv:2508.03474
3. Chen, Y., & Pennock, D. M. (2007). "A utility framework for bounded-loss market makers."
4. Abernethy, J., et al. (2013). "A collaborative mechanism for crowdsourcing prediction problems."

---

**Author:** PR3DICT Research Team  
**Last Updated:** February 2, 2026  
**Status:** Complete - Ready for implementation
