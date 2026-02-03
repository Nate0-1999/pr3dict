# Integer Programming for Arbitrage Detection

**Date:** February 2, 2026  
**Prerequisites:** [Marginal Polytope](01-marginal-polytope.md)  
**Difficulty:** Intermediate

---

## Table of Contents

1. [Intuition](#intuition)
2. [Problem Formulation](#problem-formulation)
3. [Constraint Generation](#constraint-generation)
4. [Solution Methods](#solution-methods)
5. [Worked Examples](#worked-examples)
6. [Computational Complexity](#computational-complexity)
7. [Practical Implementation](#practical-implementation)

---

## Intuition

### The Exponential Problem

**Question:** How do you check if prices are arbitrage-free when there are 2^63 possible outcomes?

**Naive approach:**
```python
for outcome in range(2**63):
    if violates_some_condition(outcome):
        continue
    check_price_consistency(outcome)
```

**Problem:** 2^63 = 9,223,372,036,854,775,808 iterations

At 1 billion checks/second: **292 years**

---

### The Integer Programming Solution

**Key insight:** Valid outcomes satisfy **logical constraints** that can be expressed as **linear inequalities**.

**Example:** NCAA Tournament
- Instead of checking 2^63 outcomes
- Write 200 linear constraints
- Let an IP solver find valid outcomes efficiently

**Time:** **1-30 seconds** instead of 292 years!

---

### How It Works

1. **Encode** market dependencies as linear constraints
2. **Formulate** arbitrage detection as integer linear program
3. **Solve** using modern solvers (Gurobi, CPLEX, CBC)
4. **Extract** arbitrage opportunities from solution

---

## Problem Formulation

### Decision Variables

For each tradable condition i, define binary variable:

```
zᵢ ∈ {0, 1}
```

Where:
- zᵢ = 1 if condition i is TRUE in some outcome
- zᵢ = 0 if condition i is FALSE

**Example:** Duke basketball
- z₁ = 1: "Duke wins first game"
- z₂ = 1: "Duke wins championship"

---

### Constraints Encoding Dependencies

**Logical dependency:** "If Duke wins championship, they must have won first game"

**Mathematical form:**
```
z₂ ≤ z₁
```

Or equivalently:
```
-z₁ + z₂ ≤ 0
```

**General form:**
```
A^T z ≥ b
```

Where A encodes all logical dependencies.

---

### Valid Outcome Space

The set of valid payoff vectors:

```
Z = {z ∈ {0,1}^n : A^T z ≥ b}
```

**This is the finite set of vertices of the marginal polytope M!**

---

### Arbitrage Detection as IP

**Problem:** Find outcome z that maximizes profit given current prices θ.

**Formulation:**
```
maximize    θ · z
subject to  A^T z ≥ b
            z ∈ {0,1}^n
```

**Interpretation:**
- θ · z = payout if outcome z occurs, given prices θ
- If max(θ · z) > 1: **Arbitrage exists!** (prices sum to > $1)
- If max(θ · z) < 1: **Reverse arbitrage!** (prices sum to < $1)

---

## Constraint Generation

### Types of Constraints

1. **Implication constraints:** A ⟹ B
2. **Mutual exclusivity:** A and B can't both be true
3. **Exhaustiveness:** At least one of {A, B, C} is true
4. **Structural constraints:** Tournament brackets, etc.

---

### Encoding Logical Implications

**Rule:** A ⟹ B

**Meaning:** If A is true, then B must be true

**IP constraint:**
```
zₐ ≤ zᵦ
```

Or:
```
zₐ - zᵦ ≤ 0
```

**Proof:**
- If zₐ = 1 (A is true), then constraint forces zᵦ ≥ 1, so zᵦ = 1 (B is true) ✓
- If zₐ = 0 (A is false), then constraint is 0 - zᵦ ≤ 0, so zᵦ ≥ 0 (always satisfied)

---

### Encoding Mutual Exclusivity

**Rule:** A and B are mutually exclusive (exactly one is true)

**IP constraint:**
```
zₐ + zᵦ = 1
```

**For multiple mutually exclusive outcomes {A, B, C}:**
```
zₐ + zᵦ + zᴄ = 1
```

---

### Encoding Exhaustiveness

**Rule:** At least one of {A, B, C} must be true

**IP constraint:**
```
zₐ + zᵦ + zᴄ ≥ 1
```

---

### Encoding Disjunction

**Rule:** (A and B) or (C and D)

**Approach:** Introduce auxiliary binary variable y:

```
zₐ ≥ y
zᵦ ≥ y
zᴄ ≥ 1 - y
zᴅ ≥ 1 - y
```

**Interpretation:**
- If y = 1: zₐ ≥ 1 and zᵦ ≥ 1, so A and B are true
- If y = 0: zᴄ ≥ 1 and zᴅ ≥ 1, so C and D are true

---

## Worked Examples

### Example 1: Two Dependent Markets

**Setup:**
- Market A: "Trump wins PA"
- Market B: "Republicans win PA by 5+ points"
- Dependency: B ⟹ A

**Variables:**
```
z₁ = Trump wins PA
z₂ = Republicans win by 5+
```

**Constraints:**
```
1. Dependency: z₂ ≤ z₁  (if z₂=1 then z₁=1)
2. Binary: z₁, z₂ ∈ {0,1}
```

**Valid outcomes:**
```
(z₁=0, z₂=0): Neither true ✓
(z₁=1, z₂=0): Trump wins, but not by 5+ ✓
(z₁=1, z₂=1): Both true ✓
(z₁=0, z₂=1): z₂ ≤ z₁ violated ✗
```

**Arbitrage detection:**

Current prices: θ = (0.40, 0.70)

Solve:
```
maximize    0.40z₁ + 0.70z₂
subject to  z₂ ≤ z₁
            z₁, z₂ ∈ {0,1}
```

Try each valid outcome:
```
(0,0): 0.40(0) + 0.70(0) = 0
(1,0): 0.40(1) + 0.70(0) = 0.40
(1,1): 0.40(1) + 0.70(1) = 1.10  ← maximum!
```

**Result:** max = 1.10 > 1.00 → **Arbitrage exists!**

**Strategy:**
- Buy both at total cost $1.10
- If (1,1) occurs (probability > 0), payout is $2.00
- Actually, this is wrong analysis...

**Correct interpretation:**
- YES prices: θ_YES = (0.40, 0.70)
- NO prices: θ_NO = (0.60, 0.30)
- Check: θ_YES + θ_NO = (1.00, 1.00) ✓ (individually consistent)
- But: P(B=YES) = 0.70 > P(A=YES) = 0.40 violates B ⟹ A

**Corrected arbitrage:**
- Sell B YES at $0.70
- Buy A YES at $0.40
- Net credit: $0.30
- If (1,1) occurs: Pay $1 (B YES), receive $1 (A YES), net $0
- If (1,0) occurs: Pay $0 (B NO), receive $1 (A YES), net $1 → profit $0.30!
- If (0,0) occurs: Pay $0 (B NO), receive $0 (A NO), net $0 → profit $0.30!

So $0.30 guaranteed profit.

---

### Example 2: Single-Elimination Tournament

**Setup:**
- 4 teams: A, B, C, D
- Semifinal 1: A vs B → winner W1
- Semifinal 2: C vs D → winner W2
- Final: W1 vs W2 → champion

**Variables:**
```
z₁ = A wins championship
z₂ = B wins championship
z₃ = C wins championship
z₄ = D wins championship
z₅ = A wins semifinal 1
z₆ = B wins semifinal 1
z₇ = C wins semifinal 2
z₈ = D wins semifinal 2
```

**Constraints:**
```
1. Exactly one champion: z₁ + z₂ + z₃ + z₄ = 1
2. Exactly one semifinal 1 winner: z₅ + z₆ = 1
3. Exactly one semifinal 2 winner: z₇ + z₈ = 1
4. Championship implies semifinal: z₁ ≤ z₅, z₂ ≤ z₆, z₃ ≤ z₇, z₄ ≤ z₈
```

**Valid outcomes:** 4 (one per champion)

**Verification:**
```
A wins: (1,0,0,0,1,0,0,1) ✓
B wins: (0,1,0,0,0,1,0,1) ✓
C wins: (0,0,1,0,1,0,1,0) ✓
D wins: (0,0,0,1,0,1,0,1) ✓
```

**Arbitrage check:**

Prices: θ = (0.30, 0.25, 0.35, 0.20, ...)

Check: z₁ + z₂ + z₃ + z₄ = 0.30 + 0.25 + 0.35 + 0.20 = 1.10

Arbitrage exists! Sell all four championship outcomes, collect $1.10, pay $1.00.

---

### Example 3: NCAA 2010 Reduction

**Setup:**
- 64 teams, 63 games
- Naive variables: One per game outcome = 63 binaries
- Naive outcomes: 2^63

**Constraint insight:**

Instead of enumerating 2^63, write constraints like:

```
"Team X advances to Round 2" ≤ "Team X wins Round 1"
"Team X wins championship" ≤ "Team X in Final Four"
Etc.
```

**From arXiv:1606.02825:**
- **200 constraints** capture tournament structure
- **Gurobi solves this in 1-30 seconds**
- No need to enumerate outcomes!

---

## Solution Methods

### Branch and Bound

**Idea:** Recursively partition feasible region.

**Algorithm:**
```
1. Solve LP relaxation (replace z ∈ {0,1} with z ∈ [0,1])
2. If solution is integer, done!
3. Otherwise, pick fractional variable zᵢ
4. Branch:
   - Subproblem 1: Add constraint zᵢ = 0
   - Subproblem 2: Add constraint zᵢ = 1
5. Recursively solve each subproblem
6. Prune branches that can't improve best solution found
```

**Key optimizations:**
- **Cutting planes:** Add constraints that cut off fractional solutions
- **Heuristics:** Find good integer solutions quickly to prune aggressively
- **Presolve:** Simplify constraints before search

---

### Modern IP Solvers

**Commercial:**
- **Gurobi:** State-of-the-art, free academic license
- **CPLEX:** IBM's solver, also excellent
- **Xpress:** Another commercial option

**Open-source:**
- **CBC (COIN-OR):** Good, but slower than commercial
- **SCIP:** Academic solver, competitive
- **GLPK:** Basic, slower

**Recommendation:** Gurobi for research (free academic license), CBC for production (open-source).

---

### Solver Performance

**From arXiv:1606.02825 (Gurobi on NCAA 2010):**

Early tournament (many viable outcomes):
- IP solve: 10-30 seconds
- Most time in branch-and-bound exploration

Mid-tournament (30-40 games resolved):
- IP solve: 5-15 seconds
- Fewer branches to explore

Late tournament (50+ games resolved):
- IP solve: <5 seconds
- Very few viable outcomes left

**Key insight:** "Gets faster as outcomes resolve" - dynamic difficulty.

---

## Computational Complexity

### Theoretical Complexity

**Worst case:** Integer programming is NP-hard.

**Implication:** No polynomial-time algorithm guaranteed to solve all instances.

**But:** Real-world instances often have **structure** that makes them tractable!

---

### Practical Complexity

**For structured IPs (like prediction markets):**

**Branch-and-bound depth:** O(log n) average (not O(n) worst case)

**Per-node work:** O(n³) for LP relaxation

**Total:** O(n³ log n) **in practice** for structured problems

**Real-world performance:**
- 63 variables (NCAA): 1-30 seconds
- 200 constraints
- Modern hardware (2020s CPU)

---

### Comparison to Naive Enumeration

**Naive:** O(2^n) - exponential

**IP solver:** O(n³ log n) - polynomial in practice

**For n=63:**
- Naive: 2^63 ≈ 10^19 operations
- IP: ~200,000 operations
- **Speedup: 10^14 (100 trillion times faster!)**

---

## Practical Implementation

### Using Gurobi in Python

```python
import gurobipy as gp
from gurobipy import GRB

def solve_arbitrage_ip(prices, constraints):
    """
    Detect arbitrage using integer programming.
    
    Args:
        prices: Price vector θ (n,)
        constraints: List of (A, b) where A^T z >= b
    
    Returns:
        z_opt: Optimal outcome (or None if infeasible)
        obj_value: Maximum payout
        is_arbitrage: True if arbitrage exists
    """
    n = len(prices)
    
    # Create model
    model = gp.Model("arbitrage_detection")
    model.setParam('OutputFlag', 0)  # Quiet mode
    
    # Variables: z[i] ∈ {0,1}
    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    
    # Objective: maximize payout
    model.setObjective(
        gp.quicksum(prices[i] * z[i] for i in range(n)),
        GRB.MAXIMIZE
    )
    
    # Constraints
    for A, b in constraints:
        for j in range(len(b)):
            model.addConstr(
                gp.quicksum(A[j,i] * z[i] for i in range(n)) >= b[j]
            )
    
    # Solve
    model.optimize()
    
    # Check status
    if model.status != GRB.OPTIMAL:
        return None, None, False
    
    # Extract solution
    z_opt = [z[i].X for i in range(n)]
    obj_value = model.objVal
    
    # Check arbitrage
    # If buying all z[i]=1 outcomes costs < payout, arbitrage exists
    cost = sum(prices[i] for i in range(n) if z_opt[i] > 0.5)
    is_arbitrage = obj_value > 1.0  # For YES/NO markets
    
    return z_opt, obj_value, is_arbitrage


# Example usage
import numpy as np

# Two dependent markets
prices = np.array([0.40, 0.70])  # (Trump wins PA, Republicans +5)

# Constraint: z[1] <= z[0] (market 1 implies market 0)
# In standard form: -z[0] + z[1] <= 0, or A^T z >= b with A^T z = z[0] - z[1], b = 0
A = np.array([[1, -1]])  # coefficients for z[0] - z[1] >= 0
b = np.array([0])

constraints = [(A, b)]

z_opt, obj, is_arb = solve_arbitrage_ip(prices, constraints)

print(f"Optimal outcome: {z_opt}")
print(f"Payout: ${obj:.2f}")
print(f"Arbitrage: {is_arb}")
```

---

### Constraint Generation from Market Data

```python
def generate_implication_constraint(market_a_id, market_b_id, n_markets):
    """
    Generate constraint: market_b_id => market_a_id
    (If B is true, A must be true)
    
    Returns: (A, b) where A^T z >= b
    """
    A = np.zeros((1, n_markets))
    A[0, market_a_id] = 1   # z_a
    A[0, market_b_id] = -1  # -z_b
    b = np.array([0])       # z_a - z_b >= 0
    
    return (A, b)


def generate_mutual_exclusivity_constraint(market_ids, n_markets):
    """
    Generate constraint: exactly one of market_ids is true.
    
    Returns: (A, b) where A^T z = 1
    """
    A = np.zeros((1, n_markets))
    for i in market_ids:
        A[0, i] = 1
    b = np.array([1])
    
    # Note: For equation, need to add both >= and <= constraints
    return [(A, b), (-A, np.array([-1]))]


# Example: Tournament bracket
n = 8  # 8 markets

# Constraint 1: Championship implies semifinal
cons1 = generate_implication_constraint(0, 4, n)  # Market 4 => Market 0

# Constraint 2: Exactly one champion
cons2 = generate_mutual_exclusivity_constraint([0, 1, 2, 3], n)

all_constraints = [cons1] + cons2
```

---

### Handling Large-Scale Problems

**Lazy constraint generation:**
- Don't generate all constraints upfront
- Start with subset
- Add violated constraints dynamically during solve

**Example:**
```python
def lazy_constraint_callback(model, where):
    """
    Add constraints on-the-fly during branch-and-bound.
    """
    if where == GRB.Callback.MIPSOL:
        # Get current solution
        z_val = model.cbGetSolution(model._vars)
        
        # Check if any logical rule is violated
        if check_violation(z_val):
            # Add constraint to cut off this solution
            expr = generate_cutting_constraint(z_val)
            model.cbLazy(expr)

# Use in model
model._vars = z
model.optimize(lazy_constraint_callback)
```

---

## Key Takeaways

1. **Integer programming reduces 2^n to polynomial time** for structured problems
2. **Logical dependencies → linear constraints** - the key encoding
3. **Modern solvers (Gurobi) are extremely efficient** - 1-30 seconds for 2^63 outcomes
4. **Constraint types:** Implication, exclusivity, exhaustiveness
5. **Real-world performance improves as outcomes resolve** - adaptive difficulty
6. **Free academic licenses available** - Gurobi, CPLEX
7. **IP is the oracle for Frank-Wolfe** - enables arbitrage-free market making

---

## Next Reading

- [Python Implementation](../implementations/integer_programming.py) - Full working code
- [Worked Examples](../examples/ip_examples.ipynb) - Interactive notebook
- [Frank-Wolfe Integration](03-frank-wolfe.md) - Using IP as oracle

---

## References

1. Dudík, M., et al. (2016). "Arbitrage-Free Combinatorial Market Making via Integer Programming."
2. Wolsey, L. A. (1998). "Integer Programming." Wiley.
3. Bertsimas, D., & Weismantel, R. (2005). "Optimization over Integers."
4. Gurobi Optimization. "Gurobi Optimizer Reference Manual."

---

**Author:** PR3DICT Research Team  
**Last Updated:** February 2, 2026  
**Status:** Complete - Ready for implementation
