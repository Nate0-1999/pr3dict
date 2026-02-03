"""
Frank-Wolfe Algorithm Implementation

Iterative Bregman projection onto marginal polytope for arbitrage detection
in combinatorial prediction markets.

Author: PR3DICT Research Team
Date: February 2, 2026
References: arXiv:1606.02825 (NCAA 2010 implementation)
"""

import numpy as np
from typing import Tuple, Optional, Callable, List
import time
import warnings

# Try to import Gurobi (optional, for IP oracle)
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    warnings.warn("Gurobi not available. IP oracle will use brute-force enumeration.")


class FrankWolfeOptimizer:
    """
    Frank-Wolfe (Conditional Gradient) algorithm for Bregman projection
    onto the marginal polytope.
    
    Solves: min_{μ ∈ M} D_R(μ || θ)
    
    Where:
    - M = marginal polytope (convex hull of valid outcomes)
    - D_R = Bregman divergence (KL for LMSR markets)
    - θ = current market prices
    
    Key feature: Only requires linear optimization over M per iteration
    (integer program oracle), not full convex optimization.
    """
    
    def __init__(self, 
                 max_iterations: int = 150,
                 convergence_threshold: float = 1e-6,
                 epsilon: float = 1e-10):
        """
        Args:
            max_iterations: Maximum Frank-Wolfe iterations
            convergence_threshold: Stop when duality gap < this
            epsilon: Small value for numerical stability
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.epsilon = epsilon
    
    def kl_divergence(self, mu: np.ndarray, theta: np.ndarray) -> float:
        """KL(μ || θ) = Σᵢ μᵢ ln(μᵢ/θᵢ)"""
        mu = np.clip(mu, self.epsilon, 1.0)
        theta = np.clip(theta, self.epsilon, 1.0)
        return np.sum(mu * np.log(mu / theta))
    
    def gradient_kl(self, mu: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """∇_μ KL(μ || θ) = ln(μ/θ)"""
        mu = np.clip(mu, self.epsilon, 1.0)
        theta = np.clip(theta, self.epsilon, 1.0)
        return np.log(mu / theta)
    
    def line_search(self,
                   mu_k: np.ndarray,
                   z_k: np.ndarray,
                   theta: np.ndarray,
                   num_trials: int = 20) -> float:
        """
        Find optimal step size γ ∈ [0,1] by grid search.
        
        Minimizes: D((1-γ)μₖ + γzₖ || θ)
        
        Args:
            mu_k: Current iterate
            z_k: New vertex from IP oracle
            theta: Reference prices
            num_trials: Number of γ values to try
        
        Returns:
            Optimal step size γ
        """
        best_gamma = 0.0
        best_obj = float('inf')
        
        for gamma in np.linspace(0, 1, num_trials):
            mu_trial = (1 - gamma) * mu_k + gamma * z_k
            obj = self.kl_divergence(mu_trial, theta)
            
            if obj < best_obj:
                best_obj = obj
                best_gamma = gamma
        
        return best_gamma
    
    def optimize(self,
                theta: np.ndarray,
                ip_oracle: Callable[[np.ndarray], np.ndarray],
                initial_vertex: Optional[np.ndarray] = None,
                verbose: bool = True) -> dict:
        """
        Run Frank-Wolfe algorithm to project θ onto marginal polytope M.
        
        Args:
            theta: Current market prices (n,)
            ip_oracle: Function that solves:
                       z = argmin_{z ∈ Z} gradient · z
                       where Z = valid outcomes (vertices of M)
            initial_vertex: Starting vertex (if None, use uniform)
            verbose: Print progress
        
        Returns:
            Dict with:
                - mu_star: Projected prices
                - iterations: Number of iterations
                - gap_history: Duality gap at each iteration
                - objective_history: Objective value history
                - converged: Whether algorithm converged
                - total_time: Total execution time
        
        Example:
            >>> # Define IP oracle for simple 3-outcome market
            >>> def oracle(gradient):
            ...     # Return vertex that minimizes gradient · z
            ...     i = np.argmin(gradient)
            ...     z = np.zeros(3)
            ...     z[i] = 1.0
            ...     return z
            >>> 
            >>> optimizer = FrankWolfeOptimizer()
            >>> theta = np.array([0.35, 0.40, 0.35])
            >>> result = optimizer.optimize(theta, oracle, verbose=False)
            >>> result['converged']
            True
        """
        n = len(theta)
        start_time = time.time()
        
        # Initialize
        if initial_vertex is None:
            mu_k = np.ones(n) / n  # Uniform distribution
        else:
            mu_k = initial_vertex
        
        # Storage
        gap_history = []
        objective_history = []
        
        if verbose:
            print(f"Frank-Wolfe Optimization")
            print(f"{'Iter':>6} {'Gap':>12} {'Objective':>12} {'Gamma':>8} {'Time(s)':>8}")
            print("-" * 60)
        
        for k in range(self.max_iterations):
            iter_start = time.time()
            
            # (a) Compute gradient
            gradient = self.gradient_kl(mu_k, theta)
            
            # (b) Solve IP oracle
            try:
                z_k = ip_oracle(gradient)
            except Exception as e:
                warnings.warn(f"IP oracle failed at iteration {k}: {e}")
                break
            
            # (c) Line search
            gamma_k = self.line_search(mu_k, z_k, theta)
            
            # (d) Update
            mu_k_new = (1 - gamma_k) * mu_k + gamma_k * z_k
            
            # (e) Compute duality gap
            gap = np.dot(gradient, mu_k - z_k)
            gap_history.append(gap)
            
            # Compute objective
            obj = self.kl_divergence(mu_k_new, theta)
            objective_history.append(obj)
            
            iter_time = time.time() - iter_start
            
            if verbose and (k % 10 == 0 or gap < self.convergence_threshold):
                print(f"{k:6d} {gap:12.6e} {obj:12.6e} {gamma_k:8.4f} {iter_time:8.4f}")
            
            # (f) Check convergence
            if gap < self.convergence_threshold:
                mu_k = mu_k_new
                if verbose:
                    print(f"\nConverged in {k+1} iterations!")
                break
            
            mu_k = mu_k_new
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nTotal time: {total_time:.2f} seconds")
        
        return {
            'mu_star': mu_k,
            'iterations': k + 1,
            'gap_history': gap_history,
            'objective_history': objective_history,
            'converged': gap < self.convergence_threshold,
            'total_time': total_time
        }


class BarrierFrankWolfe(FrankWolfeOptimizer):
    """
    Barrier Frank-Wolfe for numerical stability.
    
    Prevents gradient explosion when μ approaches 0 by
    contracting the polytope:
    
    M_ε = (1 - ε)M + εu
    
    where u is the center (e.g., uniform distribution).
    """
    
    def __init__(self,
                 initial_epsilon: float = 0.1,
                 epsilon_reduction: float = 0.9,
                 min_epsilon: float = 1e-6,
                 **kwargs):
        """
        Args:
            initial_epsilon: Initial contraction (0.1 = 10% contraction)
            epsilon_reduction: Multiply epsilon by this each outer iteration
            min_epsilon: Stop when epsilon < this
        """
        super().__init__(**kwargs)
        self.initial_epsilon = initial_epsilon
        self.epsilon_reduction = epsilon_reduction
        self.min_epsilon = min_epsilon
    
    def contract_vertex(self,
                       z: np.ndarray,
                       center: np.ndarray,
                       epsilon: float) -> np.ndarray:
        """
        Contract vertex z toward center u:
        z_ε = (1 - ε)z + εu
        
        Args:
            z: Original vertex
            center: Center point (e.g., uniform distribution)
            epsilon: Contraction parameter
        
        Returns:
            Contracted vertex
        """
        return (1 - epsilon) * z + epsilon * center
    
    def optimize_with_barrier(self,
                             theta: np.ndarray,
                             ip_oracle: Callable[[np.ndarray], np.ndarray],
                             center: Optional[np.ndarray] = None,
                             verbose: bool = True) -> dict:
        """
        Run Barrier Frank-Wolfe with adaptive epsilon reduction.
        
        Args:
            theta: Current prices
            ip_oracle: IP oracle for original polytope M
            center: Center of polytope (if None, use uniform)
            verbose: Print progress
        
        Returns:
            Same as optimize(), plus:
                - epsilon_history: Epsilon value at each outer iteration
        """
        n = len(theta)
        
        if center is None:
            center = np.ones(n) / n
        
        epsilon = self.initial_epsilon
        epsilon_history = []
        
        # Overall result tracking
        best_result = None
        best_obj = float('inf')
        
        if verbose:
            print(f"Barrier Frank-Wolfe Optimization")
            print(f"Initial ε: {epsilon:.6f}, Reduction: {self.epsilon_reduction}")
            print("=" * 60)
        
        while epsilon > self.min_epsilon:
            if verbose:
                print(f"\n--- Outer iteration: ε = {epsilon:.6e} ---")
            
            # Create contracted IP oracle
            def contracted_oracle(gradient):
                # Solve over original polytope
                z_original = ip_oracle(gradient)
                # Contract toward center
                z_contracted = self.contract_vertex(z_original, center, epsilon)
                return z_contracted
            
            # Run Frank-Wolfe on contracted polytope
            result = self.optimize(
                theta,
                contracted_oracle,
                initial_vertex=center if best_result is None else best_result['mu_star'],
                verbose=False
            )
            
            epsilon_history.append(epsilon)
            
            # Track best solution
            final_obj = result['objective_history'][-1] if result['objective_history'] else float('inf')
            if final_obj < best_obj:
                best_obj = final_obj
                best_result = result
            
            if verbose:
                print(f"  Iterations: {result['iterations']}")
                print(f"  Final gap: {result['gap_history'][-1]:.6e}")
                print(f"  Objective: {final_obj:.6e}")
            
            # Reduce epsilon
            epsilon *= self.epsilon_reduction
        
        if verbose:
            print(f"\nBarrier method complete!")
        
        best_result['epsilon_history'] = epsilon_history
        
        return best_result


def simple_3outcome_oracle(gradient: np.ndarray) -> np.ndarray:
    """
    IP oracle for 3-outcome mutually exclusive market.
    
    Valid outcomes: {[1,0,0], [0,1,0], [0,0,1]}
    
    Returns vertex that minimizes gradient · z.
    """
    # For probability simplex, oracle is trivial:
    # Pick the outcome with minimum gradient component
    i = np.argmin(gradient)
    z = np.zeros(len(gradient))
    z[i] = 1.0
    return z


def dependent_markets_oracle(gradient: np.ndarray) -> np.ndarray:
    """
    IP oracle for two dependent markets: B implies A.
    
    Valid outcomes: {[0,0], [1,0], [1,1]}
    
    Constraint: z[1] <= z[0] (if B=1 then A=1)
    """
    # Enumerate valid outcomes
    valid_outcomes = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0])
    ]
    
    # Find minimum
    best_z = valid_outcomes[0]
    best_val = np.dot(gradient, best_z)
    
    for z in valid_outcomes[1:]:
        val = np.dot(gradient, z)
        if val < best_val:
            best_val = val
            best_z = z
    
    return best_z


def demo_frank_wolfe():
    """
    Demonstrate Frank-Wolfe algorithm on example markets.
    """
    print("=" * 60)
    print("DEMO: Frank-Wolfe Algorithm")
    print("=" * 60)
    
    # Example 1: Simple 3-outcome market
    print("\n--- Example 1: Three-outcome market ---")
    theta1 = np.array([0.35, 0.40, 0.35])
    print(f"Current prices: {theta1} (sum = {np.sum(theta1):.2f})")
    
    optimizer1 = FrankWolfeOptimizer(max_iterations=50, convergence_threshold=1e-6)
    result1 = optimizer1.optimize(theta1, simple_3outcome_oracle, verbose=True)
    
    print(f"\nProjected prices: {result1['mu_star']}")
    print(f"Sum: {np.sum(result1['mu_star']):.6f}")
    print(f"Converged: {result1['converged']}")
    
    # Example 2: Dependent markets
    print("\n" + "=" * 60)
    print("--- Example 2: Dependent markets (B implies A) ---")
    theta2 = np.array([0.40, 0.70])
    print(f"Market A (Trump wins PA): ${theta2[0]:.2f}")
    print(f"Market B (Republicans +5): ${theta2[1]:.2f}")
    
    optimizer2 = FrankWolfeOptimizer(max_iterations=50, convergence_threshold=1e-6)
    result2 = optimizer2.optimize(theta2, dependent_markets_oracle, verbose=True)
    
    print(f"\nProjected prices: {result2['mu_star']}")
    print(f"Now satisfies B <= A: {result2['mu_star'][1]:.3f} <= {result2['mu_star'][0]:.3f}")
    
    # Example 3: Barrier Frank-Wolfe
    print("\n" + "=" * 60)
    print("--- Example 3: Barrier Frank-Wolfe ---")
    
    barrier_optimizer = BarrierFrankWolfe(
        initial_epsilon=0.1,
        epsilon_reduction=0.8,
        min_epsilon=1e-4,
        max_iterations=30
    )
    
    result3 = barrier_optimizer.optimize_with_barrier(
        theta1,
        simple_3outcome_oracle,
        verbose=True
    )
    
    print(f"\nFinal projected prices: {result3['mu_star']}")
    print(f"Epsilon history: {[f'{e:.6f}' for e in result3['epsilon_history']]}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_frank_wolfe()
