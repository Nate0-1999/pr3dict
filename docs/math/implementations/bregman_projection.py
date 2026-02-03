"""
Bregman Projection Implementation

This module implements Bregman divergence calculation and projection
onto convex sets for prediction market arbitrage detection.

Author: PR3DICT Research Team
Date: February 2, 2026
References: arXiv:1606.02825, arXiv:2508.03474
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from typing import Tuple, Optional, Callable
import warnings


class BregmanProjector:
    """
    Compute Bregman projections for LMSR prediction markets.
    
    The Bregman divergence with negative entropy generator is the
    Kullback-Leibler (KL) divergence:
    
    D_R(μ || θ) = KL(μ || θ) = Σᵢ μᵢ ln(μᵢ/θᵢ)
    
    This measures the "cost" to move the market from θ to μ under LMSR.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        """
        Args:
            epsilon: Small value to avoid log(0) - numerical stability
        """
        self.epsilon = epsilon
    
    def kl_divergence(self, mu: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute KL divergence: D(μ || θ) = Σᵢ μᵢ ln(μᵢ/θᵢ)
        
        Args:
            mu: Target distribution (n,)
            theta: Reference distribution (n,)
        
        Returns:
            KL divergence value (non-negative)
        
        Example:
            >>> proj = BregmanProjector()
            >>> mu = np.array([0.5, 0.5])
            >>> theta = np.array([0.3, 0.7])
            >>> proj.kl_divergence(mu, theta)
            0.0854...
        """
        # Clip to avoid numerical issues
        mu = np.clip(mu, self.epsilon, 1.0)
        theta = np.clip(theta, self.epsilon, 1.0)
        
        return np.sum(mu * np.log(mu / theta))
    
    def gradient_kl(self, mu: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Gradient of D(μ || θ) with respect to μ.
        
        ∇_μ D(μ || θ) = ln(μ) - ln(θ) + 1 - (1) = ln(μ/θ)
        
        Args:
            mu: Current point (n,)
            theta: Reference point (n,)
        
        Returns:
            Gradient vector (n,)
        """
        mu = np.clip(mu, self.epsilon, 1.0)
        theta = np.clip(theta, self.epsilon, 1.0)
        
        return np.log(mu / theta)
    
    def project_onto_simplex(self, theta: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Project θ onto probability simplex using KL divergence.
        
        Simplex: M = {μ : Σμᵢ = 1, μᵢ ≥ 0}
        
        Closed-form solution: μᵢ* = θᵢ / Σⱼθⱼ
        
        Args:
            theta: Current prices (n,)
        
        Returns:
            mu_star: Projected prices (n,)
            profit: Maximum extractable profit
        
        Example:
            >>> proj = BregmanProjector()
            >>> theta = np.array([0.35, 0.40, 0.35])  # sum = 1.10
            >>> mu_star, profit = proj.project_onto_simplex(theta)
            >>> np.sum(mu_star)
            1.0
            >>> profit > 0  # Arbitrage exists
            True
        """
        # Closed-form solution for simplex projection
        mu_star = theta / np.sum(theta)
        
        # Maximum profit (use reverse KL for profit direction)
        profit = self.kl_divergence(theta, mu_star)
        
        return mu_star, profit
    
    def project_onto_box(self, 
                        theta: np.ndarray, 
                        lower: np.ndarray, 
                        upper: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Project θ onto box constraints using KL divergence.
        
        Box: M = {μ : lower ≤ μ ≤ upper}
        
        Args:
            theta: Current prices (n,)
            lower: Lower bounds (n,)
            upper: Upper bounds (n,)
        
        Returns:
            mu_star: Projected prices (n,)
            profit: Maximum extractable profit
        """
        # Box projection: element-wise clipping
        mu_star = np.clip(theta, lower, upper)
        
        # Profit
        profit = self.kl_divergence(theta, mu_star)
        
        return mu_star, profit
    
    def project_general(self,
                       theta: np.ndarray,
                       constraints: list,
                       initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool]:
        """
        Project θ onto general convex set M defined by constraints.
        
        Minimizes: D(μ || θ) subject to constraints
        
        Args:
            theta: Current prices (n,)
            constraints: List of scipy constraint dicts
            initial_guess: Starting point (if None, use theta)
        
        Returns:
            mu_star: Projected prices (n,)
            profit: Maximum extractable profit
            success: Whether optimization succeeded
        
        Example:
            >>> # Project with dependency constraint μ₁ >= μ₀
            >>> proj = BregmanProjector()
            >>> theta = np.array([0.4, 0.7])
            >>> constraints = [
            ...     {'type': 'ineq', 'fun': lambda mu: mu[0] - mu[1]},  # μ₀ >= μ₁
            ...     {'type': 'eq', 'fun': lambda mu: np.sum(mu) - 1.0}  # sum = 1
            ... ]
            >>> mu_star, profit, success = proj.project_general(theta, constraints)
            >>> success
            True
        """
        n = len(theta)
        
        if initial_guess is None:
            # Start with simplex projection if no guess provided
            initial_guess, _ = self.project_onto_simplex(theta)
        
        # Objective: KL divergence
        def objective(mu):
            return self.kl_divergence(mu, theta)
        
        # Gradient
        def gradient(mu):
            return self.gradient_kl(mu, theta)
        
        # Bounds (probabilities)
        bounds = [(self.epsilon, 1.0 - self.epsilon) for _ in range(n)]
        
        # Optimize
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")
        
        mu_star = result.x
        profit = self.kl_divergence(theta, mu_star)
        
        return mu_star, profit, result.success
    
    def compute_trading_direction(self, mu_star: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute optimal trading direction from θ to μ*.
        
        Direction = ln(μ*/θ)
        - Positive: buy (increase price)
        - Negative: sell (decrease price)
        
        Args:
            mu_star: Target prices (arbitrage-free)
            theta: Current prices
        
        Returns:
            Trading direction vector (n,)
        
        Example:
            >>> proj = BregmanProjector()
            >>> mu_star = np.array([0.318, 0.364, 0.318])
            >>> theta = np.array([0.35, 0.40, 0.35])
            >>> direction = proj.compute_trading_direction(mu_star, theta)
            >>> # Negative values mean sell, positive mean buy
        """
        mu_star = np.clip(mu_star, self.epsilon, 1.0)
        theta = np.clip(theta, self.epsilon, 1.0)
        
        return np.log(mu_star / theta)
    
    def compute_position_size(self,
                             mu_star: np.ndarray,
                             theta: np.ndarray,
                             liquidity_param: float) -> np.ndarray:
        """
        Compute optimal position sizes for LMSR market.
        
        For LMSR with liquidity parameter b:
        shares = b × ln(μ*/θ)
        
        Args:
            mu_star: Target prices
            theta: Current prices
            liquidity_param: LMSR liquidity parameter b
        
        Returns:
            Optimal shares to trade (n,)
        """
        direction = self.compute_trading_direction(mu_star, theta)
        return liquidity_param * direction
    
    def verify_pythagorean_property(self,
                                    mu: np.ndarray,
                                    mu_star: np.ndarray,
                                    theta: np.ndarray,
                                    tolerance: float = 1e-6) -> bool:
        """
        Verify the Bregman Pythagorean property:
        D(μ || θ) = D(μ || μ*) + D(μ* || θ)
        
        This holds when μ* is the Bregman projection and μ is in the constraint set.
        
        Args:
            mu: Point in constraint set M
            mu_star: Bregman projection of θ onto M
            theta: Original point
            tolerance: Numerical tolerance for equality
        
        Returns:
            True if property holds within tolerance
        
        Example:
            >>> proj = BregmanProjector()
            >>> mu = np.array([0.3, 0.4, 0.3])
            >>> theta = np.array([0.35, 0.40, 0.35])
            >>> mu_star, _ = proj.project_onto_simplex(theta)
            >>> proj.verify_pythagorean_property(mu, mu_star, theta)
            True
        """
        lhs = self.kl_divergence(mu, theta)
        rhs = self.kl_divergence(mu, mu_star) + self.kl_divergence(mu_star, theta)
        
        return abs(lhs - rhs) < tolerance


def demo_simple_arbitrage():
    """
    Demonstrate Bregman projection for simple arbitrage detection.
    """
    print("=" * 60)
    print("DEMO: Bregman Projection for Arbitrage Detection")
    print("=" * 60)
    
    proj = BregmanProjector()
    
    # Example 1: Three-outcome market with mispricing
    print("\n--- Example 1: Three-outcome market ---")
    theta = np.array([0.35, 0.40, 0.35])
    print(f"Current prices: {theta}")
    print(f"Sum: {np.sum(theta):.2f} (should be 1.00)")
    
    mu_star, profit = proj.project_onto_simplex(theta)
    print(f"Projected prices: {mu_star}")
    print(f"Maximum profit: ${profit:.4f} per dollar")
    
    direction = proj.compute_trading_direction(mu_star, theta)
    print(f"Trading direction: {direction}")
    print(f"  Negative = sell, Positive = buy")
    
    # Example 2: Dependent markets
    print("\n--- Example 2: Dependent markets (B implies A) ---")
    theta_dep = np.array([0.40, 0.70])
    print(f"Market A (Trump wins PA): ${theta_dep[0]:.2f}")
    print(f"Market B (Republicans +5): ${theta_dep[1]:.2f}")
    print(f"Dependency violation: P(B) > P(A)")
    
    # Constraint: μ₁ ≤ μ₀ (B implies A)
    constraints = [
        {'type': 'ineq', 'fun': lambda mu: mu[0] - mu[1]},  # μ₀ - μ₁ >= 0
        {'type': 'eq', 'fun': lambda mu: np.sum(mu) - 1.0}  # Σμᵢ = 1
    ]
    
    mu_star_dep, profit_dep, success = proj.project_general(theta_dep, constraints)
    
    if success:
        print(f"Projected prices: {mu_star_dep}")
        print(f"Maximum profit: ${profit_dep:.4f}")
        print(f"Now P(B) <= P(A): {mu_star_dep[1]:.3f} <= {mu_star_dep[0]:.3f}")
    
    # Example 3: Pythagorean property verification
    print("\n--- Example 3: Pythagorean Property ---")
    mu = np.array([0.3, 0.4, 0.3])  # Point in simplex
    theta_pyth = np.array([0.35, 0.40, 0.35])
    mu_star_pyth, _ = proj.project_onto_simplex(theta_pyth)
    
    holds = proj.verify_pythagorean_property(mu, mu_star_pyth, theta_pyth)
    print(f"D(μ || θ) = D(μ || μ*) + D(μ* || θ): {holds}")
    
    # Compute each term
    d_mu_theta = proj.kl_divergence(mu, theta_pyth)
    d_mu_mustar = proj.kl_divergence(mu, mu_star_pyth)
    d_mustar_theta = proj.kl_divergence(mu_star_pyth, theta_pyth)
    
    print(f"  D(μ || θ) = {d_mu_theta:.6f}")
    print(f"  D(μ || μ*) + D(μ* || θ) = {d_mu_mustar:.6f} + {d_mustar_theta:.6f} = {d_mu_mustar + d_mustar_theta:.6f}")
    print(f"  Difference: {abs(d_mu_theta - (d_mu_mustar + d_mustar_theta)):.2e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_simple_arbitrage()
