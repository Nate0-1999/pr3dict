"""
PR3DICT Optimization Module

Integer Programming and Linear Programming solvers for optimal arbitrage execution.

Key Components:
- solver.py: Main optimization engine (LP/IP/Frank-Wolfe)
- benchmarks.py: Performance and quality metrics
- constraints.py: Constraint definitions and marginal polytope
"""

from .solver import (
    ArbitrageSolver,
    OptimizationResult,
    SolverBackend,
    TradeAllocation
)

__all__ = [
    'ArbitrageSolver',
    'OptimizationResult',
    'SolverBackend',
    'TradeAllocation'
]
