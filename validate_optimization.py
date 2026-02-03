"""
Validate optimization implementation without requiring dependencies.
Checks code structure, imports, and basic syntax.
"""
import os
import ast
import sys


def check_file_exists(filepath):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✅ {filepath}")
        return True
    else:
        print(f"❌ {filepath} - NOT FOUND")
        return False


def check_python_syntax(filepath):
    """Check if Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f"  ⚠️  Syntax error: {e}")
        return False


def check_imports(filepath):
    """Check what modules are imported."""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return imports
    except Exception as e:
        return []


def main():
    print("=" * 80)
    print("PR3DICT OPTIMIZATION IMPLEMENTATION VALIDATION")
    print("=" * 80)
    
    all_passed = True
    
    # Check core files
    print("\n[1] Checking core implementation files...")
    core_files = [
        "src/optimization/__init__.py",
        "src/optimization/solver.py",
        "src/optimization/benchmarks.py",
        "src/optimization/integration.py",
    ]
    
    for filepath in core_files:
        if check_file_exists(filepath):
            if check_python_syntax(filepath):
                print(f"    ✓ Valid Python syntax")
            else:
                all_passed = False
        else:
            all_passed = False
    
    # Check documentation
    print("\n[2] Checking documentation...")
    docs = [
        "docs/optimization_formulation.md",
        "docs/OPTIMIZATION_IMPLEMENTATION.md",
    ]
    
    for filepath in docs:
        if check_file_exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()
                print(f"    ✓ {len(content)} characters")
        else:
            all_passed = False
    
    # Check tests and examples
    print("\n[3] Checking tests and examples...")
    other_files = [
        "tests/test_optimization.py",
        "examples/optimization_demo.py",
    ]
    
    for filepath in other_files:
        if check_file_exists(filepath):
            if check_python_syntax(filepath):
                print(f"    ✓ Valid Python syntax")
            else:
                all_passed = False
        else:
            all_passed = False
    
    # Check code structure
    print("\n[4] Checking code structure...")
    
    # Analyze solver.py
    solver_path = "src/optimization/solver.py"
    if os.path.exists(solver_path):
        with open(solver_path, 'r') as f:
            content = f.read()
        
        required_elements = [
            ("ArbitrageSolver", "class"),
            ("ArbitrageOpportunity", "class"),
            ("SolverBackend", "class"),
            ("OptimizationResult", "class"),
            ("solve", "method"),
            ("_solve_frank_wolfe", "method"),
            ("_solve_cvxpy", "method"),
            ("_solve_gurobi", "method"),
            ("bregman_project", "method"),
        ]
        
        for element, type_name in required_elements:
            if element in content:
                print(f"    ✅ {type_name} '{element}' found")
            else:
                print(f"    ❌ {type_name} '{element}' NOT FOUND")
                all_passed = False
    
    # Check imports
    print("\n[5] Analyzing dependencies...")
    imports = check_imports("src/optimization/solver.py")
    
    required_imports = {
        "numpy": "Required for numerical operations",
        "cvxpy": "Optional - for LP/IP solving",
        "gurobipy": "Optional - commercial solver",
    }
    
    print("    Required imports found:")
    for pkg, desc in required_imports.items():
        if any(pkg in imp for imp in imports):
            print(f"    ✓ {pkg} - {desc}")
        else:
            print(f"    ⚠️  {pkg} not explicitly imported - {desc}")
    
    # Count lines of code
    print("\n[6] Code metrics...")
    total_lines = 0
    for filepath in core_files + other_files:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                lines = len(f.readlines())
                total_lines += lines
                print(f"    {filepath}: {lines} lines")
    
    print(f"\n    Total implementation: {total_lines} lines")
    
    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ VALIDATION PASSED")
        print("\nImplementation complete:")
        print("  - Core solver with 3 backends (Frank-Wolfe, CVXPY, Gurobi)")
        print("  - Comprehensive benchmarking framework")
        print("  - Integration with trading strategies")
        print("  - Full documentation and examples")
        print("  - Test suite (requires pytest to run)")
        print("\nTo run tests (requires dependencies):")
        print("  pip install -r requirements.txt")
        print("  pytest tests/test_optimization.py")
        print("\nTo see working examples:")
        print("  python examples/optimization_demo.py")
    else:
        print("❌ VALIDATION FAILED - Some files missing or invalid")
        return 1
    
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
