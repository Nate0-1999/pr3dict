#!/usr/bin/env python3
"""
Quick test of the PR3DICT Code Inspector system.

Run: python test_inspector.py
"""
import asyncio
import tempfile
from pathlib import Path
from src.validation import InspectionManager, IssueSeverity


# Test files
VALID_CODE = '''"""
Simple valid strategy for testing.
"""
from typing import List
from src.strategies.base import TradingStrategy, Signal
from src.platforms.base import Market, Position


class TestStrategy(TradingStrategy):
    """A simple test strategy."""
    
    @property
    def name(self) -> str:
        return "test_strategy"
    
    async def scan_markets(self, markets: List[Market]) -> List[Signal]:
        """Scan markets for signals."""
        return []
    
    async def check_exit(self, position: Position, market: Market):
        """Check if should exit position."""
        return None
'''

INVALID_CODE = '''"""
Broken strategy with syntax errors.
"""
def broken_function(:  # Syntax error
    pass

class BrokenStrategy:  # Missing base class
    def some_method(self):
        return undefined_variable  # Will fail
'''

MISSING_DOCSTRINGS = '''
from typing import List

class NoDocstring:
    def no_docs(self):
        return 42
'''


async def test_tier1():
    """Test Tier 1 (hardcoded checks)."""
    print("\n" + "=" * 60)
    print("TEST 1: Tier 1 Hardcoded Checks")
    print("=" * 60)
    
    manager = InspectionManager(enable_llm=False, enable_testing=False)
    
    # Test 1.1: Valid code
    print("\n1.1 Testing valid code...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(VALID_CODE)
        temp_file = f.name
    
    result = await manager.inspect_file(temp_file)
    Path(temp_file).unlink()
    
    print(f"   Issues found: {len(result.all_issues())}")
    print(f"   Duration: {result.tier1_duration_ms:.1f}ms")
    
    if not result.has_critical_issues():
        print("   ✅ PASS - No critical issues in valid code")
    else:
        print("   ❌ FAIL - Found unexpected critical issues")
        for issue in result.get_critical_issues():
            print(f"      {issue}")
    
    # Test 1.2: Invalid code
    print("\n1.2 Testing invalid code...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(INVALID_CODE)
        temp_file = f.name
    
    result = await manager.inspect_file(temp_file)
    Path(temp_file).unlink()
    
    print(f"   Issues found: {len(result.all_issues())}")
    print(f"   Critical: {len(result.get_critical_issues())}")
    
    if result.has_critical_issues():
        print("   ✅ PASS - Correctly detected syntax errors")
        for issue in result.get_critical_issues()[:3]:
            print(f"      {issue}")
    else:
        print("   ❌ FAIL - Should have found syntax errors")
    
    # Test 1.3: Missing docstrings
    print("\n1.3 Testing missing docstrings...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(MISSING_DOCSTRINGS)
        temp_file = f.name
    
    result = await manager.inspect_file(temp_file)
    Path(temp_file).unlink()
    
    has_doc_warnings = any(
        'docstring' in issue.message.lower()
        for issue in result.all_issues()
    )
    
    if has_doc_warnings:
        print("   ✅ PASS - Detected missing docstrings")
    else:
        print("   ⚠️  WARNING - Should detect missing docstrings")


async def test_existing_file():
    """Test on real PR3DICT file."""
    print("\n" + "=" * 60)
    print("TEST 2: Real File Inspection")
    print("=" * 60)
    
    # Find a real strategy file
    strategy_file = Path("src/strategies/market_making.py")
    
    if not strategy_file.exists():
        print("   ⚠️  SKIP - market_making.py not found")
        return
    
    print(f"\n2.1 Inspecting {strategy_file}...")
    
    manager = InspectionManager(enable_llm=False, enable_testing=False)
    result = await manager.inspect_file(str(strategy_file))
    
    print(f"   Issues found: {len(result.all_issues())}")
    print(f"   Critical: {len(result.get_critical_issues())}")
    print(f"   Errors: {len(result.get_errors())}")
    print(f"   Duration: {result.tier1_duration_ms:.1f}ms")
    
    if result.has_critical_issues():
        print("   ❌ FAIL - Existing file has critical issues!")
        for issue in result.get_critical_issues():
            print(f"      {issue}")
    else:
        print("   ✅ PASS - No critical issues in existing code")
    
    if result.all_issues():
        print(f"\n   Sample issues:")
        for issue in result.all_issues()[:5]:
            print(f"      [{issue.severity.value}] {issue.message}")


async def test_caching():
    """Test caching functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Caching System")
    print("=" * 60)
    
    print("\n3.1 Testing cache write/read...")
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(VALID_CODE)
        temp_file = f.name
    
    # First inspection - should cache
    manager1 = InspectionManager(enable_llm=False, enable_cache=True)
    result1 = await manager1.inspect_file(temp_file)
    hash1 = result1.file_hash
    
    # Second inspection - should use cache (file unchanged)
    manager2 = InspectionManager(enable_llm=False, enable_cache=True)
    result2 = await manager2.inspect_file(temp_file)
    hash2 = result2.file_hash
    
    Path(temp_file).unlink()
    
    if hash1 == hash2:
        print("   ✅ PASS - Cache hash consistency")
    else:
        print("   ❌ FAIL - Cache hash mismatch")


async def test_performance():
    """Test performance benchmarks."""
    print("\n" + "=" * 60)
    print("TEST 4: Performance Benchmarks")
    print("=" * 60)
    
    print("\n4.1 Speed test (10 files)...")
    
    manager = InspectionManager(enable_llm=False, enable_testing=False)
    
    import time
    start = time.time()
    
    for i in range(10):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(VALID_CODE)
            temp_file = f.name
        
        await manager.inspect_file(temp_file)
        Path(temp_file).unlink()
    
    elapsed = time.time() - start
    per_file = elapsed / 10
    
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Per file: {per_file * 1000:.1f}ms")
    
    if per_file < 0.1:
        print("   ✅ PASS - Fast enough (< 100ms per file)")
    else:
        print(f"   ⚠️  WARNING - Slower than expected ({per_file * 1000:.1f}ms)")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PR3DICT Code Inspector - Test Suite")
    print("=" * 60)
    
    try:
        await test_tier1()
        await test_existing_file()
        await test_caching()
        await test_performance()
        
        print("\n" + "=" * 60)
        print("All tests complete!")
        print("=" * 60)
        print("\nNote: Tier 2 (LLM) tests skipped (requires API)")
        print("To test LLM: ./inspect.sh file <your_file.py>")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)
