# Sub-Agent Integration Guide

## Overview

This guide shows how to integrate the PR3DICT Code Inspector into OpenClaw's sub-agent workflow for automated code generation and validation.

## Architecture

```
Main Agent
    ↓
  Spawns Sub-Agent (code generator)
    ↓
  Generates code
    ↓
  Runs Inspector (Tier 1 + 2)
    ↓
  [Critical Issues?] → Yes → Report to Main Agent → Request Fix
    ↓ No
  [Errors?] → Yes → Auto-fix attempt → Re-inspect
    ↓ No
  Success → Return to Main Agent
```

---

## Pattern 1: Validate After Generation

Use this pattern when generating new code files.

```python
from src.validation import InspectionManager, IssueSeverity
from pathlib import Path

async def generate_strategy_with_validation(strategy_spec: dict):
    """
    Generate a strategy and validate it before returning.
    
    Returns:
        dict with 'status', 'file_path', 'issues'
    """
    # 1. Generate code
    code = await generate_strategy_code(strategy_spec)
    
    # 2. Write to file
    file_path = Path(f"src/strategies/{strategy_spec['name']}.py")
    file_path.write_text(code)
    
    # 3. Inspect with Tier 1 + 2
    manager = InspectionManager(
        enable_llm=True,      # Deep semantic review
        enable_testing=False, # Skip tests for now
        enable_cache=True
    )
    
    result = await manager.inspect_file(str(file_path))
    
    # 4. Handle results
    if result.has_critical_issues():
        return {
            'status': 'critical_failure',
            'file_path': str(file_path),
            'issues': [
                {
                    'severity': issue.severity.value,
                    'message': issue.message,
                    'line': issue.line_number,
                    'suggestion': issue.suggestion
                }
                for issue in result.get_critical_issues()
            ]
        }
    
    elif result.has_errors():
        return {
            'status': 'has_errors',
            'file_path': str(file_path),
            'issues': [
                {
                    'severity': issue.severity.value,
                    'message': issue.message,
                    'line': issue.line_number,
                    'suggestion': issue.suggestion
                }
                for issue in result.get_errors()
            ]
        }
    
    else:
        return {
            'status': 'success',
            'file_path': str(file_path),
            'warnings': len([i for i in result.all_issues() 
                           if i.severity == IssueSeverity.WARNING]),
            'cost_usd': result.llm_cost_usd
        }
```

---

## Pattern 2: Auto-Fix Workflow

Attempt to fix errors automatically, with fallback to manual review.

```python
async def generate_with_auto_fix(strategy_spec: dict, max_attempts: int = 3):
    """
    Generate code with automatic fixing of common issues.
    """
    file_path = Path(f"src/strategies/{strategy_spec['name']}.py")
    manager = InspectionManager(enable_llm=True)
    
    for attempt in range(max_attempts):
        # Generate code
        if attempt == 0:
            code = await generate_strategy_code(strategy_spec)
        else:
            # Re-generate with issue context
            code = await regenerate_with_fixes(strategy_spec, previous_issues)
        
        file_path.write_text(code)
        
        # Inspect
        result = await manager.inspect_file(str(file_path))
        
        # Success?
        if not result.has_errors():
            return {
                'status': 'success',
                'file_path': str(file_path),
                'attempts': attempt + 1,
                'final_warnings': len([i for i in result.all_issues()
                                     if i.severity == IssueSeverity.WARNING])
            }
        
        # Critical issues - abort
        if result.has_critical_issues():
            return {
                'status': 'failed',
                'reason': 'critical_issues_unfixable',
                'issues': [str(i) for i in result.get_critical_issues()]
            }
        
        # Save issues for next attempt
        previous_issues = result.get_errors()
    
    # Max attempts reached
    return {
        'status': 'failed',
        'reason': 'max_fix_attempts_exceeded',
        'attempts': max_attempts,
        'remaining_issues': [str(i) for i in previous_issues]
    }


async def regenerate_with_fixes(spec: dict, issues: list):
    """Re-generate code with issue context."""
    issue_context = "\n".join([
        f"- {issue.message} (line {issue.line_number})"
        for issue in issues
    ])
    
    prompt = f"""
    Previous attempt had these issues:
    {issue_context}
    
    Please regenerate the strategy fixing these specific issues.
    
    Strategy spec: {spec}
    """
    
    # Call your code generation LLM here
    return await llm_generate_code(prompt)
```

---

## Pattern 3: Incremental Validation

Validate changes incrementally as files are modified.

```python
class IncrementalValidator:
    """Track and validate only changed files."""
    
    def __init__(self):
        self.manager = InspectionManager(enable_llm=True)
        self.file_hashes = {}
    
    async def validate_if_changed(self, file_path: str):
        """Only validate if file content changed."""
        import hashlib
        
        path = Path(file_path)
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        
        # Check if changed
        if self.file_hashes.get(str(path)) == current_hash:
            return {'status': 'unchanged', 'file_path': str(path)}
        
        # Validate
        result = await self.manager.inspect_file(str(path))
        
        # Update hash
        self.file_hashes[str(path)] = current_hash
        
        return {
            'status': 'validated',
            'file_path': str(path),
            'has_errors': result.has_errors(),
            'issue_count': len(result.all_issues())
        }


# Usage in sub-agent
validator = IncrementalValidator()

# Only validates if file changed
result = await validator.validate_if_changed("src/strategies/momentum.py")
```

---

## Pattern 4: Batch Validation

Validate multiple generated files efficiently.

```python
async def batch_generate_and_validate(strategy_specs: list):
    """
    Generate multiple strategies and validate them all.
    
    More efficient than one-by-one due to caching.
    """
    manager = InspectionManager(enable_llm=True)
    results = []
    
    # 1. Generate all files
    files = []
    for spec in strategy_specs:
        code = await generate_strategy_code(spec)
        file_path = Path(f"src/strategies/{spec['name']}.py")
        file_path.write_text(code)
        files.append(file_path)
    
    # 2. Batch validate
    validation_results = await manager.inspect_directory(
        "src/strategies/",
        pattern="*.py"  # Only newly generated files
    )
    
    # 3. Categorize results
    success = []
    needs_fix = []
    failed = []
    
    for result in validation_results:
        if result.has_critical_issues():
            failed.append(result)
        elif result.has_errors():
            needs_fix.append(result)
        else:
            success.append(result)
    
    return {
        'total': len(validation_results),
        'success': len(success),
        'needs_fix': len(needs_fix),
        'failed': len(failed),
        'total_cost': sum(r.llm_cost_usd for r in validation_results),
        'details': {
            'success_files': [r.file_path for r in success],
            'needs_fix_files': [r.file_path for r in needs_fix],
            'failed_files': [r.file_path for r in failed]
        }
    }
```

---

## Pattern 5: Report to Main Agent

Structure validation results for clear communication to main agent.

```python
def format_validation_report(result: InspectionResult) -> dict:
    """
    Format inspection result for sub-agent → main agent communication.
    """
    critical = result.get_critical_issues()
    errors = [i for i in result.all_issues() 
              if i.severity == IssueSeverity.ERROR]
    warnings = [i for i in result.all_issues() 
                if i.severity == IssueSeverity.WARNING]
    
    # Build report
    report = {
        'file': result.file_path,
        'timestamp': result.timestamp.isoformat(),
        'status': 'pass' if not result.has_errors() else 'fail',
        'summary': {
            'critical': len(critical),
            'errors': len(errors),
            'warnings': len(warnings),
            'total_issues': len(result.all_issues())
        },
        'performance': {
            'tier1_ms': result.tier1_duration_ms,
            'tier2_ms': result.tier2_duration_ms,
            'cost_usd': result.llm_cost_usd
        }
    }
    
    # Add details for failures
    if critical:
        report['critical_issues'] = [
            {
                'message': issue.message,
                'line': issue.line_number,
                'category': issue.category.value,
                'suggestion': issue.suggestion
            }
            for issue in critical
        ]
    
    if errors:
        report['errors'] = [
            {
                'message': issue.message,
                'line': issue.line_number,
                'suggestion': issue.suggestion
            }
            for issue in errors
        ]
    
    return report


# In sub-agent final response
validation_result = await manager.inspect_file("src/strategies/new_strategy.py")
report = format_validation_report(validation_result)

# Return to main agent
return {
    'task': 'generate_strategy',
    'status': 'complete',
    'validation': report,
    'action_required': 'review' if report['summary']['warnings'] > 0 else None
}
```

---

## Pattern 6: Smart LLM Usage (Cost Optimization)

Only use LLM for files that need it.

```python
async def smart_validate(file_path: str):
    """
    Use Tier 1 first, only run Tier 2 if needed.
    """
    manager = InspectionManager(enable_llm=False)
    
    # Run fast Tier 1 first
    tier1_inspector = manager.tier1
    tier1_issues = tier1_inspector.inspect(file_path)
    
    # If critical Tier 1 issues, stop here
    if any(i.severity == IssueSeverity.CRITICAL for i in tier1_issues):
        return {
            'status': 'failed_tier1',
            'issues': tier1_issues,
            'cost_usd': 0.0
        }
    
    # If no Tier 1 errors, run LLM review
    if not any(i.severity == IssueSeverity.ERROR for i in tier1_issues):
        manager_llm = InspectionManager(enable_llm=True)
        result = await manager_llm.inspect_file(file_path)
        return {
            'status': 'complete',
            'result': result,
            'cost_usd': result.llm_cost_usd
        }
    
    # Has Tier 1 errors but not critical - decide based on file type
    if 'strategies' in file_path:
        # Strategies always get LLM review
        manager_llm = InspectionManager(enable_llm=True)
        result = await manager_llm.inspect_file(file_path)
        return {
            'status': 'complete',
            'result': result,
            'cost_usd': result.llm_cost_usd
        }
    else:
        # Non-strategies: fix Tier 1 first
        return {
            'status': 'fix_tier1_first',
            'issues': tier1_issues,
            'cost_usd': 0.0
        }
```

---

## Example: Full Sub-Agent Task

```python
"""
Sub-agent task: Generate and validate a new trading strategy.
"""
import asyncio
from pathlib import Path
from src.validation import InspectionManager, IssueSeverity

async def execute_strategy_generation_task(task_spec: dict):
    """
    Complete sub-agent task with validation.
    
    Args:
        task_spec: {
            'strategy_type': 'momentum|mean_reversion|market_making',
            'name': 'my_strategy',
            'parameters': {...}
        }
    
    Returns:
        Status report for main agent
    """
    strategy_name = task_spec['name']
    file_path = Path(f"src/strategies/{strategy_name}.py")
    
    print(f"🤖 Generating {strategy_name} strategy...")
    
    # Step 1: Generate code
    code = generate_strategy_from_spec(task_spec)
    file_path.write_text(code)
    print(f"✅ Generated {file_path}")
    
    # Step 2: Quick Tier 1 check
    print(f"🔍 Running Tier 1 validation...")
    manager = InspectionManager(enable_llm=False)
    tier1_result = await manager.inspect_file(str(file_path))
    
    if tier1_result.has_critical_issues():
        print("❌ Critical Tier 1 issues found - regenerating...")
        # Attempt one fix
        code = fix_tier1_issues(code, tier1_result.get_critical_issues())
        file_path.write_text(code)
        tier1_result = await manager.inspect_file(str(file_path))
        
        if tier1_result.has_critical_issues():
            return {
                'status': 'failed',
                'reason': 'unfixable_tier1_issues',
                'issues': [str(i) for i in tier1_result.get_critical_issues()]
            }
    
    print(f"✅ Tier 1 passed ({tier1_result.tier1_duration_ms:.1f}ms)")
    
    # Step 3: Deep LLM review
    print(f"🤖 Running Tier 2 (LLM) validation...")
    manager_llm = InspectionManager(enable_llm=True)
    full_result = await manager_llm.inspect_file(str(file_path))
    
    print(f"✅ Tier 2 complete (${full_result.llm_cost_usd:.4f})")
    
    # Step 4: Analyze results
    if full_result.has_errors():
        print(f"⚠️  Found {len(full_result.get_errors())} errors")
        
        # Return for human review
        return {
            'status': 'needs_review',
            'file_path': str(file_path),
            'errors': [
                {
                    'message': i.message,
                    'line': i.line_number,
                    'suggestion': i.suggestion
                }
                for i in full_result.get_errors()
            ],
            'cost': full_result.llm_cost_usd
        }
    
    warnings = [i for i in full_result.all_issues() 
                if i.severity == IssueSeverity.WARNING]
    
    print(f"✅ Strategy generated successfully!")
    print(f"   File: {file_path}")
    print(f"   Warnings: {len(warnings)}")
    print(f"   Cost: ${full_result.llm_cost_usd:.4f}")
    
    return {
        'status': 'success',
        'file_path': str(file_path),
        'warnings': len(warnings),
        'cost': full_result.llm_cost_usd,
        'summary': full_result.summary()
    }


def generate_strategy_from_spec(spec: dict) -> str:
    """Generate strategy code from specification."""
    # Your code generation logic here
    # Could use templates, LLM, or hybrid approach
    pass


def fix_tier1_issues(code: str, issues: list) -> str:
    """Attempt to fix common Tier 1 issues."""
    # Add missing imports
    # Fix syntax errors
    # Add docstrings
    # etc.
    pass


# Run the task
if __name__ == '__main__':
    task = {
        'strategy_type': 'momentum',
        'name': 'momentum_v2',
        'parameters': {
            'lookback_periods': 5,
            'threshold': 0.02
        }
    }
    
    result = asyncio.run(execute_strategy_generation_task(task))
    print(f"\nTask Result: {result}")
```

---

## Integration with OpenClaw Sessions

```python
"""
Example of using inspector in OpenClaw sub-agent session.
"""

# In main agent - spawn sub-agent with inspection task
session_result = openclaw.spawn_subagent(
    label="code-generator",
    task="""
    Generate a new market-making strategy with these specs:
    - Min spread: 2%
    - Max inventory: 50 contracts
    - Rebalance threshold: 20
    
    After generation:
    1. Run code inspector (Tier 1 + 2)
    2. Fix any critical issues
    3. Report back with validation results
    
    Use: from src.validation import InspectionManager
    """
)

# Sub-agent executes and returns
# Main agent receives validation report
```

---

## Best Practices for Sub-Agents

1. **Always run Tier 1 first** - Catch syntax errors before expensive LLM calls
2. **Set max retry attempts** - Don't loop forever on auto-fix
3. **Cache inspection results** - Avoid re-inspecting unchanged files
4. **Report detailed issues** - Help main agent understand what went wrong
5. **Use smart LLM triggering** - Only run Tier 2 when needed
6. **Track costs** - Report LLM usage back to main agent
7. **Handle timeouts gracefully** - LLM calls can be slow
8. **Provide context** - Include tier 1 results when running tier 2

---

## Monitoring & Logging

```python
import logging

# Configure logging for inspection
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

logger = logging.getLogger('pr3dict.validation')

# Log all inspections
async def logged_inspect(file_path: str):
    logger.info(f"Starting inspection of {file_path}")
    
    manager = InspectionManager(enable_llm=True)
    result = await manager.inspect_file(file_path)
    
    logger.info(f"Inspection complete:")
    logger.info(f"  Tier 1: {result.tier1_duration_ms:.1f}ms")
    logger.info(f"  Tier 2: {result.tier2_duration_ms:.1f}ms")
    logger.info(f"  Cost: ${result.llm_cost_usd:.4f}")
    logger.info(f"  Issues: {len(result.all_issues())}")
    
    if result.has_critical_issues():
        logger.error(f"CRITICAL issues found in {file_path}!")
        for issue in result.get_critical_issues():
            logger.error(f"  - {issue.message}")
    
    return result
```

---

## Testing Your Integration

```python
"""
Test your sub-agent integration with the inspector.
"""
import asyncio
from src.validation import InspectionManager

async def test_integration():
    # Test 1: Valid file
    print("Test 1: Valid file...")
    manager = InspectionManager(enable_llm=False)
    result = await manager.inspect_file("src/strategies/market_making.py")
    assert not result.has_critical_issues(), "Should have no critical issues"
    print("✅ Pass")
    
    # Test 2: File with syntax errors
    print("\nTest 2: Syntax error detection...")
    with open("/tmp/test_bad.py", "w") as f:
        f.write("def broken(\n  pass")  # Syntax error
    
    result = await manager.inspect_file("/tmp/test_bad.py")
    assert result.has_critical_issues(), "Should detect syntax error"
    print("✅ Pass")
    
    # Test 3: LLM review
    print("\nTest 3: LLM review...")
    manager_llm = InspectionManager(enable_llm=True)
    result = await manager_llm.inspect_file("src/strategies/market_making.py")
    print(f"Cost: ${result.llm_cost_usd:.4f}")
    print(f"Issues: {len(result.all_issues())}")
    print("✅ Pass")

if __name__ == '__main__':
    asyncio.run(test_integration())
```

---

## Summary

The inspector integrates seamlessly into sub-agent workflows:

1. **Generate** code
2. **Validate** with inspector (Tier 1 → Tier 2)
3. **Fix** issues automatically or report to main agent
4. **Report** results with clear status and actionable feedback

This creates a robust code generation pipeline with quality assurance built in.
