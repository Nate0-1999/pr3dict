# PR3DICT Code Inspector - Usage Guide

## Overview

The PR3DICT Code Inspector is a **3-tier hybrid validation system** that combines fast hardcoded checks with deep LLM semantic analysis to ensure code quality.

### The 3 Tiers

**Tier 1: Fast Hardcoded Checks** (milliseconds, free)
- ✅ Python syntax validation
- ✅ Import resolution
- ✅ File structure verification
- ✅ Type hints presence
- ✅ Docstring existence
- ✅ Basic naming conventions
- ✅ Strategy-specific requirements

**Tier 2: LLM Semantic Review** (seconds, ~$0.01/file)
- 🤖 Logic correctness
- 🤖 Edge case handling
- 🤖 Architecture consistency
- 🤖 Security vulnerabilities
- 🤖 Performance anti-patterns
- 🤖 Strategy soundness
- 🤖 Risk management adequacy

**Tier 3: Runtime Testing** (varies, optional)
- 🧪 Unit tests execution
- 🧪 Integration tests
- 🧪 Quick backtests (for strategies)

---

## Quick Start

### Inspect a Single File

```bash
# With LLM review (recommended)
python -m src.validation.inspect --file src/strategies/my_strategy.py

# Fast mode (no LLM, free)
python -m src.validation.inspect --file src/strategies/my_strategy.py --no-llm

# With runtime tests
python -m src.validation.inspect --file src/strategies/my_strategy.py --with-tests
```

### Inspect a Directory

```bash
# Inspect all strategies
python -m src.validation.inspect --dir src/strategies/

# Inspect specific pattern
python -m src.validation.inspect --dir src/ --pattern "**/*strategy*.py"
```

### Verbose Output

```bash
python -m src.validation.inspect --file strategy.py -v
```

---

## When to Use Each Tier

### Use Tier 1 Only (--no-llm)
**When:**
- Quick sanity checks during development
- Pre-commit hooks
- CI/CD pipeline (fast feedback)
- Learning/exploring the codebase
- Budget constraints

**Speed:** < 100ms per file  
**Cost:** Free

### Use Tier 1 + 2 (default)
**When:**
- Pre-merge review
- New strategy development
- Refactoring existing code
- Security-sensitive changes
- Before deployment

**Speed:** 2-10 seconds per file  
**Cost:** ~$0.01 per file

### Use All Tiers (--with-tests)
**When:**
- Final validation before production
- Major releases
- After significant changes
- When debugging test failures

**Speed:** Varies (30s - 5min depending on tests)  
**Cost:** ~$0.01 + test infrastructure

---

## Exit Codes

The inspector returns different exit codes based on severity:

- **0** - ✅ No critical issues or errors (safe to deploy)
- **1** - ⚠️  Errors found (should fix before merge)
- **2** - 🚨 Critical issues found (blocks deployment)

This makes it easy to integrate into CI/CD:

```bash
#!/bin/bash
python -m src.validation.inspect --file "$FILE"
if [ $? -eq 2 ]; then
    echo "Critical issues found! Blocking merge."
    exit 1
fi
```

---

## Issue Severity Levels

### 🚨 CRITICAL
**Blocks deployment.** Must be fixed immediately.

Examples:
- Syntax errors (file won't run)
- Missing required base classes
- Security vulnerabilities
- Unbounded risk scenarios

### ❌ ERROR
**Should fix before merge.** Code may run but has serious issues.

Examples:
- Missing required methods
- Logic errors
- Performance anti-patterns
- Inadequate error handling

### ⚠️  WARNING
**Should review.** Code works but could be improved.

Examples:
- Missing docstrings
- Large file sizes
- Potential edge cases
- Style violations

### ℹ️  INFO
**Nice to have.** Optional improvements.

Examples:
- Missing type hints on some parameters
- Opportunities for optimization
- Documentation suggestions

---

## Caching

LLM reviews are cached automatically to avoid:
- Re-analyzing unchanged files
- Unnecessary API costs
- Slow repeated inspections

**Cache location:** `~/.openclaw/cache/pr3dict_inspector/`

**Cache lifetime:** 7 days

**Disable caching:**
```bash
python -m src.validation.inspect --file strategy.py --no-cache
```

**Clear cache:**
```bash
rm -rf ~/.openclaw/cache/pr3dict_inspector/
```

---

## Programmatic Usage

```python
from src.validation import InspectionManager

# Create manager
manager = InspectionManager(
    enable_llm=True,      # Use LLM semantic review
    enable_testing=False, # Skip runtime tests
    enable_cache=True     # Cache LLM results
)

# Inspect single file
result = await manager.inspect_file("src/strategies/my_strategy.py")

# Check results
if result.has_critical_issues():
    print("Critical issues found!")
    for issue in result.get_critical_issues():
        print(f"  - {issue}")
else:
    print("No critical issues!")

# Inspect directory
results = await manager.inspect_directory("src/strategies/")
print(manager.generate_report(results))
```

---

## Sub-Agent Integration

The inspector is designed to work seamlessly with OpenClaw's sub-agent workflow.

### Auto-Inspection Pattern

```python
# In your code-generating agent
from src.validation import InspectionManager

async def generate_and_validate_strategy(spec: dict):
    # Generate code
    code = generate_strategy_code(spec)
    
    # Write to file
    file_path = "src/strategies/generated_strategy.py"
    with open(file_path, 'w') as f:
        f.write(code)
    
    # Auto-inspect
    manager = InspectionManager(enable_llm=True)
    result = await manager.inspect_file(file_path)
    
    # Handle issues
    if result.has_critical_issues():
        # Report back to parent agent
        return {
            "status": "failed",
            "issues": [str(i) for i in result.get_critical_issues()],
            "file": file_path
        }
    
    elif result.has_errors():
        # Attempt auto-fix
        fixed_code = await auto_fix_code(code, result.get_errors())
        with open(file_path, 'w') as f:
            f.write(fixed_code)
        
        # Re-inspect
        result = await manager.inspect_file(file_path)
    
    return {
        "status": "success",
        "file": file_path,
        "warnings": len([i for i in result.all_issues() 
                        if i.severity == IssueSeverity.WARNING])
    }
```

### Workflow Integration Points

1. **After code generation** - Validate before saving
2. **After file modification** - Re-inspect changed files
3. **Before git commit** - Final validation
4. **In CI/CD** - Automated checks

---

## Cost Optimization

### Strategy 1: Tier 1 First, Tier 2 Selectively

```bash
# Quick check first
python -m src.validation.inspect --file strategy.py --no-llm

# If no Tier 1 issues, run Tier 2
if [ $? -eq 0 ]; then
    python -m src.validation.inspect --file strategy.py
fi
```

### Strategy 2: Batch Reviews

Inspect multiple files in one session to benefit from:
- Shared context
- Batch API calls
- Single LLM initialization

```python
manager = InspectionManager(enable_llm=True)
results = await manager.inspect_directory("src/strategies/")
```

### Strategy 3: Skip LLM for Trivial Changes

```bash
# Only re-inspect with LLM if significant changes
git diff --shortstat $FILE | awk '{
    if ($1 > 50) {
        system("python -m src.validation.inspect --file '$FILE'")
    } else {
        system("python -m src.validation.inspect --file '$FILE' --no-llm")
    }
}'
```

### Strategy 4: Cache Aggressively

Default 7-day cache is conservative. For stable code:

```python
# Extend cache lifetime in inspector.py
# Cache for 30 days instead of 7
if datetime.now() - cached_time > timedelta(days=30):
    return None
```

---

## Examples

### Pre-Commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running PR3DICT code inspection..."

# Get staged Python files
FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

if [ -z "$FILES" ]; then
    exit 0
fi

# Quick Tier 1 check only (fast)
for FILE in $FILES; do
    python -m src.validation.inspect --file "$FILE" --no-llm
    if [ $? -eq 2 ]; then
        echo "❌ Critical issues in $FILE - commit blocked"
        exit 1
    fi
done

echo "✅ All files passed Tier 1 validation"
exit 0
```

### CI/CD Integration (GitHub Actions)

```yaml
name: Code Inspection

on: [pull_request]

jobs:
  inspect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run code inspection
        run: |
          # Full LLM review on PR
          python -m src.validation.inspect \
            --dir src/ \
            --pattern "**/*.py"
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      
      - name: Upload report
        if: failure()
        uses: actions/upload-artifact@v2
        with:
          name: inspection-report
          path: inspection_report.txt
```

### Nightly Full Scan

```bash
#!/bin/bash
# cron: 0 2 * * * /path/to/nightly_scan.sh

cd /path/to/pr3dict

echo "Starting nightly code inspection..."

python -m src.validation.inspect \
    --dir src/ \
    --with-tests \
    --verbose \
    > nightly_inspection_$(date +%Y%m%d).log 2>&1

# Email report if issues found
if [ $? -ne 0 ]; then
    mail -s "PR3DICT Code Issues Found" team@example.com \
        < nightly_inspection_$(date +%Y%m%d).log
fi
```

---

## Troubleshooting

### "LLM call failed"

**Cause:** OpenClaw CLI not available or API key not set

**Fix:**
```bash
# Check OpenClaw is installed
which openclaw

# Set API key
export ANTHROPIC_API_KEY=your-key-here
```

### "pytest not found"

**Cause:** pytest not installed (needed for Tier 3)

**Fix:**
```bash
pip install pytest
```

### "No issues found but code is clearly broken"

**Cause:** Inspector may miss domain-specific issues

**Solution:** Add custom checks to `HardcodedInspector` or improve LLM prompts in `prompts.py`

### "Too slow / too expensive"

**Solutions:**
- Use `--no-llm` for routine checks
- Enable caching (default)
- Batch multiple files in one run
- Run Tier 2 only on changed files

---

## Custom Validators

You can extend the inspector with custom validators:

```python
from src.validation.inspector import HardcodedInspector, ValidationIssue, IssueSeverity, IssueCategory

class CustomStrategyInspector(HardcodedInspector):
    def inspect(self, file_path: str) -> List[ValidationIssue]:
        # Run base checks
        issues = super().inspect(file_path)
        
        # Add custom checks
        content = Path(file_path).read_text()
        
        # Example: Check for hardcoded API keys
        if 'api_key = "' in content:
            issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                category=IssueCategory.SECURITY,
                message="Hardcoded API key detected!",
                suggestion="Use environment variables or config files"
            ))
        
        return issues

# Use custom inspector
manager = InspectionManager()
manager.tier1 = CustomStrategyInspector()
```

---

## Best Practices

1. **Run Tier 1 frequently** - It's fast and catches most issues
2. **Run Tier 2 before merging** - Deep review prevents bugs
3. **Use caching** - Saves time and money
4. **Fix critical issues immediately** - Don't let them accumulate
5. **Review warnings regularly** - They become errors eventually
6. **Customize prompts** - Tailor LLM reviews to your domain
7. **Integrate with CI/CD** - Automate quality checks
8. **Monitor costs** - Track LLM usage in production

---

## FAQ

**Q: How accurate is the LLM review?**  
A: Very good for semantic issues, but not perfect. Always combine with testing.

**Q: Can I use a different LLM?**  
A: Yes! Modify `LLMInspector._call_llm()` to use your preferred model.

**Q: Does it work offline?**  
A: Tier 1 yes (fully local). Tier 2 needs API access.

**Q: How much does it cost?**  
A: ~$0.01 per file for LLM review. Tier 1 is free.

**Q: Can it fix issues automatically?**  
A: Not yet, but you can build auto-fix logic on top of the issue reports.

**Q: What about other languages?**  
A: Currently Python only. Extend for other languages by creating language-specific inspectors.

---

## Support

For issues or questions:
- Check this guide first
- Review the code in `src/validation/inspector.py`
- Check LLM prompts in `src/validation/prompts.py`
- File an issue or ask the team
