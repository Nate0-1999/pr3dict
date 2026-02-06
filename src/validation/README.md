# PR3DICT Code Inspector

**Hybrid LLM + Hardcoded Code Quality System**

A 3-tier validation framework that combines fast structural checks with deep semantic analysis to ensure code quality in the PR3DICT trading system.

---

## Quick Start

```bash
# Inspect a single file (with LLM)
./inspect.sh file src/strategies/my_strategy.py

# Quick check (no LLM, free)
./inspect.sh fast src/strategies/my_strategy.py

# Inspect all strategies
./inspect.sh strategies

# Run tests
python3 test_inspector.py
```

---

## The 3-Tier Architecture

### Tier 1: Fast Hardcoded Checks ⚡
**Speed:** < 5ms per file | **Cost:** Free

Validates:
- ✅ Python syntax (compilation check)
- ✅ Import resolution and structure
- ✅ File organization
- ✅ Type hints presence
- ✅ Docstring existence
- ✅ Naming conventions
- ✅ Strategy-specific requirements (base class, required methods)

**When to use:** Always! Run on every save, in pre-commit hooks, during development.

### Tier 2: LLM Semantic Review 🤖
**Speed:** 2-10 seconds | **Cost:** ~$0.01/file

Reviews:
- 🧠 Logic correctness and edge cases
- 🧠 Architecture consistency
- 🧠 Security vulnerabilities
- 🧠 Performance anti-patterns
- 🧠 Strategy soundness (market inefficiencies, arbitrage risks)
- 🧠 Risk management adequacy
- 🧠 Integration safety

**When to use:** Before merges, for new features, security-sensitive code, strategy development.

### Tier 3: Runtime Testing 🧪
**Speed:** Varies (30s - 5min) | **Cost:** Infrastructure

Executes:
- 🧪 Unit tests
- 🧪 Integration tests
- 🧪 Quick backtests (for strategies)

**When to use:** Final validation, major releases, debugging test failures.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  InspectionManager                       │
│  (Orchestrates all tiers, handles caching, reporting)   │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────┐
│ Tier 1        │  │ Tier 2       │  │ Tier 3       │
│ Hardcoded     │  │ LLM Review   │  │ Runtime      │
│ Inspector     │  │ Inspector    │  │ Tester       │
├───────────────┤  ├──────────────┤  ├──────────────┤
│ - Syntax      │  │ - Prompts    │  │ - pytest     │
│ - Imports     │  │ - LLM calls  │  │ - Test       │
│ - Structure   │  │ - Caching    │  │   discovery  │
│ - Type hints  │  │ - Parsing    │  │ - Execution  │
│ - Docstrings  │  └──────────────┘  └──────────────┘
└───────────────┘
```

---

## Issue Severity Levels

| Severity | Meaning | Action | Exit Code |
|----------|---------|--------|-----------|
| 🚨 **CRITICAL** | Syntax errors, security holes, unbounded risk | Fix immediately, blocks deployment | 2 |
| ❌ **ERROR** | Logic errors, missing requirements, serious issues | Fix before merge | 1 |
| ⚠️ **WARNING** | Style violations, potential improvements | Review, fix when convenient | 0 |
| ℹ️ **INFO** | Suggestions, optional enhancements | Nice to have | 0 |

---

## Files Structure

```
src/validation/
├── __init__.py                  # Package exports
├── __main__.py                  # CLI entry point
├── inspector.py                 # Core inspection logic (3500+ lines)
├── prompts.py                   # LLM review prompts
├── README.md                    # This file
├── USAGE.md                     # Detailed usage guide
└── SUBAGENT_INTEGRATION.md      # Sub-agent integration patterns

scripts/
├── inspect.sh                   # Convenience wrapper
└── test_inspector.py            # Test suite
```

---

## Usage Examples

### CLI Usage

```bash
# Basic file inspection
python -m src.validation.inspect --file src/strategies/momentum.py

# Skip LLM (fast mode)
python -m src.validation.inspect --file momentum.py --no-llm

# Include runtime tests
python -m src.validation.inspect --file momentum.py --with-tests

# Inspect directory
python -m src.validation.inspect --dir src/strategies/

# Verbose output
python -m src.validation.inspect --file momentum.py -v

# Disable caching
python -m src.validation.inspect --file momentum.py --no-cache
```

### Programmatic Usage

```python
from src.validation import InspectionManager

# Create inspector
manager = InspectionManager(
    enable_llm=True,      # Use LLM semantic review
    enable_testing=False, # Skip runtime tests
    enable_cache=True     # Cache LLM results
)

# Inspect file
result = await manager.inspect_file("src/strategies/my_strategy.py")

# Check results
if result.has_critical_issues():
    print("🚨 Critical issues found!")
    for issue in result.get_critical_issues():
        print(f"  {issue}")

# Get summary
print(result.summary())

# Inspect directory
results = await manager.inspect_directory("src/strategies/")
print(manager.generate_report(results))
```

### In Sub-Agent Workflow

```python
async def generate_and_validate_strategy(spec: dict):
    # Generate code
    code = generate_strategy_code(spec)
    file_path = "src/strategies/generated.py"
    
    with open(file_path, 'w') as f:
        f.write(code)
    
    # Validate
    manager = InspectionManager(enable_llm=True)
    result = await manager.inspect_file(file_path)
    
    if result.has_critical_issues():
        return {
            "status": "failed",
            "issues": [str(i) for i in result.get_critical_issues()]
        }
    
    return {"status": "success", "file": file_path}
```

---

## Features

### 🚀 Fast Tier 1 Checks
- AST-based parsing (no execution)
- Millisecond response times
- Zero cost
- Catches 80% of issues

### 🤖 Smart LLM Review
- Context-aware prompts (different for strategies, risk, execution)
- Catches semantic issues hardcoded checks miss
- Provides actionable suggestions
- Learns from tier 1 results

### 💾 Intelligent Caching
- SHA256 file hashing
- 7-day cache lifetime
- Automatic cache invalidation on file changes
- Saves $$$ on re-inspections

### 📊 Detailed Reporting
- Per-file results
- Multi-file summaries
- Cost tracking
- Performance metrics

### 🔌 Easy Integration
- CLI tool
- Python API
- Shell wrapper
- Sub-agent patterns

---

## Performance

**Tier 1 Benchmarks:**
- Simple file: ~0.5ms
- Complex strategy: ~5ms
- 100 files: ~500ms

**Tier 2 Benchmarks:**
- Small file: ~2s, $0.005
- Large strategy: ~8s, $0.015
- With cache hit: ~0ms, $0.00

**Tier 3 Benchmarks:**
- Unit tests: 5-30s
- Integration tests: 30s-2min
- Backtests: 1-5min

---

## Cost Analysis

### Typical Costs

| Scenario | LLM Calls | Cost | Time |
|----------|-----------|------|------|
| Single file (Tier 1) | 0 | $0.00 | <5ms |
| Single file (Tier 1+2) | 1 | $0.01 | ~3s |
| 10 strategies (Tier 1+2) | 10 | $0.10 | ~30s |
| Full codebase (50 files) | 50 | $0.50 | ~4min |
| With cache (0% change) | 0 | $0.00 | ~250ms |
| With cache (20% change) | 10 | $0.10 | ~1min |

### Cost Optimization

1. **Use Tier 1 first** - Free and catches most issues
2. **Enable caching** - Avoid re-reviewing unchanged files
3. **Batch inspections** - More efficient than one-by-one
4. **Smart triggering** - Only LLM review when needed
5. **CI/CD strategy** - Tier 1 on commits, Tier 2 on PRs

---

## LLM Prompts

Different prompts for different code types:

- **Strategy Review:** Logic, edge cases, risk management, soundness
- **Risk Management:** Position limits, loss tracking, circuit breakers
- **Integration:** Error handling, API safety, rate limits, security
- **Execution:** Order validation, fill handling, race conditions
- **Testing:** Coverage, mock realism, backtest accuracy
- **General:** Logic, security, performance, architecture

See `prompts.py` for full prompt text.

---

## Extending the System

### Add Custom Checks

```python
from src.validation.inspector import HardcodedInspector

class CustomInspector(HardcodedInspector):
    def inspect(self, file_path: str):
        issues = super().inspect(file_path)
        
        # Add your custom checks
        content = Path(file_path).read_text()
        if 'TODO' in content:
            issues.append(ValidationIssue(
                severity=IssueSeverity.INFO,
                category=IssueCategory.DOCUMENTATION,
                message="File contains TODO comments"
            ))
        
        return issues
```

### Customize LLM Prompts

Edit `prompts.py` to add domain-specific review criteria:

```python
CUSTOM_REVIEW_PROMPT = """
You are reviewing prediction market trading code.

Focus on:
1. Binary outcome constraints (YES + NO ≤ $1.00)
2. Time-to-resolution handling
3. Inventory risk management
4. ...
"""

REVIEW_PROMPTS['custom'] = CUSTOM_REVIEW_PROMPT
```

---

## Testing

Run the test suite:

```bash
python3 test_inspector.py
```

Tests cover:
- ✅ Tier 1 validation (syntax, structure, docstrings)
- ✅ Real file inspection
- ✅ Caching functionality
- ✅ Performance benchmarks

**Note:** Tier 2 (LLM) tests require API access and are tested manually.

---

## Troubleshooting

### "Command not found: python"
Use `python3` instead of `python`

### "LLM call failed"
Check OpenClaw is installed: `which openclaw`  
Set API key: `export ANTHROPIC_API_KEY=your-key`

### "pytest not found"
Install: `pip install pytest`

### "File not found"
Check you're in the PR3DICT root directory

### "Too slow"
Use `--no-llm` for faster checks  
Enable caching (default)  
Run Tier 1 only in development

### "Too expensive"
Use Tier 1 for routine checks (free)  
Enable caching (saves 90%+ costs)  
Only run Tier 2 on changed files

---

## Best Practices

1. **Development:** Run Tier 1 frequently (pre-commit hooks)
2. **Code Review:** Run Tier 1+2 before creating PRs
3. **CI/CD:** Tier 1 on every commit, Tier 2 on PRs
4. **Production:** Full validation (Tier 1+2+3) before deployment
5. **Caching:** Keep enabled (default) to save time and money
6. **Monitoring:** Track inspection costs and performance
7. **Customization:** Add domain-specific checks as needed

---

## Roadmap

Future enhancements:
- [ ] Auto-fix for common issues
- [ ] IDE integration (VS Code extension)
- [ ] Git hooks for automatic validation
- [ ] Advanced caching strategies
- [ ] Multi-language support
- [ ] Custom rule DSL
- [ ] Web dashboard for results
- [ ] Historical trend analysis

---

## Contributing

To improve the inspector:

1. **Add checks:** Extend `HardcodedInspector` with new validations
2. **Improve prompts:** Refine LLM prompts in `prompts.py`
3. **Add tests:** Expand `test_inspector.py`
4. **Optimize:** Improve performance or reduce costs
5. **Document:** Update this README and USAGE.md

---

## Support

- **Documentation:** See `USAGE.md` for detailed guide
- **Integration:** See `SUBAGENT_INTEGRATION.md` for sub-agent patterns
- **Code:** Read `inspector.py` for implementation details
- **Prompts:** Check `prompts.py` for LLM review logic

---

## License

Part of the PR3DICT project.

---

**Built with ❤️ for code quality in prediction market trading.**
