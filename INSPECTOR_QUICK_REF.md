# PR3DICT Code Inspector - Quick Reference Card

**3-Tier Hybrid Validation: Fast Hardcoded + LLM Semantic + Runtime Testing**

---

## ⚡ Quick Commands

```bash
# Single file (with LLM)
./inspect.sh file src/strategies/my_strategy.py

# Fast check (no LLM, free)
./inspect.sh fast src/strategies/my_strategy.py

# All strategies
./inspect.sh strategies

# Full codebase
./inspect.sh full

# With verbose output
./inspect.sh file my_strategy.py -v
```

---

## 🎯 The 3 Tiers

| Tier | Speed | Cost | What It Checks |
|------|-------|------|----------------|
| 1️⃣ Fast Hardcoded | <5ms | $0.00 | Syntax, imports, structure, type hints, docstrings |
| 2️⃣ LLM Semantic | ~3s | $0.01 | Logic, edge cases, security, performance, strategy soundness |
| 3️⃣ Runtime Tests | 30s-5m | Infra | Unit tests, integration tests, backtests |

---

## 🚦 Issue Severity

| Level | Icon | Meaning | Exit Code |
|-------|------|---------|-----------|
| **CRITICAL** | 🚨 | Blocks deployment | 2 |
| **ERROR** | ❌ | Fix before merge | 1 |
| **WARNING** | ⚠️ | Should review | 0 |
| **INFO** | ℹ️ | Nice to have | 0 |

---

## 📖 When to Use What

### Use Tier 1 Only (`--no-llm`)
- ✅ Active development
- ✅ Pre-commit hooks
- ✅ CI on every commit
- ✅ Quick sanity checks

### Use Tier 1 + 2 (default)
- ✅ Before PRs
- ✅ New features
- ✅ Refactoring
- ✅ Security-sensitive code

### Use All Tiers (`--with-tests`)
- ✅ Pre-deployment
- ✅ Major releases
- ✅ Final validation

---

## 💻 Python API

```python
from src.validation import InspectionManager

# Create manager
manager = InspectionManager(
    enable_llm=True,      # Use LLM review
    enable_testing=False, # Skip tests
    enable_cache=True     # Cache results
)

# Inspect file
result = await manager.inspect_file("path/to/file.py")

# Check results
if result.has_critical_issues():
    for issue in result.get_critical_issues():
        print(f"🚨 {issue}")

# Summary
print(result.summary())
```

---

## 🔧 Sub-Agent Integration

```python
async def generate_and_validate(spec):
    # Generate code
    code = generate_code(spec)
    file_path.write_text(code)
    
    # Validate
    manager = InspectionManager(enable_llm=True)
    result = await manager.inspect_file(str(file_path))
    
    # Handle results
    if result.has_critical_issues():
        return {"status": "failed", "issues": [...]}
    return {"status": "success"}
```

---

## 💰 Cost Guide

| Scenario | Calls | Cost | Time |
|----------|-------|------|------|
| 1 file (Tier 1) | 0 | $0.00 | <5ms |
| 1 file (Tier 1+2) | 1 | $0.01 | ~3s |
| 10 files (Tier 1+2) | 10 | $0.10 | ~30s |
| 50 files (cached 90%) | 5 | $0.05 | ~30s |

**Optimization:** Enable caching (default) to save 90%+ on costs

---

## 📁 File Locations

```
pr3dict/
├── src/validation/
│   ├── inspector.py              # Core system
│   ├── prompts.py                # LLM prompts
│   └── README.md, USAGE.md       # Docs
├── inspect.sh                    # CLI wrapper
└── test_inspector.py             # Tests
```

---

## 🔌 Exit Codes

```bash
./inspect.sh file my_strategy.py
echo $?

# 0 = Pass (no critical/errors)
# 1 = Errors found (fix before merge)
# 2 = Critical issues (blocks deployment)
```

---

## 🎨 CLI Options

```bash
python -m src.validation.inspect \
    --file path/to/file.py \      # Single file
    --dir path/to/dir/ \          # Directory
    --no-llm \                    # Skip LLM (fast)
    --with-tests \                # Include Tier 3
    --no-cache \                  # Disable caching
    -v                            # Verbose output
```

---

## 🐛 Common Issues

**"LLM call failed"**
→ Check: `which openclaw` and `echo $ANTHROPIC_API_KEY`

**"pytest not found"**
→ Install: `pip install pytest`

**"Too slow"**
→ Use `--no-llm` or enable caching

**"Too expensive"**
→ Use Tier 1 only for routine checks

---

## 📚 Documentation

- `README.md` - System overview
- `USAGE.md` - Detailed guide (300+ lines)
- `SUBAGENT_INTEGRATION.md` - Integration patterns
- `CODE_INSPECTOR_IMPLEMENTATION.md` - Complete spec

---

## ✅ Best Practices

1. **Run Tier 1 frequently** (it's free!)
2. **Enable caching** (saves $$)
3. **Tier 2 before merging** (catch semantic issues)
4. **Batch inspections** (more efficient)
5. **Fix critical issues immediately**

---

## 🧪 Test It

```bash
# Run test suite
python3 test_inspector.py

# Quick test
./inspect.sh fast src/strategies/market_making.py
```

---

## 🎯 Typical Workflow

```
1. Write code
2. ./inspect.sh fast <file>     # Quick check
3. Fix any critical issues
4. ./inspect.sh file <file>     # Full review
5. Address errors/warnings
6. Commit & push
7. CI runs Tier 1 on all files
8. PR triggers Tier 1+2 on changed files
```

---

**Quick Start:**
```bash
./inspect.sh file src/strategies/my_strategy.py
```

**Location:** `~/.openclaw/workspace/pr3dict/src/validation/`  
**Status:** ✅ Ready to use
