"""
PR3DICT Code Inspector - Hybrid LLM + Hardcoded Validation

3-Tier Validation Architecture:
1. Tier 1: Fast Hardcoded Checks (milliseconds, free)
2. Tier 2: LLM Semantic Review (seconds, ~$0.01/file)  
3. Tier 3: Runtime Testing (varies)

Usage:
    from src.validation.inspector import InspectionManager
    
    manager = InspectionManager(enable_llm=True)
    results = await manager.inspect_file("src/strategies/my_strategy.py")
    
    if results.has_critical_issues():
        print("Critical issues found!")
        for issue in results.get_critical_issues():
            print(f"  - {issue}")
"""
import ast
import sys
import os
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import importlib.util
import subprocess

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class IssueSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"  # Blocks deployment
    ERROR = "error"       # Should fix before merge
    WARNING = "warning"   # Should review
    INFO = "info"        # Nice to have


class IssueCategory(Enum):
    """Categories for validation issues."""
    SYNTAX = "syntax"
    IMPORTS = "imports"
    STRUCTURE = "structure"
    TYPE_HINTS = "type_hints"
    DOCUMENTATION = "documentation"
    LOGIC = "logic"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"
    RISK_MANAGEMENT = "risk_management"
    TESTING = "testing"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: IssueSeverity
    category: IssueCategory
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        location = f"Line {self.line_number}" if self.line_number else "File"
        return f"[{self.severity.value.upper()}] {location}: {self.message}"


@dataclass
class InspectionResult:
    """Results from code inspection."""
    file_path: str
    timestamp: datetime
    tier1_issues: List[ValidationIssue] = field(default_factory=list)
    tier2_issues: List[ValidationIssue] = field(default_factory=list)
    tier3_issues: List[ValidationIssue] = field(default_factory=list)
    tier1_duration_ms: float = 0.0
    tier2_duration_ms: float = 0.0
    tier3_duration_ms: float = 0.0
    llm_cost_usd: float = 0.0
    file_hash: Optional[str] = None
    cached: bool = False
    
    def all_issues(self) -> List[ValidationIssue]:
        """Get all issues across all tiers."""
        return self.tier1_issues + self.tier2_issues + self.tier3_issues
    
    def has_critical_issues(self) -> bool:
        """Check if any critical issues exist."""
        return any(issue.severity == IssueSeverity.CRITICAL for issue in self.all_issues())
    
    def has_errors(self) -> bool:
        """Check if any errors exist."""
        return any(
            issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.ERROR] 
            for issue in self.all_issues()
        )
    
    def get_critical_issues(self) -> List[ValidationIssue]:
        """Get only critical issues."""
        return [i for i in self.all_issues() if i.severity == IssueSeverity.CRITICAL]
    
    def get_errors(self) -> List[ValidationIssue]:
        """Get critical and error level issues."""
        return [
            i for i in self.all_issues() 
            if i.severity in [IssueSeverity.CRITICAL, IssueSeverity.ERROR]
        ]
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        issues_by_severity = {}
        for issue in self.all_issues():
            issues_by_severity.setdefault(issue.severity, []).append(issue)
        
        lines = [
            f"Inspection of {self.file_path}",
            f"{'=' * 60}",
            f"Tier 1 (Hardcoded): {self.tier1_duration_ms:.1f}ms",
            f"Tier 2 (LLM): {self.tier2_duration_ms:.1f}ms (${self.llm_cost_usd:.4f})",
            f"Tier 3 (Testing): {self.tier3_duration_ms:.1f}ms",
            f"Total Issues: {len(self.all_issues())}",
        ]
        
        for severity in IssueSeverity:
            count = len(issues_by_severity.get(severity, []))
            if count > 0:
                lines.append(f"  {severity.value}: {count}")
        
        if self.cached:
            lines.append("\n⚡ Results from cache")
        
        return "\n".join(lines)


# ============================================================================
# Tier 1: Fast Hardcoded Checks
# ============================================================================

class HardcodedInspector:
    """
    Tier 1: Fast structural and syntactic validation.
    
    Checks performed:
    - Python syntax validity (compile check)
    - Import resolution
    - File structure (classes, functions)
    - Type hints present
    - Docstrings exist
    - Basic naming conventions
    - Required base class inheritance (for strategies)
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
    
    def inspect(self, file_path: str) -> List[ValidationIssue]:
        """Run all hardcoded checks on a file."""
        issues = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                category=IssueCategory.SYNTAX,
                message=f"File not found: {file_path}"
            ))
            return issues
        
        # Read file content
        try:
            content = file_path.read_text()
        except Exception as e:
            issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                category=IssueCategory.SYNTAX,
                message=f"Cannot read file: {e}"
            ))
            return issues
        
        # 1. Syntax validation
        issues.extend(self._check_syntax(content, file_path))
        
        # If syntax is invalid, stop here
        if any(i.severity == IssueSeverity.CRITICAL for i in issues):
            return issues
        
        # Parse AST for deeper checks
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                category=IssueCategory.SYNTAX,
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno
            ))
            return issues
        
        # 2. Import checks
        issues.extend(self._check_imports(tree, file_path))
        
        # 3. Structure checks
        issues.extend(self._check_structure(tree, content))
        
        # 4. Type hints
        issues.extend(self._check_type_hints(tree))
        
        # 5. Docstrings
        issues.extend(self._check_docstrings(tree, content))
        
        # 6. Strategy-specific checks
        if "strategies" in str(file_path):
            issues.extend(self._check_strategy_requirements(tree, content))
        
        return issues
    
    def _check_syntax(self, content: str, file_path: Path) -> List[ValidationIssue]:
        """Check if Python syntax is valid."""
        issues = []
        
        try:
            compile(content, str(file_path), 'exec')
        except SyntaxError as e:
            issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                category=IssueCategory.SYNTAX,
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno,
                column=e.offset
            ))
        
        return issues
    
    def _check_imports(self, tree: ast.AST, file_path: Path) -> List[ValidationIssue]:
        """Check import statements for issues."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Check for relative imports outside package
                if isinstance(node, ast.ImportFrom) and node.level > 0:
                    # Relative import - ensure we're in a package
                    if not self._is_in_package(file_path):
                        issues.append(ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.IMPORTS,
                            message="Relative import in non-package file",
                            line_number=node.lineno
                        ))
                
                # Check for star imports (bad practice)
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name == '*':
                            issues.append(ValidationIssue(
                                severity=IssueSeverity.WARNING,
                                category=IssueCategory.IMPORTS,
                                message="Avoid star imports (from X import *)",
                                line_number=node.lineno,
                                suggestion="Import specific names explicitly"
                            ))
        
        return issues
    
    def _check_structure(self, tree: ast.AST, content: str) -> List[ValidationIssue]:
        """Check file structure."""
        issues = []
        
        # Check for module docstring
        if not ast.get_docstring(tree):
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.DOCUMENTATION,
                message="Missing module docstring",
                line_number=1,
                suggestion='Add """Module description""" at the top of the file'
            ))
        
        # Count classes and functions
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        
        # Warn if file is too large (>500 lines)
        line_count = content.count('\n') + 1
        if line_count > 500:
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.STRUCTURE,
                message=f"File is large ({line_count} lines). Consider splitting.",
                suggestion="Split into multiple files for better maintainability"
            ))
        
        return issues
    
    def _check_type_hints(self, tree: ast.AST) -> List[ValidationIssue]:
        """Check for presence of type hints."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip dunder methods
                if node.name.startswith('__') and node.name.endswith('__'):
                    continue
                
                # Check return type annotation
                if node.returns is None and node.name != '__init__':
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.INFO,
                        category=IssueCategory.TYPE_HINTS,
                        message=f"Function '{node.name}' missing return type hint",
                        line_number=node.lineno,
                        suggestion="Add -> ReturnType annotation"
                    ))
                
                # Check parameter type annotations
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != 'self' and arg.arg != 'cls':
                        issues.append(ValidationIssue(
                            severity=IssueSeverity.INFO,
                            category=IssueCategory.TYPE_HINTS,
                            message=f"Parameter '{arg.arg}' in '{node.name}' missing type hint",
                            line_number=node.lineno
                        ))
        
        return issues
    
    def _check_docstrings(self, tree: ast.AST, content: str) -> List[ValidationIssue]:
        """Check for docstrings in classes and functions."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not ast.get_docstring(node):
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.DOCUMENTATION,
                        message=f"Class '{node.name}' missing docstring",
                        line_number=node.lineno
                    ))
            
            elif isinstance(node, ast.FunctionDef):
                # Skip private methods for docstring requirement
                if node.name.startswith('_') and not node.name.startswith('__'):
                    continue
                
                if not ast.get_docstring(node):
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.INFO,
                        category=IssueCategory.DOCUMENTATION,
                        message=f"Function '{node.name}' missing docstring",
                        line_number=node.lineno
                    ))
        
        return issues
    
    def _check_strategy_requirements(self, tree: ast.AST, content: str) -> List[ValidationIssue]:
        """Check strategy-specific requirements."""
        issues = []
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                # Check if inherits from TradingStrategy or BaseStrategy
                bases = [self._get_name(base) for base in node.bases]
                
                if 'TradingStrategy' not in bases and 'BaseStrategy' not in bases:
                    if 'Strategy' in node.name:
                        issues.append(ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.STRUCTURE,
                            message=f"Strategy class '{node.name}' should inherit from TradingStrategy",
                            line_number=node.lineno
                        ))
                
                # Check for required methods
                if 'Strategy' in node.name:
                    methods = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}
                    required_methods = {'scan_markets', 'check_exit'}
                    
                    for required in required_methods:
                        if required not in methods:
                            issues.append(ValidationIssue(
                                severity=IssueSeverity.ERROR,
                                category=IssueCategory.STRUCTURE,
                                message=f"Strategy missing required method: {required}",
                                line_number=node.lineno
                            ))
        
        return issues
    
    def _is_in_package(self, file_path: Path) -> bool:
        """Check if file is part of a package."""
        return (file_path.parent / '__init__.py').exists()
    
    def _get_name(self, node: ast.AST) -> str:
        """Extract name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""


# ============================================================================
# Tier 2: LLM Semantic Review
# ============================================================================

class LLMInspector:
    """
    Tier 2: Deep semantic analysis using LLM.
    
    Reviews:
    - Logic correctness
    - Edge case handling
    - Architecture consistency
    - Security vulnerabilities
    - Performance anti-patterns
    - Strategy soundness (for trading strategies)
    - Risk management adequacy
    """
    
    def __init__(self, enable_cache: bool = True, cache_dir: Path = None):
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir or (Path.home() / '.openclaw' / 'cache' / 'pr3dict_inspector')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def inspect(self, 
                      file_path: str, 
                      file_hash: str,
                      tier1_issues: List[ValidationIssue]) -> Tuple[List[ValidationIssue], float]:
        """
        Run LLM-based semantic review.
        
        Returns:
            (issues, cost_usd)
        """
        file_path = Path(file_path)
        
        # Check cache first
        if self.enable_cache:
            cached = self._get_cached_result(file_path, file_hash)
            if cached:
                logger.info(f"Using cached LLM review for {file_path}")
                return cached, 0.0
        
        # Read file content
        content = file_path.read_text()
        
        # Determine review type based on file path
        review_type = self._determine_review_type(file_path)
        
        # Get appropriate prompt
        prompt = self._get_review_prompt(review_type, content, tier1_issues)
        
        # Call LLM (using OpenClaw's configured model)
        issues, cost = await self._call_llm(prompt, file_path)
        
        # Cache result
        if self.enable_cache:
            self._cache_result(file_path, file_hash, issues)
        
        return issues, cost
    
    def _determine_review_type(self, file_path: Path) -> str:
        """Determine what type of review to perform."""
        path_str = str(file_path).lower()
        
        if 'strategies' in path_str:
            return 'strategy'
        elif 'risk' in path_str:
            return 'risk_management'
        elif 'execution' in path_str or 'engine' in path_str:
            return 'execution'
        elif 'platforms' in path_str:
            return 'integration'
        elif 'backtest' in path_str:
            return 'testing'
        else:
            return 'general'
    
    def _get_review_prompt(self, 
                           review_type: str, 
                           content: str, 
                           tier1_issues: List[ValidationIssue]) -> str:
        """Get the appropriate review prompt for this code."""
        from .prompts import REVIEW_PROMPTS
        
        base_prompt = REVIEW_PROMPTS.get(review_type, REVIEW_PROMPTS['general'])
        
        # Add tier 1 context
        tier1_summary = ""
        if tier1_issues:
            tier1_summary = "\n\nTier 1 (Hardcoded) Issues Found:\n"
            for issue in tier1_issues[:10]:  # Limit to 10
                tier1_summary += f"- [{issue.severity.value}] {issue.message}\n"
        
        full_prompt = f"""{base_prompt}

{tier1_summary}

Code to review:
```python
{content}
```

Please provide your review as a JSON array of issues with this structure:
[
  {{
    "severity": "critical|error|warning|info",
    "category": "logic|security|performance|architecture|risk_management",
    "message": "Description of the issue",
    "line_number": 42,  // if applicable
    "suggestion": "How to fix it"
  }}
]

Focus on semantic issues that can't be caught by static analysis. Be thorough but concise.
"""
        return full_prompt
    
    async def _call_llm(self, prompt: str, file_path: Path) -> Tuple[List[ValidationIssue], float]:
        """
        Call LLM for code review.
        
        This uses OpenClaw's configured model via subprocess.
        """
        issues = []
        cost = 0.0
        
        try:
            # Create a temporary file with the prompt
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(prompt)
                prompt_file = f.name
            
            # Call OpenClaw CLI to get LLM response
            # Using subprocess to invoke the LLM through OpenClaw
            result = subprocess.run(
                ['openclaw', 'chat', '--file', prompt_file, '--json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            os.unlink(prompt_file)
            
            if result.returncode != 0:
                logger.error(f"LLM call failed: {result.stderr}")
                return issues, cost
            
            # Parse JSON response
            try:
                response = json.loads(result.stdout)
                llm_issues = response.get('issues', [])
                
                for item in llm_issues:
                    issues.append(ValidationIssue(
                        severity=IssueSeverity(item.get('severity', 'info')),
                        category=IssueCategory(item.get('category', 'logic')),
                        message=item['message'],
                        line_number=item.get('line_number'),
                        suggestion=item.get('suggestion')
                    ))
                
                # Estimate cost: ~$0.01 per file (rough estimate)
                cost = 0.01
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response: {e}")
                # Try to extract issues from text
                issues = self._parse_text_response(result.stdout)
                cost = 0.01
        
        except subprocess.TimeoutExpired:
            logger.error("LLM call timed out")
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.LOGIC,
                message="LLM review timed out - manual review recommended"
            ))
        
        except Exception as e:
            logger.error(f"LLM call error: {e}")
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.LOGIC,
                message=f"LLM review failed: {e}"
            ))
        
        return issues, cost
    
    def _parse_text_response(self, text: str) -> List[ValidationIssue]:
        """Fallback: parse plain text response."""
        issues = []
        
        # Simple parsing - look for bullet points or numbered lists
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('- ', '* ', '1.', '2.', '3.')):
                # Extract issue
                message = line.lstrip('- *123456789.')
                if len(message) > 10:  # Meaningful message
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.LOGIC,
                        message=message
                    ))
        
        return issues
    
    def _get_cached_result(self, file_path: Path, file_hash: str) -> Optional[List[ValidationIssue]]:
        """Get cached LLM review result."""
        cache_file = self.cache_dir / f"{file_hash}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            data = json.loads(cache_file.read_text())
            
            # Check if cache is fresh (< 7 days old)
            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time > timedelta(days=7):
                return None
            
            # Reconstruct issues
            issues = []
            for item in data['issues']:
                issues.append(ValidationIssue(
                    severity=IssueSeverity(item['severity']),
                    category=IssueCategory(item['category']),
                    message=item['message'],
                    line_number=item.get('line_number'),
                    suggestion=item.get('suggestion')
                ))
            
            return issues
        
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _cache_result(self, file_path: Path, file_hash: str, issues: List[ValidationIssue]):
        """Cache LLM review result."""
        cache_file = self.cache_dir / f"{file_hash}.json"
        
        data = {
            'file_path': str(file_path),
            'file_hash': file_hash,
            'timestamp': datetime.now().isoformat(),
            'issues': [
                {
                    'severity': issue.severity.value,
                    'category': issue.category.value,
                    'message': issue.message,
                    'line_number': issue.line_number,
                    'suggestion': issue.suggestion
                }
                for issue in issues
            ]
        }
        
        cache_file.write_text(json.dumps(data, indent=2))


# ============================================================================
# Tier 3: Runtime Testing
# ============================================================================

class RuntimeTester:
    """
    Tier 3: Runtime validation through testing.
    
    Runs:
    - Unit tests for the file
    - Integration tests if available
    - Quick backtests for strategies
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
    
    def inspect(self, file_path: str) -> List[ValidationIssue]:
        """Run runtime tests."""
        issues = []
        file_path = Path(file_path)
        
        # Find corresponding test file
        test_file = self._find_test_file(file_path)
        
        if not test_file:
            issues.append(ValidationIssue(
                severity=IssueSeverity.INFO,
                category=IssueCategory.TESTING,
                message="No test file found for this module",
                suggestion=f"Create tests/test_{file_path.stem}.py"
            ))
            return issues
        
        # Run tests
        test_results = self._run_tests(test_file)
        
        if test_results['failed'] > 0:
            issues.append(ValidationIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.TESTING,
                message=f"{test_results['failed']} tests failed",
                suggestion="Fix failing tests before deployment"
            ))
        
        if test_results['passed'] == 0 and test_results['failed'] == 0:
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.TESTING,
                message="No tests executed",
                suggestion="Add test cases"
            ))
        
        return issues
    
    def _find_test_file(self, file_path: Path) -> Optional[Path]:
        """Find the test file for this module."""
        # Check tests/ directory
        test_dir = self.project_root / 'tests'
        test_file = test_dir / f"test_{file_path.stem}.py"
        
        if test_file.exists():
            return test_file
        
        # Check same directory
        test_file = file_path.parent / f"test_{file_path.stem}.py"
        if test_file.exists():
            return test_file
        
        return None
    
    def _run_tests(self, test_file: Path) -> Dict[str, int]:
        """Run pytest on test file."""
        try:
            result = subprocess.run(
                ['pytest', str(test_file), '--tb=short', '-v'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.project_root)
            )
            
            # Parse output for pass/fail counts
            output = result.stdout
            
            passed = output.count(' PASSED')
            failed = output.count(' FAILED')
            
            return {'passed': passed, 'failed': failed}
        
        except subprocess.TimeoutExpired:
            logger.warning(f"Tests timed out for {test_file}")
            return {'passed': 0, 'failed': 0}
        
        except FileNotFoundError:
            logger.warning("pytest not found - skipping runtime tests")
            return {'passed': 0, 'failed': 0}
        
        except Exception as e:
            logger.error(f"Test execution error: {e}")
            return {'passed': 0, 'failed': 0}


# ============================================================================
# Inspection Manager - Orchestrates All Tiers
# ============================================================================

class InspectionManager:
    """
    Orchestrates all three tiers of validation.
    
    Usage:
        manager = InspectionManager(enable_llm=True, enable_testing=False)
        result = await manager.inspect_file("src/strategies/my_strategy.py")
        
        if result.has_critical_issues():
            print("❌ Critical issues found!")
            for issue in result.get_critical_issues():
                print(f"  {issue}")
    """
    
    def __init__(self, 
                 enable_llm: bool = True,
                 enable_testing: bool = False,
                 enable_cache: bool = True,
                 project_root: Path = None):
        self.enable_llm = enable_llm
        self.enable_testing = enable_testing
        self.project_root = project_root or Path.cwd()
        
        self.tier1 = HardcodedInspector(project_root=self.project_root)
        self.tier2 = LLMInspector(enable_cache=enable_cache) if enable_llm else None
        self.tier3 = RuntimeTester(project_root=self.project_root) if enable_testing else None
    
    async def inspect_file(self, file_path: str) -> InspectionResult:
        """
        Run complete inspection on a file.
        
        Returns:
            InspectionResult with all findings
        """
        import time
        
        file_path = Path(file_path)
        result = InspectionResult(
            file_path=str(file_path),
            timestamp=datetime.now()
        )
        
        # Calculate file hash for caching
        result.file_hash = self._calculate_file_hash(file_path)
        
        # Tier 1: Fast hardcoded checks (always run)
        start = time.time()
        result.tier1_issues = self.tier1.inspect(str(file_path))
        result.tier1_duration_ms = (time.time() - start) * 1000
        
        logger.info(f"Tier 1 complete: {len(result.tier1_issues)} issues in {result.tier1_duration_ms:.1f}ms")
        
        # Tier 2: LLM semantic review (if enabled and no critical tier1 issues)
        if self.tier2 and not any(i.severity == IssueSeverity.CRITICAL for i in result.tier1_issues):
            start = time.time()
            result.tier2_issues, result.llm_cost_usd = await self.tier2.inspect(
                str(file_path),
                result.file_hash,
                result.tier1_issues
            )
            result.tier2_duration_ms = (time.time() - start) * 1000
            
            logger.info(f"Tier 2 complete: {len(result.tier2_issues)} issues in {result.tier2_duration_ms:.1f}ms (${result.llm_cost_usd:.4f})")
        
        # Tier 3: Runtime testing (if enabled)
        if self.tier3:
            start = time.time()
            result.tier3_issues = self.tier3.inspect(str(file_path))
            result.tier3_duration_ms = (time.time() - start) * 1000
            
            logger.info(f"Tier 3 complete: {len(result.tier3_issues)} issues in {result.tier3_duration_ms:.1f}ms")
        
        return result
    
    async def inspect_directory(self, directory: str, pattern: str = "**/*.py") -> List[InspectionResult]:
        """Inspect all Python files in a directory."""
        directory = Path(directory)
        results = []
        
        for file_path in directory.glob(pattern):
            if self._should_skip(file_path):
                continue
            
            logger.info(f"Inspecting {file_path}")
            result = await self.inspect_file(str(file_path))
            results.append(result)
        
        return results
    
    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = [
            '__pycache__',
            '.pyc',
            '__init__.py',
            'test_',
            '.git',
            'venv',
            '.venv'
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in skip_patterns)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content."""
        return hashlib.sha256(file_path.read_bytes()).hexdigest()
    
    def generate_report(self, results: List[InspectionResult]) -> str:
        """Generate a summary report for multiple inspections."""
        total_files = len(results)
        total_issues = sum(len(r.all_issues()) for r in results)
        total_critical = sum(len(r.get_critical_issues()) for r in results)
        total_errors = sum(len(r.get_errors()) for r in results)
        total_cost = sum(r.llm_cost_usd for r in results)
        
        lines = [
            "=" * 70,
            "PR3DICT Code Inspection Report",
            "=" * 70,
            f"Files Inspected: {total_files}",
            f"Total Issues: {total_issues}",
            f"  Critical: {total_critical}",
            f"  Errors: {total_errors}",
            f"LLM Review Cost: ${total_cost:.4f}",
            "",
            "Files with Critical Issues:",
        ]
        
        files_with_critical = [r for r in results if r.has_critical_issues()]
        if files_with_critical:
            for result in files_with_critical:
                lines.append(f"  ❌ {result.file_path}")
                for issue in result.get_critical_issues():
                    lines.append(f"     - {issue.message}")
        else:
            lines.append("  ✅ None")
        
        lines.extend([
            "",
            "Files with Errors:",
        ])
        
        files_with_errors = [r for r in results if r.has_errors() and not r.has_critical_issues()]
        if files_with_errors:
            for result in files_with_errors:
                lines.append(f"  ⚠️  {result.file_path}")
        else:
            lines.append("  ✅ None")
        
        return "\n".join(lines)
