"""
CLI for PR3DICT Code Inspector

Usage:
    python -m src.validation.inspect --file src/strategies/my_strategy.py
    python -m src.validation.inspect --dir src/strategies/
    python -m src.validation.inspect --file strategy.py --no-llm
    python -m src.validation.inspect --file strategy.py --with-tests
"""
import asyncio
import argparse
import sys
import logging
from pathlib import Path

from .inspector import InspectionManager, IssueSeverity


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_result(result, verbose: bool = False):
    """Print inspection result in a readable format."""
    print("\n" + "=" * 70)
    print(result.summary())
    print("=" * 70)
    
    if not result.all_issues():
        print("\n✅ No issues found!")
        return
    
    # Group issues by severity
    critical = result.get_critical_issues()
    errors = [i for i in result.all_issues() 
              if i.severity == IssueSeverity.ERROR]
    warnings = [i for i in result.all_issues() 
                if i.severity == IssueSeverity.WARNING]
    infos = [i for i in result.all_issues() 
             if i.severity == IssueSeverity.INFO]
    
    if critical:
        print("\n🚨 CRITICAL ISSUES:")
        for issue in critical:
            print(f"\n  {issue}")
            if issue.suggestion:
                print(f"  💡 Suggestion: {issue.suggestion}")
            if verbose and issue.code_snippet:
                print(f"  Code: {issue.code_snippet}")
    
    if errors:
        print("\n❌ ERRORS:")
        for issue in errors:
            print(f"\n  {issue}")
            if issue.suggestion:
                print(f"  💡 Suggestion: {issue.suggestion}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for issue in warnings[:10]:  # Limit to first 10
            print(f"  {issue}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
    
    if verbose and infos:
        print(f"\nℹ️  INFO ({len(infos)}):")
        for issue in infos[:5]:
            print(f"  {issue}")
        if len(infos) > 5:
            print(f"  ... and {len(infos) - 5} more")


async def inspect_file(args):
    """Inspect a single file."""
    file_path = Path(args.file)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return 1
    
    print(f"🔍 Inspecting {file_path}...")
    
    manager = InspectionManager(
        enable_llm=args.llm,
        enable_testing=args.tests,
        enable_cache=not args.no_cache,
        project_root=Path.cwd()
    )
    
    result = await manager.inspect_file(str(file_path))
    
    print_result(result, verbose=args.verbose)
    
    # Exit code based on severity
    if result.has_critical_issues():
        return 2
    elif result.has_errors():
        return 1
    else:
        return 0


async def inspect_directory(args):
    """Inspect all files in a directory."""
    dir_path = Path(args.dir)
    
    if not dir_path.exists():
        print(f"❌ Directory not found: {dir_path}")
        return 1
    
    print(f"🔍 Inspecting directory {dir_path}...")
    
    manager = InspectionManager(
        enable_llm=args.llm,
        enable_testing=args.tests,
        enable_cache=not args.no_cache,
        project_root=Path.cwd()
    )
    
    pattern = args.pattern or "**/*.py"
    results = await manager.inspect_directory(str(dir_path), pattern=pattern)
    
    if not results:
        print("No Python files found to inspect.")
        return 0
    
    # Print individual results
    if args.verbose:
        for result in results:
            print_result(result, verbose=True)
            print()
    
    # Print summary report
    print("\n" + "=" * 70)
    print(manager.generate_report(results))
    print("=" * 70)
    
    # Exit code based on worst issue
    has_critical = any(r.has_critical_issues() for r in results)
    has_errors = any(r.has_errors() for r in results)
    
    if has_critical:
        return 2
    elif has_errors:
        return 1
    else:
        return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PR3DICT Code Inspector - Hybrid LLM + Hardcoded Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a single file with LLM review
  python -m src.validation.inspect --file src/strategies/my_strategy.py
  
  # Inspect without LLM (fast, free)
  python -m src.validation.inspect --file strategy.py --no-llm
  
  # Inspect with runtime tests
  python -m src.validation.inspect --file strategy.py --with-tests
  
  # Inspect entire directory
  python -m src.validation.inspect --dir src/strategies/
  
  # Inspect with verbose output
  python -m src.validation.inspect --file strategy.py -v

Exit Codes:
  0 - No critical issues or errors
  1 - Errors found
  2 - Critical issues found
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file', '-f',
        help='File to inspect'
    )
    input_group.add_argument(
        '--dir', '-d',
        help='Directory to inspect'
    )
    
    # Validation tier options
    parser.add_argument(
        '--no-llm',
        dest='llm',
        action='store_false',
        default=True,
        help='Disable LLM semantic review (Tier 2) - faster, free'
    )
    parser.add_argument(
        '--with-tests',
        dest='tests',
        action='store_true',
        default=False,
        help='Enable runtime testing (Tier 3) - slower'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable LLM result caching'
    )
    
    # Directory options
    parser.add_argument(
        '--pattern',
        help='File pattern for directory inspection (default: **/*.py)'
    )
    
    # Output options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Run inspection
    try:
        if args.file:
            exit_code = asyncio.run(inspect_file(args))
        else:
            exit_code = asyncio.run(inspect_directory(args))
        
        sys.exit(exit_code)
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Inspection cancelled by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Inspection failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
