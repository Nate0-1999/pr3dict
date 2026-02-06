#!/bin/bash
# Convenience wrapper for PR3DICT Code Inspector

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Help text
show_help() {
    cat << EOF
PR3DICT Code Inspector - Quick Commands

Usage: ./inspect.sh [command] [options]

Commands:
  file <path>              Inspect a single file (with LLM)
  fast <path>              Quick inspect (no LLM, fast & free)
  dir <path>               Inspect directory (with LLM)
  strategies               Inspect all strategies
  full                     Inspect entire src/ directory

Options:
  --no-cache               Disable LLM result caching
  --with-tests             Include runtime tests (Tier 3)
  -v, --verbose            Verbose output

Examples:
  ./inspect.sh file src/strategies/momentum.py
  ./inspect.sh fast src/strategies/momentum.py
  ./inspect.sh strategies
  ./inspect.sh dir src/strategies/ -v

Exit codes:
  0 - No critical issues or errors
  1 - Errors found
  2 - Critical issues found
EOF
}

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: Run this script from the PR3DICT root directory${NC}"
    exit 1
fi

# Parse command
COMMAND="$1"
shift

case "$COMMAND" in
    file)
        if [ -z "$1" ]; then
            echo -e "${RED}Error: No file specified${NC}"
            echo "Usage: ./inspect.sh file <path>"
            exit 1
        fi
        echo -e "${GREEN}🔍 Inspecting $1 with LLM review...${NC}"
        python -m src.validation.inspect --file "$@"
        ;;
    
    fast)
        if [ -z "$1" ]; then
            echo -e "${RED}Error: No file specified${NC}"
            echo "Usage: ./inspect.sh fast <path>"
            exit 1
        fi
        echo -e "${GREEN}⚡ Quick inspect (no LLM): $1${NC}"
        python -m src.validation.inspect --file "$1" --no-llm "${@:2}"
        ;;
    
    dir)
        if [ -z "$1" ]; then
            echo -e "${RED}Error: No directory specified${NC}"
            echo "Usage: ./inspect.sh dir <path>"
            exit 1
        fi
        echo -e "${GREEN}🔍 Inspecting directory $1 with LLM review...${NC}"
        python -m src.validation.inspect --dir "$@"
        ;;
    
    strategies)
        echo -e "${GREEN}🔍 Inspecting all strategies...${NC}"
        python -m src.validation.inspect --dir src/strategies/ "$@"
        ;;
    
    full)
        echo -e "${GREEN}🔍 Full inspection of src/ directory...${NC}"
        echo -e "${YELLOW}⚠️  This may take several minutes and cost ~\$0.50${NC}"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python -m src.validation.inspect --dir src/ "$@"
        else
            echo "Cancelled."
            exit 0
        fi
        ;;
    
    help|--help|-h)
        show_help
        exit 0
        ;;
    
    *)
        echo -e "${RED}Error: Unknown command '$COMMAND'${NC}"
        echo "Run './inspect.sh help' for usage information"
        exit 1
        ;;
esac

EXIT_CODE=$?

# Print status based on exit code
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}✅ All checks passed!${NC}"
elif [ $EXIT_CODE -eq 1 ]; then
    echo -e "\n${YELLOW}⚠️  Errors found - review recommended${NC}"
elif [ $EXIT_CODE -eq 2 ]; then
    echo -e "\n${RED}🚨 Critical issues found - fix required${NC}"
fi

exit $EXIT_CODE
