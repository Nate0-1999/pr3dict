#!/bin/bash
# PR3DICT Startup Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         PR3DICT Trading Engine        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"
echo

# Check .env
if [ ! -f "config/.env" ]; then
    echo -e "${RED}✗ config/.env not found!${NC}"
    echo "  Copy config/example.env to config/.env and configure"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found!${NC}"
    exit 1
fi

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
python3 -c "import httpx, dotenv, redis" 2>/dev/null || {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip3 install -r requirements.txt
}

echo -e "${GREEN}✓ Dependencies OK${NC}"
echo

# Parse arguments (default: paper mode, kalshi only)
MODE="${1:-paper}"
PLATFORM="${2:-kalshi}"

echo -e "${GREEN}Starting PR3DICT:${NC}"
echo "  Mode: $MODE"
echo "  Platform: $PLATFORM"
echo

# Run
python3 -m src.engine.main --mode "$MODE" --platform "$PLATFORM"
