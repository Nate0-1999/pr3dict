#!/usr/bin/env python3
"""
PR3DICT Real-time Monitor

Displays live statistics from the running trading engine.
"""
import asyncio
import os
import sys
from datetime import datetime
from decimal import Decimal
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.platforms import KalshiPlatform
from src.data.cache import DataCache

# ANSI colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'


async def main():
    """Display live stats."""
    load_dotenv("config/.env")
    
    # Connect to cache
    cache = DataCache()
    await cache.connect()
    
    # Connect to Kalshi
    kalshi = KalshiPlatform(sandbox=True)
    connected = await kalshi.connect()
    
    if not connected:
        print(f"{RED}✗ Failed to connect to Kalshi{RESET}")
        return
    
    print(f"{GREEN}╔═══════════════════════════════════════╗{RESET}")
    print(f"{GREEN}║      PR3DICT Live Monitor             ║{RESET}")
    print(f"{GREEN}╚═══════════════════════════════════════╝{RESET}")
    print()
    
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name != 'nt' else 'cls')
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{BLUE}═══ PR3DICT Monitor ═══ {now}{RESET}\n")
            
            # Account stats
            try:
                balance = await kalshi.get_balance()
                positions = await kalshi.get_positions()
                
                print(f"{GREEN}Account (Kalshi Sandbox){RESET}")
                print(f"  Balance:        ${balance:.2f}")
                print(f"  Positions:      {len(positions)}")
                
                if positions:
                    total_value = sum(p.quantity * p.current_price for p in positions)
                    total_pnl = sum(p.unrealized_pnl for p in positions)
                    
                    print(f"  Position Value: ${total_value:.2f}")
                    print(f"  Unrealized P&L: {GREEN if total_pnl >= 0 else RED}${total_pnl:+.2f}{RESET}")
                    print()
                    
                    print(f"{YELLOW}Open Positions:{RESET}")
                    for p in positions:
                        pnl_color = GREEN if p.unrealized_pnl >= 0 else RED
                        print(f"  {p.ticker[:30]:30s} {p.side.value:3s} x{p.quantity:3d} @ ${p.avg_price:.2f} → ${p.current_price:.2f} ({pnl_color}{p.unrealized_pnl:+.2f}{RESET})")
                
                print()
                
            except Exception as e:
                print(f"{RED}Error fetching account data: {e}{RESET}\n")
            
            # Cache stats
            cache_stats = await cache.get_stats()
            if cache_stats.get("enabled"):
                print(f"{BLUE}Cache Stats{RESET}")
                print(f"  Total Keys:     {cache_stats['total_keys']}")
                print(f"  Hit Rate:       {cache_stats['hit_rate']:.1%}")
                print()
            
            # Refresh every 5 seconds
            await asyncio.sleep(5)
            
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Monitor stopped{RESET}")
    finally:
        await kalshi.disconnect()
        await cache.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
