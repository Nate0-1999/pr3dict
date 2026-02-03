"""
PR3DICT: Main Entry Point

Starts the prediction market trading engine.
"""
import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import TradingEngine, EngineConfig
from src.platforms import KalshiPlatform
from src.strategies import ArbitrageStrategy
from src.risk import RiskManager, RiskConfig
from src.notifications import NotificationManager, load_notification_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("pr3dict")


async def main():
    """Main entry point."""
    # Load environment
    load_dotenv("config/.env")
    
    # Parse args
    import argparse
    parser = argparse.ArgumentParser(description="PR3DICT Trading Engine")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--platform", choices=["kalshi", "polymarket", "all"], default="kalshi")
    args = parser.parse_args()
    
    paper_mode = args.mode == "paper"
    
    # Initialize platforms
    platforms = []
    
    if args.platform in ["kalshi", "all"]:
        kalshi = KalshiPlatform(sandbox=paper_mode)
        platforms.append(kalshi)
    
    if args.platform in ["polymarket", "all"]:
        try:
            from src.platforms import PolymarketPlatform
            polymarket = PolymarketPlatform()
            platforms.append(polymarket)
        except ImportError:
            logger.warning("Polymarket not available (missing py-clob-client)")
    
    if not platforms:
        logger.error("No platforms configured!")
        return
    
    # Initialize strategies
    strategies = [
        ArbitrageStrategy(min_spread=0.025)
    ]
    
    # Initialize risk manager
    risk_config = RiskConfig(
        daily_loss_limit=500,
        max_position_size=100,
        max_portfolio_heat=0.25
    )
    risk_manager = RiskManager(risk_config)
    
    # Initialize notifications (optional)
    notification_config = load_notification_config()
    notifier = None
    if notification_config.telegram_enabled or notification_config.discord_enabled:
        notifier = NotificationManager(notification_config)
        logger.info("Notifications enabled")
    else:
        logger.info("Notifications disabled (set TELEGRAM_ENABLED or DISCORD_ENABLED in .env)")
    
    # Initialize engine
    engine_config = EngineConfig(
        scan_interval_seconds=30,
        max_positions=10,
        paper_mode=paper_mode
    )
    
    engine = TradingEngine(
        platforms=platforms,
        strategies=strategies,
        risk_manager=risk_manager,
        config=engine_config,
        notifications=notifier  # Add notifications
    )
    
    # Run
    try:
        await engine.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested...")
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
