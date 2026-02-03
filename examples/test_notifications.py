#!/usr/bin/env python3
"""
PR3DICT: Notification System Test

Test script to verify Telegram and Discord notifications are working.
Sends sample alerts to configured channels.

Usage:
    python examples/test_notifications.py
"""
import asyncio
import logging
import os
from dotenv import load_dotenv

from src.notifications import NotificationManager, NotificationConfig, NotificationLevel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Test notification system."""
    
    # Load environment variables
    load_dotenv()
    
    # Create config from environment
    config = NotificationConfig(
        telegram_enabled=os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        discord_enabled=os.getenv("DISCORD_ENABLED", "false").lower() == "true",
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL"),
        discord_username=os.getenv("DISCORD_USERNAME", "PR3DICT Bot"),
        min_notification_level=NotificationLevel.INFO,
        enable_trade_alerts=True,
        enable_signal_alerts=True,
        enable_risk_alerts=True,
        enable_daily_summary=True
    )
    
    # Check if any channel is enabled
    if not config.telegram_enabled and not config.discord_enabled:
        logger.error("No notification channels enabled!")
        logger.info("Set TELEGRAM_ENABLED=true or DISCORD_ENABLED=true in .env")
        return
    
    logger.info("Testing PR3DICT Notification System")
    logger.info(f"Telegram: {'✓' if config.telegram_enabled else '✗'}")
    logger.info(f"Discord: {'✓' if config.discord_enabled else '✗'}")
    print()
    
    # Initialize notification manager
    notifier = NotificationManager(config)
    
    # Connect
    logger.info("Connecting to notification channels...")
    if not await notifier.connect():
        logger.error("Failed to connect to any notification channel")
        return
    
    logger.info("✓ Connected successfully!")
    print()
    
    try:
        # Test 1: Trading Signal
        logger.info("Test 1: Sending trading signal...")
        await notifier.send_signal(
            ticker="TRUMP-2024-WINNER",
            side="YES",
            price=0.643,
            size=50,
            reason="Arbitrage spread 3.2% detected",
            confidence=0.875,
            strategy="arbitrage"
        )
        await asyncio.sleep(2)
        
        # Test 2: Order Filled
        logger.info("Test 2: Sending order filled notification...")
        await notifier.send_order_placed(
            ticker="TRUMP-2024-WINNER",
            side="YES",
            price=0.645,
            size=50,
            order_id="test_order_123",
            platform="Polymarket"
        )
        await asyncio.sleep(2)
        
        # Test 3: Position Closed (Profit)
        logger.info("Test 3: Sending profitable exit...")
        await notifier.send_position_closed(
            ticker="TRUMP-2024-WINNER",
            pnl=12.50,
            pnl_pct=0.039,
            hold_time="2h 15m",
            reason="Spread closed",
            entry_price=0.645,
            exit_price=0.670
        )
        await asyncio.sleep(2)
        
        # Test 4: Position Closed (Loss)
        logger.info("Test 4: Sending losing exit...")
        await notifier.send_position_closed(
            ticker="BTC-100K-2024",
            pnl=-8.75,
            pnl_pct=-0.025,
            hold_time="1h 30m",
            reason="Stop loss triggered",
            entry_price=0.350,
            exit_price=0.325
        )
        await asyncio.sleep(2)
        
        # Test 5: Risk Alert
        logger.info("Test 5: Sending risk alert...")
        await notifier.send_risk_alert(
            alert_type="PORTFOLIO_HEAT_HIGH",
            details="Portfolio heat at 22% (limit: 25%)",
            severity="WARNING"
        )
        await asyncio.sleep(2)
        
        # Test 6: Daily Summary
        logger.info("Test 6: Sending daily summary...")
        await notifier.send_daily_summary(
            trades=12,
            pnl=127.50,
            win_rate=0.667,
            wins=8,
            losses=4,
            best_trade="ETH-2500-EOY (+$35.20)",
            worst_trade="BTC-100K-2024 (-$18.75)"
        )
        await asyncio.sleep(2)
        
        # Test 7: Error Alert
        logger.info("Test 7: Sending error notification...")
        await notifier.send_error(
            error_msg="Test error: Connection timeout",
            context="Market: TEST-MARKET, Side: YES",
            traceback="  File test.py, line 1, in test_function\n    raise Exception('Test')"
        )
        await asyncio.sleep(2)
        
        # Test 8: Engine Status
        logger.info("Test 8: Sending engine status...")
        await notifier.send_engine_status(
            status="RUNNING",
            uptime="5h 23m",
            cycle_count=642
        )
        
        print()
        logger.info("✓ All tests completed successfully!")
        logger.info("Check your Telegram/Discord for notifications")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    
    finally:
        # Cleanup
        logger.info("Disconnecting...")
        await notifier.disconnect()
        logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
