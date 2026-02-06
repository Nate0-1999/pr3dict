"""
PR3DICT: Notification Configuration Loader

Load notification settings from environment variables.
"""
import os
from typing import Optional
from datetime import time

from .manager import NotificationConfig, NotificationLevel


def load_notification_config() -> NotificationConfig:
    """
    Load notification configuration from environment variables.
    
    Expected environment variables:
    - TELEGRAM_ENABLED (bool)
    - TELEGRAM_BOT_TOKEN (string)
    - TELEGRAM_CHAT_ID (string)
    - DISCORD_ENABLED (bool)
    - DISCORD_WEBHOOK_URL (string)
    - DISCORD_USERNAME (string, default: "PR3DICT Bot")
    - NOTIFY_MIN_LEVEL (string, default: "INFO")
    - NOTIFY_SIGNALS (bool, default: true)
    - NOTIFY_TRADES (bool, default: true)
    - NOTIFY_RISK_ALERTS (bool, default: true)
    - NOTIFY_DAILY_SUMMARY (bool, default: true)
    - DAILY_SUMMARY_TIME (string, default: "00:00")
    
    Returns:
        NotificationConfig object
    """
    
    # Parse boolean env vars
    def parse_bool(key: str, default: bool = False) -> bool:
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    # Parse level
    level_str = os.getenv("NOTIFY_MIN_LEVEL", "INFO").upper()
    try:
        min_level = NotificationLevel[level_str]
    except KeyError:
        min_level = NotificationLevel.INFO
    
    return NotificationConfig(
        # Telegram
        telegram_enabled=parse_bool("TELEGRAM_ENABLED", False),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        
        # Discord
        discord_enabled=parse_bool("DISCORD_ENABLED", False),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL"),
        discord_username=os.getenv("DISCORD_USERNAME", "PR3DICT Bot"),
        
        # Alert settings
        min_notification_level=min_level,
        enable_signal_alerts=parse_bool("NOTIFY_SIGNALS", True),
        enable_trade_alerts=parse_bool("NOTIFY_TRADES", True),
        enable_risk_alerts=parse_bool("NOTIFY_RISK_ALERTS", True),
        enable_daily_summary=parse_bool("NOTIFY_DAILY_SUMMARY", True)
    )


def parse_daily_summary_time() -> time:
    """
    Parse daily summary time from environment.
    
    Expected format: "HH:MM" (24-hour, UTC)
    Default: "00:00" (midnight)
    
    Returns:
        time object
    """
    time_str = os.getenv("DAILY_SUMMARY_TIME", "00:00")
    
    try:
        hour, minute = map(int, time_str.split(":"))
        return time(hour, minute)
    except (ValueError, AttributeError):
        # Invalid format, use default
        return time(0, 0)


# Example usage
if __name__ == "__main__":
    config = load_notification_config()
    print(f"Telegram: {config.telegram_enabled}")
    print(f"Discord: {config.discord_enabled}")
    print(f"Min Level: {config.min_notification_level}")
    print(f"Daily Summary Time: {parse_daily_summary_time()}")
