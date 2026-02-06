"""
PR3DICT: Notification Manager

Unified notification dispatcher coordinating multiple channels.
Handles async delivery, error handling, and alert routing.
"""
import asyncio
import logging
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from .telegram import TelegramNotifier
from .discord import DiscordNotifier

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SIGNAL = "SIGNAL"
    TRADE = "TRADE"
    EXIT = "EXIT"


class AlertType(Enum):
    """Types of trading alerts."""
    SIGNAL_FOUND = "signal_found"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    POSITION_CLOSED = "position_closed"
    RISK_LIMIT = "risk_limit"
    DAILY_SUMMARY = "daily_summary"
    ENGINE_STATUS = "engine_status"
    ERROR = "error"


@dataclass
class NotificationConfig:
    """Notification system configuration."""
    # Telegram settings
    telegram_enabled: bool = False
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # Discord settings
    discord_enabled: bool = False
    discord_webhook_url: Optional[str] = None
    discord_username: str = "PR3DICT Bot"
    
    # Alert filtering
    min_notification_level: NotificationLevel = NotificationLevel.INFO
    enable_trade_alerts: bool = True
    enable_signal_alerts: bool = True
    enable_risk_alerts: bool = True
    enable_daily_summary: bool = True
    
    # Rate limiting
    max_alerts_per_minute: int = 30
    batch_similar_alerts: bool = True


class NotificationManager:
    """
    Unified notification dispatcher.
    
    Manages multiple notification channels (Telegram, Discord)
    with async delivery, error handling, and alert routing.
    
    Features:
    - Multi-channel delivery
    - Async fire-and-forget notifications
    - Error isolation (one channel failure doesn't block others)
    - Alert batching and rate limiting
    - Level-based filtering
    """
    
    def __init__(self, config: NotificationConfig):
        """
        Initialize notification manager.
        
        Args:
            config: Notification configuration
        """
        self.config = config
        
        # Initialize notifiers
        self.telegram: Optional[TelegramNotifier] = None
        self.discord: Optional[DiscordNotifier] = None
        
        # State tracking
        self._connected = False
        self._alert_count = 0
        self._last_reset = datetime.now()
        
        # Alert queue for batching
        self._alert_queue: List[Dict[str, Any]] = []
        self._batch_task: Optional[asyncio.Task] = None
    
    async def connect(self) -> bool:
        """
        Initialize all enabled notification channels.
        
        Returns:
            True if at least one channel connected successfully
        """
        success = False
        
        # Initialize Telegram
        if self.config.telegram_enabled and self.config.telegram_bot_token:
            try:
                self.telegram = TelegramNotifier(
                    bot_token=self.config.telegram_bot_token,
                    chat_id=self.config.telegram_chat_id,
                    enabled=True
                )
                if await self.telegram.connect():
                    logger.info("Telegram notifications enabled")
                    success = True
                else:
                    logger.warning("Telegram connection failed")
                    self.telegram = None
            except Exception as e:
                logger.error(f"Telegram initialization error: {e}")
                self.telegram = None
        
        # Initialize Discord
        if self.config.discord_enabled and self.config.discord_webhook_url:
            try:
                self.discord = DiscordNotifier(
                    webhook_url=self.config.discord_webhook_url,
                    enabled=True,
                    username=self.config.discord_username
                )
                if await self.discord.connect():
                    logger.info("Discord notifications enabled")
                    success = True
                else:
                    logger.warning("Discord connection failed")
                    self.discord = None
            except Exception as e:
                logger.error(f"Discord initialization error: {e}")
                self.discord = None
        
        self._connected = success
        
        if not success:
            logger.warning("No notification channels connected")
        
        return success
    
    async def disconnect(self) -> None:
        """Disconnect all notification channels."""
        if self.telegram:
            await self.telegram.disconnect()
        if self.discord:
            await self.discord.disconnect()
        
        self._connected = False
        logger.info("Notification channels disconnected")
    
    def _should_send(self, level: NotificationLevel) -> bool:
        """Check if alert should be sent based on level and rate limits."""
        # Check minimum level
        level_priority = {
            NotificationLevel.INFO: 0,
            NotificationLevel.SIGNAL: 1,
            NotificationLevel.TRADE: 1,
            NotificationLevel.EXIT: 1,
            NotificationLevel.WARNING: 2,
            NotificationLevel.ERROR: 3,
            NotificationLevel.CRITICAL: 4
        }
        
        min_priority = level_priority[self.config.min_notification_level]
        alert_priority = level_priority[level]
        
        if alert_priority < min_priority:
            return False
        
        # Rate limiting
        now = datetime.now()
        if (now - self._last_reset).total_seconds() >= 60:
            self._alert_count = 0
            self._last_reset = now
        
        if self._alert_count >= self.config.max_alerts_per_minute:
            # Allow critical alerts even if rate limited
            if level != NotificationLevel.CRITICAL:
                logger.warning("Notification rate limit reached")
                return False
        
        self._alert_count += 1
        return True
    
    async def _send_to_all(self, method_name: str, *args, **kwargs) -> None:
        """
        Send notification to all enabled channels.
        
        Args:
            method_name: Method name to call on each notifier
            *args, **kwargs: Arguments to pass to the method
        """
        tasks = []
        
        if self.telegram:
            method = getattr(self.telegram, method_name, None)
            if method:
                tasks.append(self._safe_send(method, *args, **kwargs))
        
        if self.discord:
            method = getattr(self.discord, method_name, None)
            if method:
                tasks.append(self._safe_send(method, *args, **kwargs))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_send(self, method, *args, **kwargs) -> bool:
        """
        Safely execute notification method with error handling.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return await method(*args, **kwargs)
        except Exception as e:
            logger.error(f"Notification send error: {e}", exc_info=True)
            return False
    
    # Public notification methods
    
    async def send_signal(self,
                         ticker: str,
                         side: str,
                         price: float,
                         size: int,
                         reason: str,
                         confidence: Optional[float] = None,
                         strategy: Optional[str] = None) -> None:
        """
        Send trading signal notification.
        
        Args:
            ticker: Market ticker
            side: YES/NO
            price: Target price
            size: Position size (contracts)
            reason: Signal reason
            confidence: Optional confidence score
            strategy: Optional strategy name
        """
        if not self.config.enable_signal_alerts:
            return
        
        if not self._should_send(NotificationLevel.SIGNAL):
            return
        
        logger.info(f"Sending signal notification: {ticker} {side}")
        
        await self._send_to_all(
            "send_trade_signal",
            ticker=ticker,
            side=side,
            price=price,
            size=size,
            reason=reason,
            confidence=confidence,
            strategy=strategy
        )
    
    async def send_order_placed(self,
                               ticker: str,
                               side: str,
                               price: float,
                               size: int,
                               order_id: str,
                               platform: str) -> None:
        """Send order placed notification."""
        if not self.config.enable_trade_alerts:
            return
        
        if not self._should_send(NotificationLevel.TRADE):
            return
        
        logger.info(f"Sending order notification: {order_id}")
        
        await self._send_to_all(
            "send_order_filled",
            ticker=ticker,
            side=side,
            price=price,
            size=size,
            order_id=order_id,
            platform=platform
        )
    
    async def send_position_closed(self,
                                   ticker: str,
                                   pnl: float,
                                   pnl_pct: float,
                                   hold_time: str,
                                   reason: str,
                                   entry_price: Optional[float] = None,
                                   exit_price: Optional[float] = None) -> None:
        """Send position closed notification."""
        if not self.config.enable_trade_alerts:
            return
        
        level = NotificationLevel.TRADE if pnl >= 0 else NotificationLevel.WARNING
        if not self._should_send(level):
            return
        
        logger.info(f"Sending exit notification: {ticker} P&L ${pnl:+.2f}")
        
        await self._send_to_all(
            "send_position_closed",
            ticker=ticker,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_time=hold_time,
            reason=reason,
            entry_price=entry_price,
            exit_price=exit_price
        )
    
    async def send_risk_alert(self,
                             alert_type: str,
                             details: str,
                             severity: str = "WARNING") -> None:
        """Send risk management alert."""
        if not self.config.enable_risk_alerts:
            return
        
        level = NotificationLevel[severity] if severity in NotificationLevel.__members__ else NotificationLevel.WARNING
        if not self._should_send(level):
            return
        
        logger.info(f"Sending risk alert: {alert_type}")
        
        await self._send_to_all(
            "send_risk_alert",
            alert_type=alert_type,
            details=details,
            severity=severity
        )
    
    async def send_daily_summary(self,
                                trades: int,
                                pnl: float,
                                win_rate: float,
                                wins: int = 0,
                                losses: int = 0,
                                best_trade: Optional[str] = None,
                                worst_trade: Optional[str] = None) -> None:
        """Send daily performance summary."""
        if not self.config.enable_daily_summary:
            return
        
        if not self._should_send(NotificationLevel.INFO):
            return
        
        logger.info(f"Sending daily summary: {trades} trades, ${pnl:+.2f}")
        
        await self._send_to_all(
            "send_daily_summary",
            trades=trades,
            pnl=pnl,
            win_rate=win_rate,
            wins=wins,
            losses=losses,
            best_trade=best_trade,
            worst_trade=worst_trade
        )
    
    async def send_error(self,
                        error_msg: str,
                        context: Optional[str] = None,
                        traceback: Optional[str] = None) -> None:
        """Send error notification."""
        if not self._should_send(NotificationLevel.ERROR):
            return
        
        logger.info(f"Sending error notification: {error_msg[:50]}...")
        
        await self._send_to_all(
            "send_error",
            error_msg=error_msg,
            context=context,
            traceback=traceback
        )
    
    async def send_engine_status(self,
                                status: str,
                                uptime: Optional[str] = None,
                                cycle_count: Optional[int] = None) -> None:
        """Send engine status update."""
        if not self._should_send(NotificationLevel.INFO):
            return
        
        logger.info(f"Sending engine status: {status}")
        
        # Only Discord has this method
        if self.discord:
            await self._safe_send(
                self.discord.send_engine_status,
                status=status,
                uptime=uptime,
                cycle_count=cycle_count
            )
