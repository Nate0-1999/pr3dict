"""
PR3DICT: Telegram Notifier

Async Telegram bot client for trading alerts.
Uses Bot API for reliable message delivery with retry logic.
"""
import asyncio
import logging
from typing import Optional
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Async Telegram notification client.
    
    Features:
    - Message formatting with Markdown
    - Retry logic with exponential backoff
    - Rate limit handling
    - Connection pooling
    """
    
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram Bot API token (from @BotFather)
            chat_id: Target chat ID (user or group)
            enabled: Whether notifications are active
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Async HTTP client with connection pooling
        self.client: Optional[httpx.AsyncClient] = None
        
        # Rate limiting
        self._last_send = 0.0
        self._min_interval = 1.0  # Min 1s between messages
    
    async def connect(self) -> bool:
        """
        Initialize HTTP client and verify bot credentials.
        
        Returns:
            True if connection successful
        """
        if not self.enabled:
            logger.info("Telegram notifications disabled")
            return False
        
        try:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
                limits=httpx.Limits(max_keepalive_connections=5)
            )
            
            # Verify bot token
            response = await self.client.get(f"{self.base_url}/getMe")
            response.raise_for_status()
            
            bot_info = response.json()
            if bot_info.get("ok"):
                username = bot_info["result"]["username"]
                logger.info(f"Telegram bot connected: @{username}")
                return True
            else:
                logger.error(f"Telegram auth failed: {bot_info}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def send_message(self,
                          text: str,
                          parse_mode: str = "Markdown",
                          disable_preview: bool = True,
                          max_retries: int = 3) -> bool:
        """
        Send message to Telegram chat.
        
        Args:
            text: Message text (supports Markdown)
            parse_mode: Formatting mode (Markdown or HTML)
            disable_preview: Disable link previews
            max_retries: Number of retry attempts
            
        Returns:
            True if message sent successfully
        """
        if not self.enabled or not self.client:
            logger.debug("Telegram disabled or not connected")
            return False
        
        # Rate limiting
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_send
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        
        # Prepare request
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_preview
        }
        
        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    f"{self.base_url}/sendMessage",
                    json=payload
                )
                
                self._last_send = asyncio.get_event_loop().time()
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("ok"):
                        logger.debug(f"Telegram message sent: {text[:50]}...")
                        return True
                    else:
                        logger.error(f"Telegram error: {result}")
                        return False
                
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = response.json().get("parameters", {}).get("retry_after", 1)
                    logger.warning(f"Telegram rate limit, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                
                else:
                    logger.error(f"Telegram HTTP {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return False
                    
            except httpx.HTTPError as e:
                logger.error(f"Telegram HTTP error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return False
            
            except Exception as e:
                logger.error(f"Telegram send error: {e}", exc_info=True)
                return False
        
        return False
    
    async def send_alert(self,
                        title: str,
                        message: str,
                        level: str = "INFO",
                        data: Optional[dict] = None) -> bool:
        """
        Send formatted alert message.
        
        Args:
            title: Alert title
            message: Alert message
            level: Alert level (INFO/WARNING/ERROR/CRITICAL)
            data: Optional data dict to include
            
        Returns:
            True if sent successfully
        """
        # Format alert with emoji and markdown
        emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "🚨",
            "CRITICAL": "🔥",
            "SIGNAL": "📊",
            "TRADE": "💰",
            "EXIT": "🔔"
        }.get(level, "📢")
        
        text = f"{emoji} *{title}*\n\n{message}"
        
        # Add data if provided
        if data:
            text += "\n\n"
            for key, value in data.items():
                text += f"• *{key}:* `{value}`\n"
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text += f"\n_Time: {timestamp}_"
        
        return await self.send_message(text)
    
    async def send_trade_signal(self,
                               ticker: str,
                               side: str,
                               price: float,
                               size: int,
                               reason: str,
                               confidence: Optional[float] = None) -> bool:
        """
        Send trading signal alert.
        
        Args:
            ticker: Market ticker
            side: YES/NO
            price: Target price
            size: Position size (contracts)
            reason: Signal reason
            confidence: Optional confidence score
        """
        title = f"🎯 New Signal: {ticker}"
        
        message = f"*Side:* {side}\n"
        message += f"*Price:* ${price:.3f}\n"
        message += f"*Size:* {size} contracts\n"
        message += f"*Reason:* {reason}"
        
        if confidence:
            message += f"\n*Confidence:* {confidence:.1%}"
        
        return await self.send_alert(title, message, "SIGNAL")
    
    async def send_order_filled(self,
                               ticker: str,
                               side: str,
                               price: float,
                               size: int,
                               order_id: str) -> bool:
        """Send order filled notification."""
        title = f"✅ Order Filled: {ticker}"
        
        message = f"*Side:* {side}\n"
        message += f"*Price:* ${price:.3f}\n"
        message += f"*Size:* {size} contracts\n"
        message += f"*Order ID:* `{order_id}`"
        
        return await self.send_alert(title, message, "TRADE")
    
    async def send_position_closed(self,
                                   ticker: str,
                                   pnl: float,
                                   pnl_pct: float,
                                   hold_time: str,
                                   reason: str) -> bool:
        """Send position closed notification."""
        is_profit = pnl >= 0
        emoji = "💰" if is_profit else "📉"
        
        title = f"{emoji} Position Closed: {ticker}"
        
        message = f"*P&L:* ${pnl:+.2f} ({pnl_pct:+.1%})\n"
        message += f"*Hold Time:* {hold_time}\n"
        message += f"*Reason:* {reason}"
        
        level = "TRADE" if is_profit else "WARNING"
        return await self.send_alert(title, message, level)
    
    async def send_risk_alert(self, alert_type: str, details: str) -> bool:
        """Send risk management alert."""
        title = f"⚠️ Risk Alert: {alert_type}"
        return await self.send_alert(title, details, "WARNING")
    
    async def send_daily_summary(self,
                                trades: int,
                                pnl: float,
                                win_rate: float,
                                best_trade: Optional[str] = None,
                                worst_trade: Optional[str] = None) -> bool:
        """Send daily performance summary."""
        title = "📊 Daily Summary"
        
        emoji = "🎉" if pnl >= 0 else "📉"
        
        message = f"*Total P&L:* {emoji} ${pnl:+.2f}\n"
        message += f"*Trades:* {trades}\n"
        message += f"*Win Rate:* {win_rate:.1%}\n"
        
        if best_trade:
            message += f"*Best Trade:* {best_trade}\n"
        if worst_trade:
            message += f"*Worst Trade:* {worst_trade}"
        
        return await self.send_alert(title, message, "INFO")
    
    async def send_error(self, error_msg: str, context: Optional[str] = None) -> bool:
        """Send error notification."""
        title = "🚨 System Error"
        message = error_msg
        if context:
            message += f"\n\n*Context:* {context}"
        
        return await self.send_alert(title, message, "ERROR")
