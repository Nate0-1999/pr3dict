"""
PR3DICT: Discord Notifier

Async Discord webhook client for trading alerts.
Uses embeds for rich formatting with color coding.
"""
import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """
    Async Discord webhook notification client.
    
    Features:
    - Rich embeds with color coding
    - Field formatting for structured data
    - Retry logic with exponential backoff
    - Rate limit handling (5 messages per 2 seconds)
    """
    
    def __init__(self, webhook_url: str, enabled: bool = True, username: str = "PR3DICT Bot"):
        """
        Initialize Discord notifier.
        
        Args:
            webhook_url: Discord webhook URL
            enabled: Whether notifications are active
            username: Bot display name
        """
        self.webhook_url = webhook_url
        self.enabled = enabled
        self.username = username
        
        # Async HTTP client
        self.client: Optional[httpx.AsyncClient] = None
        
        # Rate limiting (5 messages per 2 seconds)
        self._message_times: List[float] = []
        self._rate_limit = 5
        self._rate_window = 2.0
    
    async def connect(self) -> bool:
        """
        Initialize HTTP client and verify webhook.
        
        Returns:
            True if connection successful
        """
        if not self.enabled:
            logger.info("Discord notifications disabled")
            return False
        
        try:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
                limits=httpx.Limits(max_keepalive_connections=5)
            )
            
            # Test webhook with GET request
            response = await self.client.get(self.webhook_url)
            
            if response.status_code == 200:
                webhook_info = response.json()
                logger.info(f"Discord webhook connected: {webhook_info.get('name', 'Unknown')}")
                return True
            else:
                logger.error(f"Discord webhook verification failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Discord connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def _enforce_rate_limit(self) -> None:
        """Enforce Discord rate limits (5 messages per 2 seconds)."""
        now = asyncio.get_event_loop().time()
        
        # Clean old message times outside window
        self._message_times = [t for t in self._message_times if now - t < self._rate_window]
        
        # If at limit, wait until oldest message expires
        if len(self._message_times) >= self._rate_limit:
            oldest = self._message_times[0]
            wait_time = self._rate_window - (now - oldest)
            if wait_time > 0:
                logger.debug(f"Discord rate limit, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                now = asyncio.get_event_loop().time()
        
        # Record this message time
        self._message_times.append(now)
    
    async def send_webhook(self,
                          content: Optional[str] = None,
                          embeds: Optional[List[Dict[str, Any]]] = None,
                          max_retries: int = 3) -> bool:
        """
        Send message to Discord webhook.
        
        Args:
            content: Plain text content
            embeds: List of embed objects
            max_retries: Number of retry attempts
            
        Returns:
            True if message sent successfully
        """
        if not self.enabled or not self.client:
            logger.debug("Discord disabled or not connected")
            return False
        
        # Rate limiting
        await self._enforce_rate_limit()
        
        # Prepare payload
        payload = {"username": self.username}
        if content:
            payload["content"] = content
        if embeds:
            payload["embeds"] = embeds
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    self.webhook_url,
                    json=payload
                )
                
                if response.status_code == 204:
                    logger.debug(f"Discord message sent")
                    return True
                
                elif response.status_code == 429:
                    # Rate limited by Discord
                    retry_after = response.json().get("retry_after", 1)
                    logger.warning(f"Discord rate limit, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                
                else:
                    logger.error(f"Discord HTTP {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return False
                    
            except httpx.HTTPError as e:
                logger.error(f"Discord HTTP error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return False
            
            except Exception as e:
                logger.error(f"Discord send error: {e}", exc_info=True)
                return False
        
        return False
    
    def _create_embed(self,
                     title: str,
                     description: str,
                     color: int,
                     fields: Optional[List[Dict[str, Any]]] = None,
                     footer: Optional[str] = None) -> Dict[str, Any]:
        """
        Create Discord embed object.
        
        Args:
            title: Embed title
            description: Main description text
            color: Color in decimal (e.g., 0x00ff00)
            fields: List of field dicts with name/value/inline
            footer: Footer text
            
        Returns:
            Embed dict
        """
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if fields:
            embed["fields"] = fields
        
        if footer:
            embed["footer"] = {"text": footer}
        
        return embed
    
    async def send_alert(self,
                        title: str,
                        message: str,
                        level: str = "INFO",
                        fields: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Send formatted alert message.
        
        Args:
            title: Alert title
            message: Alert message
            level: Alert level (INFO/WARNING/ERROR/CRITICAL)
            fields: Optional list of fields
            
        Returns:
            True if sent successfully
        """
        # Color coding by level
        colors = {
            "INFO": 0x3498db,      # Blue
            "WARNING": 0xf39c12,   # Orange
            "ERROR": 0xe74c3c,     # Red
            "CRITICAL": 0x992d22,  # Dark red
            "SIGNAL": 0x9b59b6,    # Purple
            "TRADE": 0x2ecc71,     # Green
            "EXIT": 0x95a5a6       # Gray
        }
        
        color = colors.get(level, 0x95a5a6)
        
        # Add emoji to title
        emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "🚨",
            "CRITICAL": "🔥",
            "SIGNAL": "📊",
            "TRADE": "💰",
            "EXIT": "🔔"
        }.get(level, "📢")
        
        embed = self._create_embed(
            title=f"{emoji} {title}",
            description=message,
            color=color,
            fields=fields,
            footer="PR3DICT Trading Bot"
        )
        
        return await self.send_webhook(embeds=[embed])
    
    async def send_trade_signal(self,
                               ticker: str,
                               side: str,
                               price: float,
                               size: int,
                               reason: str,
                               confidence: Optional[float] = None,
                               strategy: Optional[str] = None) -> bool:
        """Send trading signal alert."""
        title = f"New Signal: {ticker}"
        message = f"**{side}** position signal detected"
        
        fields = [
            {"name": "Price", "value": f"${price:.3f}", "inline": True},
            {"name": "Size", "value": f"{size} contracts", "inline": True},
            {"name": "Reason", "value": reason, "inline": False}
        ]
        
        if confidence:
            fields.append({
                "name": "Confidence",
                "value": f"{confidence:.1%}",
                "inline": True
            })
        
        if strategy:
            fields.append({
                "name": "Strategy",
                "value": strategy,
                "inline": True
            })
        
        return await self.send_alert(title, message, "SIGNAL", fields)
    
    async def send_order_filled(self,
                               ticker: str,
                               side: str,
                               price: float,
                               size: int,
                               order_id: str,
                               platform: str) -> bool:
        """Send order filled notification."""
        title = f"Order Filled: {ticker}"
        message = f"**{side}** order executed successfully"
        
        fields = [
            {"name": "Price", "value": f"${price:.3f}", "inline": True},
            {"name": "Size", "value": f"{size} contracts", "inline": True},
            {"name": "Platform", "value": platform, "inline": True},
            {"name": "Order ID", "value": f"`{order_id}`", "inline": False}
        ]
        
        return await self.send_alert(title, message, "TRADE", fields)
    
    async def send_position_closed(self,
                                   ticker: str,
                                   pnl: float,
                                   pnl_pct: float,
                                   hold_time: str,
                                   reason: str,
                                   entry_price: Optional[float] = None,
                                   exit_price: Optional[float] = None) -> bool:
        """Send position closed notification."""
        is_profit = pnl >= 0
        title = f"Position Closed: {ticker}"
        message = f"**{'Profit' if is_profit else 'Loss'}** - {reason}"
        
        fields = [
            {"name": "P&L", "value": f"${pnl:+.2f} ({pnl_pct:+.1%})", "inline": True},
            {"name": "Hold Time", "value": hold_time, "inline": True}
        ]
        
        if entry_price and exit_price:
            fields.extend([
                {"name": "Entry", "value": f"${entry_price:.3f}", "inline": True},
                {"name": "Exit", "value": f"${exit_price:.3f}", "inline": True}
            ])
        
        level = "TRADE" if is_profit else "WARNING"
        return await self.send_alert(title, message, level, fields)
    
    async def send_risk_alert(self,
                             alert_type: str,
                             details: str,
                             severity: str = "WARNING") -> bool:
        """Send risk management alert."""
        title = f"Risk Alert: {alert_type}"
        return await self.send_alert(title, details, severity)
    
    async def send_daily_summary(self,
                                trades: int,
                                pnl: float,
                                win_rate: float,
                                wins: int,
                                losses: int,
                                best_trade: Optional[str] = None,
                                worst_trade: Optional[str] = None) -> bool:
        """Send daily performance summary."""
        title = "Daily Performance Summary"
        emoji = "🎉" if pnl >= 0 else "📉"
        message = f"{emoji} End of day results"
        
        fields = [
            {"name": "Total P&L", "value": f"${pnl:+.2f}", "inline": True},
            {"name": "Trades", "value": str(trades), "inline": True},
            {"name": "Win Rate", "value": f"{win_rate:.1%}", "inline": True},
            {"name": "Wins", "value": str(wins), "inline": True},
            {"name": "Losses", "value": str(losses), "inline": True}
        ]
        
        if best_trade:
            fields.append({
                "name": "Best Trade",
                "value": best_trade,
                "inline": False
            })
        
        if worst_trade:
            fields.append({
                "name": "Worst Trade",
                "value": worst_trade,
                "inline": False
            })
        
        return await self.send_alert(title, message, "INFO", fields)
    
    async def send_error(self,
                        error_msg: str,
                        context: Optional[str] = None,
                        traceback: Optional[str] = None) -> bool:
        """Send error notification."""
        title = "System Error"
        message = error_msg
        
        fields = []
        if context:
            fields.append({
                "name": "Context",
                "value": context,
                "inline": False
            })
        
        if traceback:
            # Truncate long tracebacks
            tb = traceback[:1000] + "..." if len(traceback) > 1000 else traceback
            fields.append({
                "name": "Traceback",
                "value": f"```\n{tb}\n```",
                "inline": False
            })
        
        return await self.send_alert(title, message, "ERROR", fields)
    
    async def send_engine_status(self,
                                status: str,
                                uptime: Optional[str] = None,
                                cycle_count: Optional[int] = None) -> bool:
        """Send engine status update."""
        title = f"Engine Status: {status}"
        message = f"Trading engine is now **{status}**"
        
        fields = []
        if uptime:
            fields.append({"name": "Uptime", "value": uptime, "inline": True})
        if cycle_count:
            fields.append({"name": "Cycles", "value": str(cycle_count), "inline": True})
        
        return await self.send_alert(title, message, "INFO", fields if fields else None)
