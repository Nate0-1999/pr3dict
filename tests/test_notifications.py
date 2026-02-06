"""
PR3DICT: Notification System Tests

Unit tests for notification manager, Telegram, and Discord clients.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from src.notifications import (
    NotificationManager,
    NotificationConfig,
    NotificationLevel,
    TelegramNotifier,
    DiscordNotifier
)


class TestTelegramNotifier:
    """Test Telegram notification client."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test Telegram client initialization."""
        telegram = TelegramNotifier(
            bot_token="test_token",
            chat_id="123456",
            enabled=True
        )
        
        assert telegram.bot_token == "test_token"
        assert telegram.chat_id == "123456"
        assert telegram.enabled is True
        assert telegram.client is None
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        telegram = TelegramNotifier(
            bot_token="test_token",
            chat_id="123456"
        )
        
        # Mock HTTP client
        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            
            # Mock getMe response
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "ok": True,
                "result": {"username": "test_bot"}
            }
            mock_instance.get.return_value = mock_response
            
            result = await telegram.connect()
            
            assert result is True
            assert telegram.client is not None
    
    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending message."""
        telegram = TelegramNotifier(
            bot_token="test_token",
            chat_id="123456"
        )
        
        # Setup mock client
        telegram.client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        telegram.client.post.return_value = mock_response
        
        result = await telegram.send_message("Test message")
        
        assert result is True
        telegram.client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_trade_signal(self):
        """Test sending trade signal."""
        telegram = TelegramNotifier(
            bot_token="test_token",
            chat_id="123456"
        )
        
        telegram.client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        telegram.client.post.return_value = mock_response
        
        result = await telegram.send_trade_signal(
            ticker="TEST-MARKET",
            side="YES",
            price=0.5,
            size=10,
            reason="Test signal",
            confidence=0.8
        )
        
        assert result is True


class TestDiscordNotifier:
    """Test Discord notification client."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test Discord client initialization."""
        discord = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/test",
            enabled=True,
            username="Test Bot"
        )
        
        assert discord.webhook_url == "https://discord.com/api/webhooks/test"
        assert discord.enabled is True
        assert discord.username == "Test Bot"
        assert discord.client is None
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        discord = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/test"
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"name": "Test Webhook"}
            mock_instance.get.return_value = mock_response
            
            result = await discord.connect()
            
            assert result is True
            assert discord.client is not None
    
    @pytest.mark.asyncio
    async def test_send_webhook(self):
        """Test sending webhook message."""
        discord = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/test"
        )
        
        discord.client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 204
        discord.client.post.return_value = mock_response
        
        result = await discord.send_webhook(content="Test message")
        
        assert result is True
        discord.client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_embed(self):
        """Test embed creation."""
        discord = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/test"
        )
        
        embed = discord._create_embed(
            title="Test",
            description="Test description",
            color=0x00ff00,
            fields=[{"name": "Field", "value": "Value", "inline": True}]
        )
        
        assert embed["title"] == "Test"
        assert embed["description"] == "Test description"
        assert embed["color"] == 0x00ff00
        assert len(embed["fields"]) == 1


class TestNotificationManager:
    """Test notification manager."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test manager initialization."""
        config = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="test",
            telegram_chat_id="123"
        )
        
        manager = NotificationManager(config)
        
        assert manager.config == config
        assert manager.telegram is None
        assert manager.discord is None
    
    @pytest.mark.asyncio
    async def test_connect_telegram_only(self):
        """Test connecting with Telegram only."""
        config = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="test_token",
            telegram_chat_id="123456"
        )
        
        manager = NotificationManager(config)
        
        # Mock Telegram connect
        with patch.object(TelegramNotifier, 'connect', return_value=True):
            result = await manager.connect()
            
            assert result is True
            assert manager.telegram is not None
            assert manager.discord is None
    
    @pytest.mark.asyncio
    async def test_connect_both_channels(self):
        """Test connecting with both channels."""
        config = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="test_token",
            telegram_chat_id="123456",
            discord_enabled=True,
            discord_webhook_url="https://discord.com/api/webhooks/test"
        )
        
        manager = NotificationManager(config)
        
        with patch.object(TelegramNotifier, 'connect', return_value=True), \
             patch.object(DiscordNotifier, 'connect', return_value=True):
            
            result = await manager.connect()
            
            assert result is True
            assert manager.telegram is not None
            assert manager.discord is not None
    
    @pytest.mark.asyncio
    async def test_should_send_level_filtering(self):
        """Test notification level filtering."""
        config = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="test",
            telegram_chat_id="123",
            min_notification_level=NotificationLevel.WARNING
        )
        
        manager = NotificationManager(config)
        
        # INFO should be filtered out
        assert manager._should_send(NotificationLevel.INFO) is False
        
        # WARNING should pass
        assert manager._should_send(NotificationLevel.WARNING) is True
        
        # ERROR should pass
        assert manager._should_send(NotificationLevel.ERROR) is True
    
    @pytest.mark.asyncio
    async def test_send_signal(self):
        """Test sending signal notification."""
        config = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="test",
            telegram_chat_id="123",
            enable_signal_alerts=True
        )
        
        manager = NotificationManager(config)
        manager.telegram = AsyncMock()
        manager.telegram.send_trade_signal = AsyncMock(return_value=True)
        
        await manager.send_signal(
            ticker="TEST",
            side="YES",
            price=0.5,
            size=10,
            reason="Test"
        )
        
        manager.telegram.send_trade_signal.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_order_placed(self):
        """Test sending order placed notification."""
        config = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="test",
            telegram_chat_id="123",
            enable_trade_alerts=True
        )
        
        manager = NotificationManager(config)
        manager.telegram = AsyncMock()
        manager.telegram.send_order_filled = AsyncMock(return_value=True)
        
        await manager.send_order_placed(
            ticker="TEST",
            side="YES",
            price=0.5,
            size=10,
            order_id="abc123",
            platform="Test"
        )
        
        manager.telegram.send_order_filled.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_position_closed(self):
        """Test sending position closed notification."""
        config = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="test",
            telegram_chat_id="123",
            enable_trade_alerts=True
        )
        
        manager = NotificationManager(config)
        manager.telegram = AsyncMock()
        manager.telegram.send_position_closed = AsyncMock(return_value=True)
        
        await manager.send_position_closed(
            ticker="TEST",
            pnl=10.0,
            pnl_pct=0.05,
            hold_time="1h 30m",
            reason="Test exit"
        )
        
        manager.telegram.send_position_closed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting."""
        config = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="test",
            telegram_chat_id="123",
            max_alerts_per_minute=2
        )
        
        manager = NotificationManager(config)
        
        # First 2 should pass
        assert manager._should_send(NotificationLevel.INFO) is True
        assert manager._should_send(NotificationLevel.INFO) is True
        
        # Third should be blocked
        assert manager._should_send(NotificationLevel.INFO) is False
        
        # Critical should bypass limit
        assert manager._should_send(NotificationLevel.CRITICAL) is True
    
    @pytest.mark.asyncio
    async def test_disabled_alerts(self):
        """Test disabled alert types."""
        config = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="test",
            telegram_chat_id="123",
            enable_signal_alerts=False  # Disabled
        )
        
        manager = NotificationManager(config)
        manager.telegram = AsyncMock()
        manager.telegram.send_trade_signal = AsyncMock(return_value=True)
        
        # Should not call send_trade_signal
        await manager.send_signal(
            ticker="TEST",
            side="YES",
            price=0.5,
            size=10,
            reason="Test"
        )
        
        manager.telegram.send_trade_signal.assert_not_called()


# Run tests with: pytest tests/test_notifications.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
