# PR3DICT Notification System

Real-time trading alerts via Telegram and Discord.

## Features

- **Multi-Channel**: Telegram bot + Discord webhooks
- **Async Delivery**: Non-blocking notifications
- **Alert Types**: Signals, trades, exits, risk alerts, daily summaries
- **Error Handling**: Retries, rate limiting, graceful degradation
- **Configurable**: Filter by level, enable/disable channels

---

## Quick Start

### 1. Setup Telegram (Optional)

**Create a Bot:**
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow prompts
3. Save your bot token (looks like `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)

**Get Your Chat ID:**
1. Message [@userinfobot](https://t.me/userinfobot)
2. Save your chat ID (a number like `123456789`)

**Add to .env:**
```bash
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### 2. Setup Discord (Optional)

**Create a Webhook:**
1. Go to your Discord server
2. Server Settings → Integrations → Webhooks
3. Click "New Webhook"
4. Name it "PR3DICT Bot", choose a channel
5. Copy the webhook URL

**Add to .env:**
```bash
DISCORD_ENABLED=true
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/1234567890/abc...xyz
DISCORD_USERNAME=PR3DICT Bot
```

### 3. Configure Alerts

Edit your `.env` file:

```bash
# What to send
NOTIFY_SIGNALS=true          # New trading signals
NOTIFY_TRADES=true           # Order fills and exits
NOTIFY_RISK_ALERTS=true      # Risk limit warnings
NOTIFY_DAILY_SUMMARY=true    # End-of-day report

# Minimum alert level (INFO, WARNING, ERROR, CRITICAL)
NOTIFY_MIN_LEVEL=INFO

# Daily summary time (24h format, UTC)
DAILY_SUMMARY_TIME=00:00
```

---

## Alert Types

### 🎯 Trading Signals

Sent when a new opportunity is detected:

```
🎯 New Signal: TRUMP-2024-WINNER

Side: YES
Price: $0.643
Size: 50 contracts
Reason: Arbitrage spread 3.2%
Confidence: 87.5%
Strategy: arbitrage

Time: 2024-02-02 14:30:15
```

### 💰 Order Filled

Sent when an order executes:

```
✅ Order Filled: TRUMP-2024-WINNER

Side: YES
Price: $0.645
Size: 50 contracts
Platform: Polymarket
Order ID: abc123xyz789

Time: 2024-02-02 14:30:22
```

### 🔔 Position Closed

Sent when you exit a position:

```
💰 Position Closed: TRUMP-2024-WINNER

P&L: +$12.50 (+3.9%)
Hold Time: 2h 15m
Reason: Spread closed
Entry: $0.645
Exit: $0.670

Time: 2024-02-02 16:45:08
```

### ⚠️ Risk Alerts

Sent when risk limits are hit:

```
⚠️ Risk Alert: DAILY_LOSS_LIMIT_REACHED

Trading blocked: DAILY_LOSS_LIMIT_REACHED

Time: 2024-02-02 18:20:05
```

### 📊 Daily Summary

Sent at midnight (configurable):

```
📊 Daily Summary

🎉 Total P&L: +$127.50
Trades: 12
Win Rate: 66.7%
Wins: 8
Losses: 4

Best Trade: ETH-2500-EOY (+$35.20)
Worst Trade: BTC-100K-2024 (-$18.75)

Time: 2024-02-03 00:00:00
```

### 🚨 System Errors

Sent when critical errors occur:

```
🚨 System Error

Order placement failed: Connection timeout

Context: Market: TRUMP-2024-WINNER, Side: YES

Traceback:
  File "engine/core.py", line 142, in _execute_entry
    order = await platform.place_order(...)
  ...

Time: 2024-02-02 15:10:33
```

---

## Architecture

### Module Structure

```
src/notifications/
├── __init__.py       # Package exports
├── manager.py        # Unified dispatcher
├── telegram.py       # Telegram bot client
├── discord.py        # Discord webhook client
└── scheduler.py      # Daily summary scheduler (in engine/)
```

### Notification Flow

```
Trading Event → NotificationManager → [Telegram, Discord]
                                      (async, parallel)
```

### Error Handling

- **Retries**: 3 attempts with exponential backoff
- **Rate Limiting**: Respects platform limits (Telegram: 1 msg/sec, Discord: 5 msg/2sec)
- **Isolation**: One channel failure doesn't block others
- **Graceful Degradation**: Missing credentials → logs warning, continues trading

---

## Configuration Reference

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TELEGRAM_ENABLED` | bool | `false` | Enable Telegram notifications |
| `TELEGRAM_BOT_TOKEN` | string | - | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | string | - | Your user/group chat ID |
| `DISCORD_ENABLED` | bool | `false` | Enable Discord notifications |
| `DISCORD_WEBHOOK_URL` | string | - | Webhook URL from server settings |
| `DISCORD_USERNAME` | string | `PR3DICT Bot` | Bot display name |
| `NOTIFY_MIN_LEVEL` | enum | `INFO` | Minimum alert level (INFO/WARNING/ERROR/CRITICAL) |
| `NOTIFY_SIGNALS` | bool | `true` | Send new signal alerts |
| `NOTIFY_TRADES` | bool | `true` | Send trade execution alerts |
| `NOTIFY_RISK_ALERTS` | bool | `true` | Send risk limit warnings |
| `NOTIFY_DAILY_SUMMARY` | bool | `true` | Send end-of-day summary |
| `DAILY_SUMMARY_TIME` | time | `00:00` | Daily summary time (UTC, 24h format) |

### Notification Levels

Levels in priority order (highest to lowest):

1. `CRITICAL` - System failures, manual intervention needed
2. `ERROR` - Errors that don't halt trading
3. `WARNING` - Risk limits hit, degraded performance
4. `INFO` - Normal operations, summaries

Alert types map to levels:
- Signals → `SIGNAL` (treated as INFO)
- Trades → `TRADE` (treated as INFO)
- Exits → `EXIT` (INFO for profit, WARNING for loss)
- Risk → `WARNING` or `ERROR`
- Errors → `ERROR` or `CRITICAL`

---

## Code Integration

### Basic Setup

```python
from pr3dict.notifications import NotificationManager, NotificationConfig, NotificationLevel

# Create config
config = NotificationConfig(
    telegram_enabled=True,
    telegram_bot_token="123456:ABC...",
    telegram_chat_id="123456789",
    discord_enabled=True,
    discord_webhook_url="https://discord.com/api/webhooks/...",
    min_notification_level=NotificationLevel.INFO
)

# Initialize manager
notifier = NotificationManager(config)
await notifier.connect()

# Send alerts
await notifier.send_signal(
    ticker="TRUMP-2024-WINNER",
    side="YES",
    price=0.643,
    size=50,
    reason="Arbitrage spread 3.2%",
    confidence=0.875,
    strategy="arbitrage"
)

# Cleanup
await notifier.disconnect()
```

### Engine Integration

The `TradingEngine` automatically sends notifications when you pass a `NotificationManager`:

```python
from pr3dict.engine import TradingEngine
from pr3dict.notifications import NotificationManager, NotificationConfig

# Setup
notifier = NotificationManager(config)
engine = TradingEngine(
    platforms=platforms,
    strategies=strategies,
    risk_manager=risk,
    notifications=notifier  # Add notifier here
)

# Engine will now send:
# - Signal alerts when opportunities found
# - Order alerts when trades execute
# - Exit alerts when positions close
# - Risk alerts when limits hit
# - Error alerts on failures
# - Engine status on start/stop

await engine.start()
```

### Daily Summary Scheduler

```python
from pr3dict.engine.scheduler import TaskScheduler

scheduler = TaskScheduler()
await scheduler.start()

# Schedule daily summary at midnight UTC
async def send_daily_summary():
    await notifier.send_daily_summary(
        trades=stats.trades_today,
        pnl=stats.daily_pnl,
        win_rate=stats.win_rate,
        wins=stats.wins,
        losses=stats.losses,
        best_trade=stats.best_trade,
        worst_trade=stats.worst_trade
    )

scheduler.schedule_daily_summary(send_daily_summary)
```

---

## Testing

### Test Telegram

```python
import asyncio
from pr3dict.notifications.telegram import TelegramNotifier

async def test():
    telegram = TelegramNotifier(
        bot_token="your_token",
        chat_id="your_chat_id"
    )
    
    if await telegram.connect():
        await telegram.send_message("✅ Test message from PR3DICT!")
        await telegram.disconnect()

asyncio.run(test())
```

### Test Discord

```python
import asyncio
from pr3dict.notifications.discord import DiscordNotifier

async def test():
    discord = DiscordNotifier(
        webhook_url="your_webhook_url"
    )
    
    if await discord.connect():
        await discord.send_alert(
            title="Test Alert",
            message="Testing Discord notifications",
            level="INFO"
        )
        await discord.disconnect()

asyncio.run(test())
```

### Full Integration Test

```bash
# Set credentials in .env
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Run engine in paper mode
python -m pr3dict.engine.main

# Watch for alerts when signals trigger
```

---

## Troubleshooting

### Telegram Not Working

**"Unauthorized" error:**
- Check bot token is correct
- Make sure you started a chat with the bot (send `/start`)

**Messages not arriving:**
- Verify chat ID (message @userinfobot)
- Check bot isn't blocked by privacy settings
- For groups: Add bot to group, make sure it can read messages

**Rate limiting:**
- Default: 1 message per second
- Increase `_min_interval` in `TelegramNotifier` if needed

### Discord Not Working

**"Invalid webhook" error:**
- Regenerate webhook URL
- Make sure webhook wasn't deleted from server

**Embeds not showing:**
- Check webhook channel permissions
- Try plain text mode: set `content` instead of `embeds`

**Rate limiting:**
- Discord allows 5 messages per 2 seconds per webhook
- System automatically handles this

### No Notifications at All

1. Check `.env` has `TELEGRAM_ENABLED=true` or `DISCORD_ENABLED=true`
2. Verify credentials are correct
3. Check logs for connection errors: `grep "notification" pr3dict.log`
4. Test individual channels (see Testing section above)
5. Check `NOTIFY_MIN_LEVEL` isn't filtering out alerts

### Too Many Notifications

Adjust settings in `.env`:

```bash
NOTIFY_MIN_LEVEL=WARNING    # Only warnings and errors
NOTIFY_SIGNALS=false         # Disable signal alerts
NOTIFY_TRADES=false          # Disable trade alerts
```

Or use Discord only (embeds batch better):
```bash
TELEGRAM_ENABLED=false
DISCORD_ENABLED=true
```

---

## Advanced Usage

### Custom Alert Levels

```python
# Send custom alert
await notifier.send_alert(
    title="Custom Alert",
    message="Something important happened",
    level="WARNING",
    fields=[
        {"name": "Detail 1", "value": "Value 1", "inline": True},
        {"name": "Detail 2", "value": "Value 2", "inline": True}
    ]
)
```

### Conditional Notifications

```python
# Only notify on large trades
if size > 100:
    await notifier.send_signal(...)

# Only notify on significant P&L
if abs(pnl) > 50:
    await notifier.send_position_closed(...)
```

### Multiple Channels

You can send to different channels for different alert types:

```python
# Main alerts to Telegram
main_notifier = NotificationManager(telegram_config)

# High-priority to Discord
priority_notifier = NotificationManager(discord_config)

# Use accordingly
await main_notifier.send_signal(...)
await priority_notifier.send_risk_alert(...)
```

---

## Security

### Protect Your Credentials

❌ **Never commit credentials:**
```bash
# Add to .gitignore
.env
*.env
config/.env
```

✅ **Use environment variables:**
```bash
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
```

✅ **Use secrets management:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("TELEGRAM_BOT_TOKEN")
```

### Webhook Security

Discord webhooks can be regenerated if compromised:
1. Server Settings → Integrations → Webhooks
2. Click your webhook → "Regenerate"
3. Update `.env` with new URL

Telegram bots can be reset:
1. Message @BotFather
2. Send `/token`
3. Select your bot
4. Update `.env` with new token

---

## Performance

### Async Design

All notifications are **fire-and-forget** - they won't block trading:

```python
# This returns immediately
await notifier.send_signal(...)

# Trading continues without waiting for delivery
await execute_order(...)
```

### Rate Limits

Built-in rate limiting prevents API bans:
- **Telegram**: 1 message/second, 20 messages/minute to same chat
- **Discord**: 5 messages/2 seconds per webhook, 30 messages/minute

### Resource Usage

Negligible impact:
- HTTP client pooling (reuses connections)
- Async I/O (non-blocking)
- ~1KB per message
- ~0.1s latency per notification

---

## Future Enhancements

Potential additions (not yet implemented):

- [ ] SMS notifications (Twilio)
- [ ] Email alerts (SendGrid)
- [ ] Slack integration
- [ ] Push notifications (mobile app)
- [ ] Alert aggregation (batch similar alerts)
- [ ] Rich media (charts, screenshots)
- [ ] Two-way commands (pause trading via Telegram)
- [ ] Alert persistence (store in DB)
- [ ] Notification analytics (delivery rates, user engagement)

---

## Support

**Issues:**
- Check logs: `tail -f pr3dict.log | grep notification`
- Enable debug logging: `LOG_LEVEL=DEBUG` in `.env`
- Test each channel individually (see Testing section)

**Questions:**
- See examples in `examples/` directory
- Check API docs: [Telegram](https://core.telegram.org/bots/api), [Discord](https://discord.com/developers/docs/resources/webhook)

---

**Built with ❤️ for PR3DICT**  
Part of the unified prediction market trading engine.
