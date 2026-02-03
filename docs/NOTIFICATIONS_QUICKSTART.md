# Notifications Quick Start

Get trading alerts in 5 minutes.

## 1. Choose Your Channel

### Option A: Telegram

**Get Bot Token:**
1. Message [@BotFather](https://t.me/BotFather)
2. Send `/newbot`
3. Copy your bot token

**Get Chat ID:**
1. Message [@userinfobot](https://t.me/userinfobot)
2. Copy your chat ID (the number)

### Option B: Discord

**Create Webhook:**
1. Server Settings → Integrations → Webhooks
2. Click "New Webhook"
3. Copy webhook URL

## 2. Configure

Edit `.env`:

```bash
# Telegram
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Discord
DISCORD_ENABLED=true
DISCORD_WEBHOOK_URL=your_webhook_url_here
```

## 3. Test

```bash
python examples/test_notifications.py
```

Check your phone/Discord for 8 test messages!

## 4. Run

```bash
python -m src.engine.main
```

You'll now get alerts for:
- 🎯 New trading signals
- 💰 Order fills
- 🔔 Position exits
- ⚠️ Risk limits hit
- 🚨 System errors
- 📊 Daily summary (midnight)

---

**Full docs:** See `docs/NOTIFICATIONS.md`  
**Troubleshooting:** Check `pr3dict.log`
