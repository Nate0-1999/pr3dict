# PR3DICT Quick Start

Get up and running in 5 minutes.

---

## Step 1: Install Dependencies

```bash
cd ~/.openclaw/workspace/pr3dict
pip3 install -r requirements.txt
```

---

## Step 2: Get Kalshi Sandbox Credentials

1. Go to https://demo.kalshi.com
2. Sign up for a free sandbox account
3. Navigate to Settings → API
4. Copy your email and generate an API password

---

## Step 3: Configure

Edit `config/.env`:

```bash
KALSHI_API_KEY=your_email@example.com
KALSHI_API_SECRET=your_api_password
KALSHI_SANDBOX=true
```

---

## Step 4: Run in Paper Mode

```bash
./run.sh paper kalshi
```

You'll see:
- Engine startup logs
- Platform connections
- Market scanning
- Arbitrage signals (if found)
- Position management

**Paper mode** = No real orders placed. Safe testing.

---

## Step 5: Monitor (Optional)

In a separate terminal:

```bash
python3 monitor.py
```

Shows real-time:
- Account balance
- Open positions
- P&L
- Cache stats

---

## What Happens Next?

The engine will:

1. **Scan markets** every 60 seconds (configurable in `.env`)
2. **Look for arbitrage** opportunities:
   - Binary complement (YES + NO < $1.00)
   - Wide spreads (mispricing)
3. **Paper trade** signals (simulated orders in logs)
4. **Track positions** and exits

---

## Going Live

**⚠️ Don't rush this.**

Before real money:

1. Run paper mode for 24+ hours
2. Verify arbitrage signals are real
3. Check risk limits work
4. Start with small position sizes
5. Monitor actively for first week

To go live:

```bash
# Set PAPER_MODE=false in config/.env
./run.sh live kalshi
```

---

## Strategy Status

✅ **Arbitrage** - Ready to use  
🚧 **Market Making** - In development  
🚧 **Behavioral** - In development  
🚧 **AI Forecasting** - Planned

---

## Troubleshooting

**"Failed to connect to Kalshi"**
- Check credentials in config/.env
- Verify sandbox mode matches endpoint
- Try generating new API password

**"No markets found"**
- Kalshi sandbox may have limited markets
- Try during US market hours

**"Redis connection failed"**
- Redis is optional (caching only)
- Install: `brew install redis` (Mac) or `apt install redis` (Linux)
- Start: `redis-server`

---

## Next Steps

- Add Polymarket integration
- Enable cross-platform arbitrage
- Deploy to cloud for 24/7 operation
- Add Telegram/Discord notifications

---

**Questions?** Check the main README.md or dive into `src/` code.
