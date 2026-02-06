# Kalshi Sandbox Setup Documentation

**Date:** 2026-02-02  
**Purpose:** Document the process to obtain Kalshi sandbox API credentials for PR3DICT testing

---

## Overview

Kalshi provides a sandbox environment for testing API integrations without real money. The PR3DICT system is configured to use this sandbox by default.

### API Endpoints
- **Production:** `https://trading-api.kalshi.com/trade-api/v2`
- **Sandbox:** `https://demo-trading-api.kalshi.com/trade-api/v2`

---

## Current Status: ⚠️ Access Issue

### Problem Discovered
The QUICKSTART.md references `https://demo.kalshi.com` for sandbox signup, but this domain **does not resolve**. 

```bash
$ curl https://demo.kalshi.com
curl: (6) Could not resolve host: demo.kalshi.com
```

### Investigation Findings

1. **Primary Website:** The main Kalshi site is `https://kalshi.com`
2. **API Documentation:** Likely at `https://trading-api.kalshi.com/docs` or `https://docs.kalshi.com`
3. **Sandbox Access:** May require:
   - Contacting Kalshi support
   - Separate application process
   - Invitation/approval system
   - Or the demo site may be temporarily down

---

## Alternative Approaches

### Option 1: Contact Kalshi Support
- Email: support@kalshi.com (inferred)
- Request: Sandbox API credentials for development/testing

### Option 2: Check Official Documentation
```bash
# Try accessing API docs directly
curl https://trading-api.kalshi.com/trade-api/v2/docs
curl https://demo-trading-api.kalshi.com/trade-api/v2/docs
```

### Option 3: Use Production with Test Account
⚠️ **Not recommended** - Production environment with real money

### Option 4: Mock Mode Testing
For initial development, we can test the engine with:
- Mock API responses
- Simulated market data
- Local validation of logic

---

## Required Credentials Format

Based on code analysis (`src/platforms/kalshi.py`), the system expects:

```bash
# In config/.env
KALSHI_API_KEY=your_email@example.com
KALSHI_API_SECRET=your_password_here
KALSHI_SANDBOX=true
```

The Kalshi API uses **email + password** authentication (not API key/secret pairs like most services).

### Authentication Flow
1. POST to `/login` with email/password
2. Receive JWT token (30-minute expiry)
3. Use token in `Authorization: Bearer <token>` header
4. Auto-refresh at 25 minutes

---

## Next Steps

1. **Immediate:** Test the system in paper mode without real credentials to validate engine logic
2. **Short-term:** Research official Kalshi developer documentation
3. **Medium-term:** Obtain legitimate sandbox credentials through proper channels

---

## Code Integration Details

### Files Modified/Reviewed
- `src/platforms/kalshi.py` - Platform integration
- `config/.env` - Configuration (currently has empty credentials)
- `run.sh` - Startup script

### Current Config State
```bash
$ cat config/.env | grep KALSHI
KALSHI_API_KEY=
KALSHI_API_SECRET=
KALSHI_SANDBOX=true
```

Credentials are **not set** - system will fail at connection phase.

---

## Testing Without Credentials

We can still test:
- ✅ Engine startup logic
- ✅ Strategy initialization
- ✅ Configuration parsing
- ❌ Actual API connections
- ❌ Real market data
- ❌ Order placement

Expected behavior: The engine will start but fail at the `connect()` phase with authentication errors.

---

## Recommendations

1. **Document the limitation** in QUICKSTART.md (demo.kalshi.com issue)
2. **Add fallback instructions** for users who can't access sandbox
3. **Implement mock mode** for development without API access
4. **Research Kalshi's current developer onboarding** process

---

**Last Updated:** 2026-02-02 22:13 CST  
**Status:** Awaiting official Kalshi documentation/support response
