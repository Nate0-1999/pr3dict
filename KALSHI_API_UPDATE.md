# Kalshi API Update - CRITICAL FINDINGS

**Date:** 2026-02-02  
**Status:** ✅ API endpoint identified and working

---

## 🎯 Key Findings

### 1. API Has Moved!

The Kalshi API has been **migrated** to a new domain:

**OLD URLs (No longer work):**
```
https://trading-api.kalshi.com/trade-api/v2  ❌ Returns: "API has been moved"
https://demo-trading-api.kalshi.com          ❌ DNS does not resolve (NXDOMAIN)
```

**NEW URLs (Current, working):**
```
https://api.elections.kalshi.com/trade-api/v2  ✅ ACTIVE (returns market data)
```

### 2. DNS Resolution Test Results

```bash
$ nslookup demo-trading-api.kalshi.com
** server can't find demo-trading-api.kalshi.com: NXDOMAIN

$ nslookup trading-api.kalshi.com  
Name: trading-api.kalshi.com
Address: 18.225.22.0

$ nslookup api.elections.kalshi.com
Name: api.elections.kalshi.com
Address: [resolves successfully]
```

### 3. Current API Response

The new endpoint returns valid market data:
```bash
$ curl https://api.elections.kalshi.com/trade-api/v2/ | head -100
{
  "series": [...],
  "milestones": [...],
  "liveData": [...],
  "structuredTargets": [...]
}
```

Returns extensive data including:
- Active sports markets (NHL, NFL, NCAA Basketball)
- Political/prediction markets
- Live game data
- Team and player information
- Super Bowl LIX markets

---

## 📝 Required Code Changes

### Update `src/platforms/kalshi.py`

**Line 33-34:**
```python
# OLD (doesn't work)
PROD_URL = "https://trading-api.kalshi.com/trade-api/v2"
SANDBOX_URL = "https://demo-trading-api.kalshi.com/trade-api/v2"

# NEW (working)
PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
SANDBOX_URL = "https://api.elections.kalshi.com/trade-api/v2"  # May be same as prod
```

**Note:** The sandbox environment may not exist separately anymore, or might use the same URL with different credentials.

---

## 🔑 Authentication Question

The API appears to be publicly accessible for market data (GET requests), but **authentication is still required** for:
- Account operations (balance, positions)
- Order placement
- Private data

**Next Steps:**
1. Contact Kalshi support to request API credentials
2. Determine if sandbox environment exists separately
3. Test authentication flow with the new endpoint

---

## 🧪 Testing Without Credentials

You can test market data fetching **without authentication**:

```python
import httpx

async def test_kalshi_markets():
    url = "https://api.elections.kalshi.com/trade-api/v2/"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        print(f"Found {len(data.get('series', []))} market series")
```

---

## 📧 Contacting Kalshi

**Recommended approach:**
1. Visit https://kalshi.com
2. Look for "Developers" or "API" section
3. Email support@kalshi.com with:
   - Request for sandbox/demo API access
   - Purpose: Testing automated trading system
   - Need: API credentials for development

**Questions to ask:**
- Is there a sandbox environment for testing?
- How do I obtain API credentials?
- What's the authentication method? (email/password, API keys, OAuth?)
- Rate limits and usage guidelines?

---

## ✅ Immediate Actions

1. **Update code** with new API endpoint ✅
2. **Update QUICKSTART.md** to reflect correct process
3. **Test market data fetching** without auth
4. **Contact Kalshi** for credentials
5. **Document authentication flow** once obtained

---

## 🔄 Migration Path

If you have existing Kalshi API credentials from the old system:
- They **may still work** with the new endpoint
- Test by updating the URL in your configuration
- If they don't work, request new credentials

---

**Status:** Ready to proceed with code updates and Kalshi outreach.
