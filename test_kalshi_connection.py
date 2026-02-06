#!/usr/bin/env python3
"""
Test Kalshi API connectivity without authentication.
Tests the new api.elections.kalshi.com endpoint.
"""
import asyncio
import httpx
import json


async def test_public_api():
    """Test public market data access."""
    url = "https://api.elections.kalshi.com/trade-api/v2/"
    
    print("🔍 Testing Kalshi API connectivity...")
    print(f"📍 Endpoint: {url}\n")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            print("⏳ Fetching market data...")
            response = await client.get(url)
            
            print(f"✅ Status: {response.status_code}")
            print(f"📊 Response size: {len(response.content)} bytes\n")
            
            if response.status_code == 200:
                data = response.json()
                
                # Analyze response structure
                print("📦 Response structure:")
                for key in data.keys():
                    if isinstance(data[key], list):
                        print(f"  - {key}: {len(data[key])} items")
                    elif isinstance(data[key], dict):
                        print(f"  - {key}: dict with {len(data[key])} keys")
                    else:
                        print(f"  - {key}: {type(data[key]).__name__}")
                
                # Show some market series
                if 'series' in data and data['series']:
                    print(f"\n🎯 Sample markets (first 5):")
                    for i, series in enumerate(data['series'][:5]):
                        ticker = series.get('ticker', 'N/A')
                        title = series.get('title', 'N/A')
                        category = series.get('category', 'N/A')
                        print(f"  {i+1}. {ticker}: {title} ({category})")
                
                return True
            else:
                print(f"❌ Failed: {response.status_code}")
                print(f"Response: {response.text[:500]}")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_auth_required():
    """Test if authentication is required for account endpoints."""
    base_url = "https://api.elections.kalshi.com/trade-api/v2"
    endpoints = [
        "/portfolio/balance",
        "/portfolio/positions",
        "/markets",
    ]
    
    print("\n\n🔐 Testing authenticated endpoints...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"
            try:
                response = await client.get(url)
                if response.status_code == 401:
                    print(f"  {endpoint}: ❌ 401 Unauthorized (expected)")
                elif response.status_code == 200:
                    print(f"  {endpoint}: ✅ 200 OK (public data)")
                else:
                    print(f"  {endpoint}: ⚠️  {response.status_code}")
            except Exception as e:
                print(f"  {endpoint}: ❌ Error: {e}")


async def main():
    success = await test_public_api()
    
    if success:
        print("\n" + "="*60)
        print("✅ API ENDPOINT VERIFIED!")
        print("="*60)
        print("\nNext steps:")
        print("1. The API endpoint is correct and working")
        print("2. Public market data is accessible")
        print("3. Need to obtain credentials for trading operations")
        print("4. Contact Kalshi for API access\n")
    
    await test_auth_required()


if __name__ == "__main__":
    asyncio.run(main())
