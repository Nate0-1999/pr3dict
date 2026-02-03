# PR3DICT v2.0 - Technical Architecture Diagram

**Last Updated:** February 2, 2026  
**Version:** 2.0.0

---

## System Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         PR3DICT v2.0 SYSTEM ARCHITECTURE                   │
│                    High-Frequency Prediction Market Trading                │
└───────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   Data Sources  │
                              └────────┬────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │ Kalshi WebSocket│     │Polymarket WS    │     │  REST APIs      │
    │ (elections API) │     │ (CLOB)          │     │ (fallback)      │
    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  WebSocket Manager  │
                          │  - Auto-reconnect   │
                          │  - <5ms latency     │
                          │  - L2 orderbook     │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  Orderbook Manager  │
                          │  - Reconstruct book │
                          │  - Incremental upd  │
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                                 │
                    ▼                                 ▼
         ┌─────────────────────┐          ┌─────────────────────┐
         │   VWAP Calculator   │          │    Redis Cache      │
         │   - Multi-depth     │◀─────────│    Multi-TTL        │
         │   - Price impact    │          │    - Orderbooks 5s  │
         │   - Slippage est    │          │    - VWAP 60s       │
         └──────────┬──────────┘          │    - Metadata 24h   │
                    │                     └─────────────────────┘
                    │                                 │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  Strategy Manager   │
                          │  Scans markets      │
                          │  Generates signals  │
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
         ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
         │  Arbitrage  │  │Market Making│  │ Behavioral  │
         │             │  │             │  │             │
         │ - Binary    │  │ - Dynamic   │  │ - Longshot  │
         │   complement│  │   spread    │  │ - Favorite  │
         │ - Cross-    │  │ - Inventory │  │ - Overreact │
         │   platform  │  │   mgmt      │  │ - Recency   │
         │             │  │ - Skew price│  │ - Time arb  │
         └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
                │                │                │
                └────────────────┼────────────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ Optimization Engine │
                      │ (IP/LP Solver)      │
                      │                     │
                      │ - Gurobi <50ms      │
                      │ - Multi-leg alloc   │
                      │ - Capital opt       │
                      │ - Constraints       │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │   Risk Manager      │
                      │                     │
                      │ - Position limits   │
                      │ - Daily loss limit  │
                      │ - Portfolio heat    │
                      │ - Kill switch       │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ Parallel Executor   │
                      │                     │
                      │ - MARKET orders     │
                      │ - LIMIT orders      │
                      │ - HYBRID strategy   │
                      │ - Atomic execution  │
                      │ - <30ms target      │
                      │ - Rollback logic    │
                      └──────────┬──────────┘
                                 │
                    ┌────────────┼────────────┐
                    │                         │
                    ▼                         ▼
         ┌─────────────────┐       ┌─────────────────┐
         │ Kalshi Platform │       │Polymarket (CLOB)│
         │                 │       │                 │
         │ - REST API      │       │ - py-clob-client│
         │ - Order mgmt    │       │ - Polygon tx    │
         │ - Position track│       │ - Smart contract│
         └────────┬────────┘       └────────┬────────┘
                  │                         │
                  └────────────┬────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Position Manager    │
                    │ - Track all positions│
                    │ - P&L calculation   │
                    │ - Exit logic        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
   ┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
   │  Notifications  │ │  PostgreSQL  │ │ Monitoring   │
   │                 │ │              │ │              │
   │ - Telegram      │ │ - Trade log  │ │ - Grafana    │
   │ - Discord       │ │ - Positions  │ │ - Prometheus │
   │ - Alerts        │ │ - P&L        │ │ - Datadog    │
   └─────────────────┘ └──────────────┘ └──────────────┘
```

---

## Data Flow Diagram

### Flow 1: Arbitrage Detection & Execution

```
Market Update
     │
     ▼
WebSocket ──────────────────> Redis Cache
     │                             │
     │                             ▼
     │                        VWAP Calc ───> Cache VWAP
     │                             │
     │                             │
     └─────────────────────────────┤
                                   │
                                   ▼
                          Arbitrage Strategy
                          "Scan for opportunities"
                                   │
                   YES: Edge >2%?  │  NO: Skip
                                   ▼
                          Optimization Engine
                          "Allocate capital optimally"
                                   │
                                   ▼
                            Risk Manager
                          "Check position limits"
                                   │
                         OK? ┌─────┴─────┐ REJECT
                             │           │
                             ▼           ▼
                      Parallel Executor  Log & Alert
                      "Atomic multi-leg"
                             │
                  ┌──────────┼──────────┐
                  ▼          ▼          ▼
               Leg 1      Leg 2      Leg 3
             (Kalshi)   (Kalshi)  (Polymarket)
                  │          │          │
       All Fill? ─┴──────────┴──────────┘
                  │
         YES ┌────┴────┐ NO
             │         │
             ▼         ▼
          Commit    Rollback
             │         │
             └────┬────┘
                  │
                  ▼
          Position Manager
          "Track P&L"
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
  Notification  Database  Monitoring
```

**Latency Budget:**
- WebSocket → Cache: 5ms
- VWAP calc: 2ms
- Strategy scan: 10ms
- Optimization: 50ms
- Risk check: 5ms
- Execution: 30ms
- **Total: 102ms** (target: <100ms)

---

### Flow 2: Market Making Quote Generation

```
Orderbook Update
     │
     ▼
WebSocket ──> Orderbook Manager
     │              │
     │              ▼
     │         Calculate:
     │         - Best bid/ask
     │         - Spread
     │         - Liquidity depth
     │              │
     └──────────────┤
                    │
                    ▼
          Market Making Strategy
          "Generate quote signals"
                    │
         ┌──────────┼──────────┐
         │          │          │
         ▼          ▼          ▼
    Calc Base   Inventory   Market
     Spread       Skew     Volatility
         │          │          │
         └──────────┼──────────┘
                    │
                    ▼
              Adjust Spread
              - Base: 4%
              - Volatility: ×1.5
              - Inventory: +1% per 10 pos
              - Time: ×2.0 if <6h
                    │
                    ▼
           Generate BID & ASK
           Limit orders
                    │
                    ▼
              Risk Manager
              "Check inventory limits"
                    │
                    ▼
              Place Orders
              (LIMIT @ calculated prices)
                    │
              Wait for fill...
                    │
           Fill? ┌──┴──┐
                 │     │
            YES  │     │ NO (timeout)
                 ▼     ▼
           Update   Cancel
          Inventory  Order
                 │
                 ▼
            Rebalance
           if skew > 20
                 │
                 ▼
          Exit opposite side
```

**Quote Frequency:**
- Scan: Every 30 seconds
- Requote: On 2% price change
- Max quote age: 30 seconds

---

### Flow 3: Behavioral Bias Detection

```
Price History Update
     │
     ▼
VWAP Calculator
"Build price history"
     │
     ▼
Behavioral Strategy
"Detect bias patterns"
     │
┌────┴──────────────────┬─────────────┬──────────────┐
│                       │             │              │
▼                       ▼             ▼              ▼
Longshot?          Favorite?     Overreaction?  Recency?
P<15%              P>70%         ΔP>20% in 6h   Vol spike
│                       │             │              │
│ BET NO                │ BET YES     │ FADE         │ COUNTER
│                       │             │              │
└───────────────────────┴─────────────┴──────────────┘
                        │
                        ▼
                Entry Criteria Met?
                - Min volume >$1000
                - Min edge >2%
                        │
                   YES  │  NO
                        ▼
                  Place Order
                  (LIMIT @ target)
                        │
                  Wait for fill...
                        │
                    Monitor:
                    - Profit target (50% of edge)
                    - Stop loss (2× edge)
                    - Time exit (7 days)
                    - Signal reversal
                        │
                   Exit signal?
                        │
                        ▼
                   Close Position
                        │
                        ▼
                  Calculate P&L
                        │
                        ▼
                     Notify
```

**Hold Time:**
- Minimum: 1 hour
- Average: 2 days
- Maximum: 7 days

---

## Component Interfaces

### WebSocket Client → Cache

```python
class WebSocketClient:
    async def on_orderbook_update(self, snapshot: OrderBookSnapshot):
        """
        Publishes to Redis:
        - Channel: "polymarket:orderbook:{asset_id}"
        - Key: "orderbook:polymarket:{asset_id}"
        - TTL: 5 seconds
        """
        await redis.publish(f"orderbook:{snapshot.asset_id}", snapshot.to_json())
        await redis.setex(f"orderbook:{snapshot.asset_id}", 5, snapshot.to_json())
```

### Cache → VWAP Calculator

```python
class VWAPCalculator:
    async def calculate(self, asset_id: str, depth_usdc: Decimal) -> Decimal:
        """
        Reads from Redis:
        - Key: "orderbook:polymarket:{asset_id}"
        
        Writes to Redis:
        - Key: "vwap:{asset_id}:{depth}"
        - TTL: 60 seconds
        """
        orderbook = await redis.get(f"orderbook:{asset_id}")
        vwap = self._calculate_vwap(orderbook, depth_usdc)
        await redis.setex(f"vwap:{asset_id}:{depth}", 60, str(vwap))
        return vwap
```

### Strategy → Optimizer

```python
class ArbitrageStrategy:
    async def generate_signals(self) -> List[ArbitrageOpportunity]:
        """
        Returns list of opportunities:
        {
            "legs": [
                {"market_id": "...", "side": "YES", "price": 0.48},
                {"market_id": "...", "side": "NO", "price": 0.54}
            ],
            "expected_edge": 0.02,  # 2%
            "liquidity": 1000.0
        }
        """
        opportunities = self._scan_markets()
        return [opp for opp in opportunities if opp.edge >= self.min_edge]

class OptimizationEngine:
    async def allocate_capital(
        self, 
        opportunities: List[Opportunity],
        available_capital: Decimal
    ) -> Dict[str, Allocation]:
        """
        Solves IP/LP problem:
        - Maximize: Σ (edge_i × allocation_i)
        - Subject to:
            - Σ allocation_i ≤ available_capital
            - allocation_i ≤ liquidity_i
            - allocation_i ≤ max_position_size
        
        Returns:
        {
            "opp_1": {"capital": 1000, "legs": [...]},
            "opp_2": {"capital": 500, "legs": [...]}
        }
        """
        return self.solver.solve(opportunities, available_capital)
```

### Optimizer → Executor

```python
class ParallelExecutor:
    async def execute_arbitrage(
        self,
        legs: List[TradeLeg],
        strategy: ExecutionStrategy = HYBRID
    ) -> MultiLegTrade:
        """
        Executes multi-leg trade atomically:
        
        1. Preflight checks (risk, capital)
        2. Submit all legs simultaneously
        3. Wait for fills (timeout: 30ms for MARKET)
        4. If all filled → Commit
        5. If incomplete → Rollback
        
        Returns:
        {
            "trade_id": "arb_123456_1",
            "legs": [...],
            "committed": True,
            "execution_time_ms": 28.5,
            "actual_profit": 19.50
        }
        """
        trade = self._create_trade(legs, strategy)
        await self._execute_strategy(trade)
        await self._finalize_trade(trade)
        return trade
```

### Executor → Platform

```python
class KalshiPlatform:
    async def place_order(
        self,
        market_id: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: int,
        price: Optional[Decimal] = None
    ) -> Order:
        """
        Places order on Kalshi:
        
        POST /trade-api/v2/portfolio/orders
        {
            "ticker": market_id,
            "action": side.value,
            "type": order_type.value,
            "count": quantity,
            "yes_price": price (if LIMIT)
        }
        
        Returns:
        {
            "order_id": "...",
            "status": "resting" | "filled" | ...,
            "filled_quantity": 0,
            ...
        }
        """
        response = await self.client.post("/orders", json=payload)
        return Order.from_dict(response)
```

---

## Database Schema

### PostgreSQL: Trade History

```sql
-- Trades table
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(64) UNIQUE NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    strategy VARCHAR(32) NOT NULL,  -- 'arbitrage', 'market_making', 'behavioral'
    status VARCHAR(16) NOT NULL,     -- 'committed', 'rolled_back'
    expected_profit DECIMAL(10, 2),
    actual_profit DECIMAL(10, 2),
    execution_time_ms DECIMAL(8, 2),
    num_legs INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_trades_strategy ON trades(strategy);

-- Trade legs table
CREATE TABLE trade_legs (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(64) REFERENCES trades(trade_id),
    platform VARCHAR(16) NOT NULL,   -- 'kalshi', 'polymarket'
    market_id VARCHAR(128) NOT NULL,
    side VARCHAR(4) NOT NULL,        -- 'YES', 'NO'
    quantity INT NOT NULL,
    target_price DECIMAL(6, 4),
    avg_fill_price DECIMAL(6, 4),
    filled_quantity INT,
    status VARCHAR(16) NOT NULL,     -- 'filled', 'cancelled', 'failed'
    order_id VARCHAR(64),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_legs_trade ON trade_legs(trade_id);
CREATE INDEX idx_legs_market ON trade_legs(market_id);

-- Positions table
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    market_id VARCHAR(128) NOT NULL,
    platform VARCHAR(16) NOT NULL,
    side VARCHAR(4) NOT NULL,
    quantity INT NOT NULL,
    avg_entry_price DECIMAL(6, 4) NOT NULL,
    current_price DECIMAL(6, 4),
    unrealized_pnl DECIMAL(10, 2),
    realized_pnl DECIMAL(10, 2) DEFAULT 0,
    opened_at TIMESTAMP NOT NULL,
    closed_at TIMESTAMP,
    status VARCHAR(16) DEFAULT 'open',  -- 'open', 'closed'
    UNIQUE(market_id, platform, status)  -- One open position per market
);

CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_market ON positions(market_id);

-- Daily performance table
CREATE TABLE daily_performance (
    date DATE PRIMARY KEY,
    total_trades INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,
    total_profit DECIMAL(10, 2) DEFAULT 0,
    total_volume DECIMAL(12, 2) DEFAULT 0,
    sharpe_ratio DECIMAL(6, 3),
    max_drawdown DECIMAL(6, 4),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Redis: Real-Time Cache

```
Key Structure:

# Orderbooks (TTL: 5s)
orderbook:polymarket:{asset_id} -> JSON(OrderBookSnapshot)
orderbook:kalshi:{market_id} -> JSON(OrderBookSnapshot)

# VWAP (TTL: 60s)
vwap:{asset_id}:100 -> "0.5234"  # $100 depth
vwap:{asset_id}:500 -> "0.5198"  # $500 depth
vwap:{asset_id}:1000 -> "0.5165" # $1000 depth

# Market metadata (TTL: 24h)
market:polymarket:{asset_id}:metadata -> JSON({
    "question": "...",
    "end_date": "...",
    "volume": 123456.78,
    "active": true
})

# Positions (TTL: 3600s)
position:{market_id} -> JSON({
    "quantity": 50,
    "avg_price": 0.52,
    "side": "YES"
})

# Strategy state (TTL: varies)
strategy:arbitrage:last_scan -> "1709427600"  # Unix timestamp
strategy:mm:inventory:{market_id} -> JSON({
    "yes_qty": 25,
    "no_qty": -10,
    "net": 15,
    "skew": 0.6
})

# Pub/Sub channels
Channel: polymarket:orderbook:{asset_id}
Channel: polymarket:trade:{asset_id}
Channel: kalshi:orderbook:{market_id}
```

---

## Deployment Architecture

### Development (Local)

```
┌─────────────────────────────────────────┐
│         MacBook (M1/M2/M3)              │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐  ┌──────────┐            │
│  │ Python   │  │  Redis   │            │
│  │ 3.11     │  │  Docker  │            │
│  └──────────┘  └──────────┘            │
│                                         │
│  ┌──────────────────────────┐          │
│  │   Mock Data Generators   │          │
│  └──────────────────────────┘          │
│                                         │
└─────────────────────────────────────────┘
```

**Cost:** $0/month

---

### Staging (AWS)

```
┌────────────────────────────────────────────────────┐
│                   AWS us-east-1                    │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌─────────────────────────────────────┐          │
│  │         EC2 t3.medium               │          │
│  │   - 2 vCPU, 4GB RAM                 │          │
│  │   - Ubuntu 22.04                    │          │
│  │   - Docker + Docker Compose         │          │
│  │                                     │          │
│  │   ┌──────────┐  ┌──────────┐       │          │
│  │   │ PR3DICT  │  │ Grafana  │       │          │
│  │   │ Container│  │ Container│       │          │
│  │   └──────────┘  └──────────┘       │          │
│  └────────────┬────────────────────────┘          │
│               │                                   │
│  ┌────────────▼────────────────┐                 │
│  │  ElastiCache (Redis)        │                 │
│  │  cache.t3.micro             │                 │
│  │  - 1 node, no replication   │                 │
│  └─────────────────────────────┘                 │
│                                                   │
│  ┌─────────────────────────────┐                 │
│  │  CloudWatch Logs            │                 │
│  │  - Application logs         │                 │
│  │  - Performance metrics      │                 │
│  └─────────────────────────────┘                 │
│                                                   │
│  ┌─────────────────────────────┐                 │
│  │  S3 Bucket                  │                 │
│  │  - Backups                  │                 │
│  │  - Historical data          │                 │
│  └─────────────────────────────┘                 │
│                                                   │
└────────────────────────────────────────────────────┘
```

**Cost:** ~$100/month
- EC2 t3.medium: $30/month
- ElastiCache: $15/month
- Data transfer: $10/month
- S3: $5/month
- CloudWatch: $10/month

---

### Production (AWS - High Availability)

```
┌──────────────────────────────────────────────────────────────┐
│                        AWS us-east-1                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │             Auto Scaling Group                       │   │
│  │                                                      │   │
│  │   ┌──────────────┐  ┌──────────────┐              │   │
│  │   │ EC2 Instance │  │ EC2 Instance │              │   │
│  │   │ c6i.xlarge   │  │ c6i.xlarge   │              │   │
│  │   │ (Primary)    │  │ (Standby)    │              │   │
│  │   │              │  │              │              │   │
│  │   │ - 4 vCPU     │  │ - 4 vCPU     │              │   │
│  │   │ - 8GB RAM    │  │ - 8GB RAM    │              │   │
│  │   │              │  │              │              │   │
│  │   └──────────────┘  └──────────────┘              │   │
│  │                                                      │   │
│  └─────────┬────────────────────────────────────────────┘   │
│            │                                                │
│  ┌─────────▼─────────────────────────────┐                 │
│  │  Application Load Balancer            │                 │
│  │  - Health checks every 10s            │                 │
│  │  - Auto-failover <30s                 │                 │
│  └───────────────────────────────────────┘                 │
│                                                             │
│  ┌───────────────────────────────────────┐                 │
│  │  ElastiCache (Redis)                  │                 │
│  │  cache.t3.small                       │                 │
│  │  - 2 nodes (primary + replica)        │                 │
│  │  - Multi-AZ                           │                 │
│  │  - Auto-failover enabled              │                 │
│  └───────────────────────────────────────┘                 │
│                                                             │
│  ┌───────────────────────────────────────┐                 │
│  │  RDS PostgreSQL 15                    │                 │
│  │  db.t3.small                          │                 │
│  │  - Multi-AZ                           │                 │
│  │  - Automated backups (7 days)         │                 │
│  │  - Point-in-time recovery             │                 │
│  └───────────────────────────────────────┘                 │
│                                                             │
│  ┌───────────────────────────────────────┐                 │
│  │  CloudWatch + Datadog                 │                 │
│  │  - Real-time dashboards               │                 │
│  │  - Alerting (PagerDuty)               │                 │
│  └───────────────────────────────────────┘                 │
│                                                             │
│  ┌───────────────────────────────────────┐                 │
│  │  S3 + Glacier                         │                 │
│  │  - Backups (15-minute intervals)      │                 │
│  │  - Long-term storage                  │                 │
│  └───────────────────────────────────────┘                 │
│                                                             │
└──────────────────────────────────────────────────────────────┘
```

**Cost:** ~$400/month
- EC2 c6i.xlarge ×2: $240/month
- ElastiCache t3.small HA: $50/month
- RDS t3.small Multi-AZ: $60/month
- Data transfer: $20/month
- Datadog: $15/month
- S3/Glacier: $10/month
- CloudWatch: $5/month

---

## Security Architecture

### Network Security

```
Internet
    │
    ▼
┌─────────────────┐
│   CloudFlare    │  ← DDoS protection
│   or AWS Shield │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   VPC Subnet    │  ← Private network
│   10.0.0.0/16   │
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌───────┐  ┌────────┐
│ Public│  │Private │
│Subnet │  │Subnet  │
│       │  │        │
│ ALB   │  │PR3DICT │
│       │  │Redis   │
│       │  │RDS     │
└───────┘  └────────┘
```

### Access Control

```
Role-Based Access:

1. Admin (full access)
   - Deploy code
   - Modify infrastructure
   - Access production DB
   - View all logs

2. Developer (limited)
   - Deploy to staging
   - Read production logs
   - No DB write access

3. Monitor (read-only)
   - View dashboards
   - Read logs
   - No deployment

4. Trading Bot (service account)
   - API keys for exchanges
   - Redis read/write
   - DB write (trades only)
   - No infrastructure access
```

### Secrets Management

```
AWS Secrets Manager:

secrets/pr3dict/production/kalshi_api_key
secrets/pr3dict/production/kalshi_api_secret
secrets/pr3dict/production/polymarket_private_key
secrets/pr3dict/production/telegram_bot_token
secrets/pr3dict/production/discord_webhook_url
secrets/pr3dict/production/redis_password
secrets/pr3dict/production/postgres_password

Rotation: Every 90 days (automated)
Access: IAM role-based (EC2 instance profile)
```

---

## Monitoring Architecture

### Metrics Collection

```
Application
    │
    ├──> StatsD ──────────────┐
    │                         │
    ├──> Prometheus ──────────┤
    │    (scrape every 15s)   │
    │                         ├──> Aggregation
    ├──> Custom Metrics ──────┤
    │    (via API)            │
    │                         │
    └──> Logs ────────────────┘
         (CloudWatch)

              │
              ▼
         ┌─────────┐
         │ Grafana │
         │Dashboard│
         └─────────┘
              │
              ▼
      ┌──────────────┐
      │  Alerting    │
      │  - PagerDuty │
      │  - Slack     │
      │  - Email     │
      └──────────────┘
```

### Key Dashboards

1. **System Health**
   - CPU, memory, disk
   - Network I/O
   - Redis hit rate
   - Database connections

2. **Trading Performance**
   - Opportunities/sec
   - Trades/sec
   - Win rate
   - P&L (real-time)
   - Sharpe ratio

3. **Latency Breakdown**
   - WebSocket → Cache
   - Strategy scan
   - Optimizer solve
   - Execution time
   - End-to-end

4. **Error Tracking**
   - Error rate by component
   - Failed trades
   - API errors
   - Timeout rate

---

## Disaster Recovery

### Backup Strategy

```
Level 1: Real-time (Redis)
- In-memory replication
- RDB snapshots every 5 minutes
- AOF (append-only file) enabled

Level 2: Short-term (S3)
- Database backups every 15 minutes
- Trade logs every 1 minute
- Configuration every 1 hour
- Retention: 7 days

Level 3: Long-term (Glacier)
- Daily full backups
- Monthly aggregated backups
- Retention: 7 years
```

### Recovery Procedures

```
Scenario 1: Single instance failure
- Auto-scaling launches new instance: <5 minutes
- Load balancer reroutes traffic: <30 seconds
- RTO (Recovery Time Objective): 5 minutes
- RPO (Recovery Point Objective): 0 (no data loss)

Scenario 2: Redis failure
- Failover to replica: <30 seconds
- RTO: 1 minute
- RPO: <5 seconds

Scenario 3: Region failure
- Manual failover to secondary region: <30 minutes
- RTO: 30 minutes
- RPO: <15 minutes (backup interval)

Scenario 4: Data corruption
- Restore from S3 backup: <10 minutes
- RTO: 10 minutes
- RPO: <15 minutes
```

---

**Architecture Version:** 2.0.0  
**Last Updated:** February 2, 2026  
**Status:** Production Ready (pending deployment)

---

*End of Architecture Diagram*
