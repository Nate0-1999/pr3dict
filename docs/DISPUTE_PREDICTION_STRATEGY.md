# Dispute Prediction Strategy

**PR3DICT: Polymarket Resolution Edge**

---

## Core Thesis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           THE DISPUTE EDGE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Traditional Trader:                                                        │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │ Predict      │ ───► │ Trade        │ ───► │ Wait for     │              │
│  │ Real Event   │      │ YES/NO       │      │ Resolution   │              │
│  └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                                             │
│  OUR APPROACH:                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │ Predict      │ ───► │ Position     │ ───► │ Profit from  │              │
│  │ RESOLUTION   │      │ Accordingly  │      │ Resolution   │              │
│  │ PROBLEMS     │      │              │      │ Chaos        │              │
│  └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                                             │
│  WHY THIS WORKS:                                                            │
│  • Resolution ambiguity creates price volatility                            │
│  • UMA voting is predictable (whale patterns, historical bias)              │
│  • Most traders don't model this risk → mispricing                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How Polymarket Works

### Market Creation

Anyone can create a Polymarket by defining:

1. **Question** — A yes/no question (e.g., "Will Bitcoin reach $100K by March 1, 2026?")
2. **Resolution Rules** — Detailed criteria for what constitutes YES vs NO
3. **Resolution Source** — The authoritative source for determining the outcome (e.g., CoinGecko price, AP news call)
4. **End Date** — When the market closes and resolution begins

Once created, traders buy YES or NO shares. Prices reflect the market's probability estimate. If YES wins, YES shares pay $1 each, NO shares pay $0 (and vice versa).

### Resolution Process (UMA Optimistic Oracle)

When a market's end date passes:

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Market    │────►│   PROPOSAL      │────►│   SETTLEMENT    │
│   Closes    │     │   (2-4 hr wait) │     │   (Payout)      │
└─────────────┘     └────────┬────────┘     └─────────────────┘
                             │
                    Anyone disputes?
                             │
                    ┌────────┴────────┐
                    │                 │
                   NO                YES
                    │                 │
                    ▼                 ▼
            ┌─────────────┐   ┌─────────────────────┐
            │  Proposal   │   │   UMA DVM VOTE      │
            │  Accepted   │   │   (48-96 hours)     │
            │             │   │                     │
            │  Normal     │   │   UMA token holders │
            │  payout     │   │   vote on correct   │
            └─────────────┘   │   outcome           │
                              │                     │
                              │   Result is FINAL   │
                              └─────────────────────┘
```

**Step by step:**

1. **Proposal** — Anyone can propose an outcome (YES, NO, or INVALID). Proposer posts a bond (~$1,500 USDC).

2. **Challenge Window** — 2-4 hours where anyone can dispute by posting a counter-bond.

3. **If No Dispute** — Proposal is accepted, market settles, traders get paid.

4. **If Disputed** — Escalates to UMA's Data Verification Mechanism (DVM). UMA token holders vote over 48-96 hours. The majority vote determines the outcome. Winning side gets losing side's bond.

### Why Disputes Happen

- **Ambiguous contract language** — "Will X happen?" but X isn't precisely defined
- **Edge cases** — Something happened that the contract didn't anticipate
- **Resolution source issues** — Source is unavailable, contradictory, or changed
- **Timing ambiguity** — Timezone issues, exact moment unclear
- **Subjective criteria** — Contract requires interpretation

---

## Four Ways to Capitalize

| # | Strategy | Trigger | Our Edge |
|---|----------|---------|----------|
| 1 | **Pre-Dispute Detection** | Market approaching close | Identify dispute-prone markets before others |
| 2 | **Post-Proposal Detection** | Proposal submitted, window open | Predict incoming dispute, exit/hedge |
| 3 | **Active Dispute Trading** | DVM voting in progress | Predict how UMA voters will rule |
| 4 | **Dispute Initiation** | We identify incorrect proposal | File dispute ourselves, profit from correction |

### Strategy 1: Pre-Dispute Detection

Scan markets approaching close. If our analysis says P(dispute) > `DISPUTE_THRESHOLD` (likely ~30%) and we can predict the DVM outcome, take a position before liquidity disappears.

### Strategy 2: Post-Proposal Detection

Monitor proposals during challenge window. If signals suggest dispute is imminent, exit unfavorable positions or enter at distressed prices.

### Strategy 3: Active Dispute Trading

Once a dispute is filed, predict how UMA token holders will vote. If we have high confidence (likely ~70%+) and the market is mispriced, trade the expected DVM outcome.

### Strategy 4: Dispute Initiation

The highest-edge opportunity: when we identify that a proposed outcome is **wrong** per the contract terms. We position, file the dispute ourselves (posting bond), and profit when the DVM rules in our favor. Requires high confidence (likely ~75%+) since we're risking the bond.

---

## The Analysis Engine

### Approach: LLM-First

We use modern LLMs with structured thinking to analyze markets, not traditional NLP or feature engineering. The workflow:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ANALYSIS PIPELINE                                 │
│                                                                          │
│   Every ~6 hours:                                                        │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Fetch markets in target range:                                 │   │
│   │  • Liquidity > MIN_LIQUIDITY (likely ~$5K)                      │   │
│   │  • Closes within DAYS_TO_CLOSE (likely ~14 days)                │   │
│   │  • Not already analyzed this cycle                              │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  TIER 1: Fast LLM Screening                                     │   │
│   │  ─────────────────────────                                      │   │
│   │  Quick pass with lighter model (Claude Haiku / GPT-4o-mini)     │   │
│   │  "Does this contract have obvious ambiguity or edge cases?"     │   │
│   │                                                                 │   │
│   │  Output: PASS (low risk) or FLAG (needs deeper review)          │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                    ┌─────────┴─────────┐                                │
│                    │                   │                                │
│                  PASS                FLAG                               │
│                    │                   │                                │
│                    ▼                   ▼                                │
│            Log & skip         ┌───────────────────────────────────┐    │
│                               │  TIER 2: Deep Analysis            │    │
│                               │  ────────────────────             │    │
│                               │  Extended thinking model          │    │
│                               │ (Claude Opus/gemini 3 pro thinking max)│    
│                               │                                   │    │
│                               │  Analyze:                         │    │
│                               │  • Contract ambiguities           │    │
│                               │  • Edge cases not covered         │    │
│                               │  • Resolution source reliability  │    │
│                               │  • Historical similar disputes    │    │
│                               │  • Likely DVM outcome if disputed │    │
│                               │                                   │    │
│                               │  Output:                          │    │
│                               │  • P(dispute)                     │    │
│                               │  • P(YES|DVM), P(NO|DVM)          │    │
│                               │  • Confidence                     │    │
│                               │  • Reasoning                      │    │
│                               └───────────────────────────────────┘    │
│                                          │                              │
│                                          ▼                              │
│                               ┌───────────────────────────────────┐    │
│                               │  Signal Generation                │    │
│                               │  IF P(dispute) > DISPUTE_THRESHOLD│    │
│                               │  AND confidence > MIN_CONFIDENCE  │    │
│                               │  AND edge vs market price exists  │    │
│                               │  → Generate trade signal          │    │
│                               └───────────────────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Why LLM-First Works Here

Contract ambiguity analysis is fundamentally a **language understanding** problem. Modern LLMs excel at:
- Identifying vague or undefined terms
- Reasoning about edge cases
- Understanding intent vs literal wording
- Comparing to similar past situations (via RAG)

Traditional NLP (keyword counting, sentiment scores) misses nuance. A contract might have zero "ambiguous" keywords but still have a critical undefined edge case that an LLM can reason about.

### RAG Knowledge Store

We maintain a vector database of:
- Past market analyses (what we predicted, what happened)
- Historical disputes (contract text, dispute arguments, DVM outcomes)
- Our learnings (why we were wrong, what patterns we've found)

When analyzing a new market, we retrieve similar past cases to inform the LLM's analysis. This creates a learning loop.

---

## Position Sizing

We use **Kelly Criterion** — bet a fraction of bankroll proportional to your edge. This maximizes long-term growth while avoiding ruin. We apply fractional Kelly (~25%) with additional discounts for prediction confidence and P(INVALID). Capped at `MAX_POSITION_PCT` (likely ~5%) per trade.

See [Appendix: Kelly Criterion](./APPENDIX_KELLY_CRITERION.md) for full derivation and theory.

---

## Parameters Summary

| Parameter | Description | Likely Value |
|-----------|-------------|--------------|
| `SCAN_INTERVAL` | How often to fetch and screen markets | ~6 hours |
| `MIN_LIQUIDITY` | Minimum market liquidity to consider | ~$5,000 |
| `DAYS_TO_CLOSE` | How far out to scan for approaching markets | ~14 days |
| `DISPUTE_THRESHOLD` | Minimum P(dispute) to generate signal | ~30% |
| `MIN_CONFIDENCE` | Minimum model confidence to act | ~60% |
| `DVM_CONFIDENCE_THRESHOLD` | Confidence needed to trade DVM outcome | ~70% |
| `INITIATION_CONFIDENCE` | Confidence needed to file dispute ourselves | ~75% |
| `KELLY_FRACTION` | Fraction of full Kelly to use | ~25% |
| `MAX_POSITION_PCT` | Maximum position as % of bankroll | ~5% |

---

## Next Steps

1. **Build data pipeline** — Gamma API fetcher, store markets in Postgres
2. **Build Tier 1 screener** — Fast LLM pass to flag ambiguous contracts
3. **Build Tier 2 analyzer** — Extended thinking analysis with RAG
4. **Backtest on historical disputes** — How well would we have predicted?
5. **Paper trade** — Run live without real capital

---

*Document Version: 0.3*  
*Last Updated: 2026-01-22*
