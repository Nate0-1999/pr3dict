# Polymarket Arbitrage Master Spec

Status: Active
Owner: PR3DICT arbitrage planning thread
Last updated: 2026-02-04

## Implementation Status
- M1 complete in branch lane: executable pricing, staleness checks, and risk-gate plumbing added in Python.
- M2 in progress: opportunity engine ports and execution lifecycle state machine scaffolded.
- M2 update: paired-leg entry contract now wired through engine execution path (paper + live fallback handling).

## Purpose
This is the single source of truth for Polymarket arbitrage planning and implementation decisions.

## Scope Lock
- Current: Polymarket-only arbitrage, paper mode, backtest/replay required
- Future hook: cross-platform expansion after v1 validation

## Locked Decisions
- Opportunity classes: binary complement + structural mispricing
- Net edge threshold: hard minimum 100 bps, preferred 120-150 bps
- Opportunity TTL: default 500 ms, adaptive by liquidity/volatility
- Slippage guardrails: 75 bps soft per leg, 100 bps hard per leg, 150 bps multi-leg hard
- Sizing: fractional Kelly (0.25x) with hard portfolio caps
- Partial fill policy: conservative immediate hedge/flatten
- Resolution thread dependency: consume `risk_multiplier` only in v1
- Release gate: 14-day forward paper + backtest/replay criteria

## Shared Contract Summary (v1)

### Opportunity
- Required fields: `opportunity_id`, `opportunity_type`, `market_id`, `side`, `quoted_price`,
  `executable_price`, `size_max_contracts`, `edge_bps_net`, `confidence`, `ttl_ms`,
  `created_at_ms`, `expires_at_ms`, `risk_multiplier`, `reasons[]`
- Rules: net edge must be post-fee and post-slippage estimate; `confidence` in [0,1].

### RiskDecision
- Required fields: `opportunity_id`, `decision` (`allow|adjust|deny`), `size_adjusted_contracts`,
  `reason_code`, `details`
- Minimum reason codes: `RISK_OK`, `RISK_DAILY_LOSS`, `RISK_EXPOSURE`, `RISK_SLIPPAGE`, `RISK_STALE`.

### ExecutionResult
- Required fields: `opportunity_id`, `status` (`filled|partial_fill|failed|rejected`), `filled_qty`,
  `avg_fill_price`, `slippage_bps`, `latency_ms`, `pnl_estimate`, `reject_reason`, `reason_code`
- Minimum reason codes: `EXEC_OK`, `EXEC_TIMEOUT`, `EXEC_PARTIAL`, `EXEC_STALE`, `EXEC_SLIPPAGE`, `EXEC_PLATFORM`.

## Milestones
- M0: Master spec freeze
- M1: Data and risk plumbing
- M2: Opportunity engine + execution state machine
- M3: Replay/backtest + paper telemetry
- M4: v1 gate review and launch-readiness memo

## Change Control
Any change to locked decisions requires:
1. Add an entry to Decision History (below)
2. Update Shared Contract Summary if schema-affecting
3. Add a short implementation note in the relevant PR description

## Non-Goals for v1
- Live capital auto-trading
- Cross-platform smart order routing
- Full adjudication model ingestion

## Decision History

### 2026-02-04 | DEC-001 | Polymarket v1 scope
- Decision: Scope v1 to Polymarket-only, paper mode, with backtest/replay.
- Why: Reduce integration complexity and unblock parallel implementation.

### 2026-02-04 | DEC-002 | Execution guardrails
- Decision: Hard edge floor 100 bps net, TTL 500 ms default, hard slippage caps.
- Why: Prevent quoted-edge mirage and reduce execution-driven drawdowns.

### 2026-02-04 | DEC-003 | Resolution dependency boundary
- Decision: Consume only `risk_multiplier` from resolution/adjudication thread in v1.
- Why: Keep arb thread decoupled while allowing risk-aware sizing.
