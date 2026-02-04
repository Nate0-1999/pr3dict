# WT-C Tier 2 Deep Analysis

Branch: `codex/resolution-tier2`

## Scope
- Implement Tier 2 probability + decision-path output contract.

## First Tasks
1. Create `src/strategies/dispute/tier2.py` analyzer interface.
2. Enforce probability bounds and sum tolerance.
3. Enforce `no_trade_reason` for no-trade decisions.

## Exit Criteria
- Valid outputs persist.
- Invalid outputs are rejected with reason logging.
- Decision path taxonomy covered by tests.
