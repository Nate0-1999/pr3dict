# WT-D Signal + Sizing

Branch: `codex/resolution-signals`

## Scope
- Build edge calculation and position sizing with risk caps.

## First Tasks
1. Create `src/strategies/dispute/signal_engine.py`.
2. Implement edge after fee/slippage haircut.
3. Implement fractional Kelly sizing with confidence/invalid/liquidity discounts.
4. Enforce hard exposure caps.

## Exit Criteria
- Emits `ENTER_YES`, `ENTER_NO`, `EXIT`, `HOLD`, `NO_TRADE`.
- Every signal has reason code + snapshot context.
- Cap enforcement tests pass.
