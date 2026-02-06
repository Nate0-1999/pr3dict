# Resolution Advantage Worktree Plan

Execution-ready implementation plan for the dispute strategy using parallel git worktrees.

## Scope

This plan applies only to Polymarket dispute/resolution advantage components.

Out of scope:
- Arbitrage, market making, or other strategy rewrites
- Production real-money auto-execution

## Current Status (2026-02-05)

- Completed and merged:
  - WT-A schema + data contracts
  - WT-B Tier 1 contract scaffold
  - WT-C Tier 2 contract scaffold
  - WT-D signal/sizing primitives
  - WT-E evaluation helpers
- New baseline integration step added:
  - Tier 1/Tier 2 persistence wiring to WT-A contracts via `src/strategies/dispute/pipeline.py`
  - Persistence integration tests in `test_dispute_pipeline_persistence.py`
  - Tier 2 probability simplex policy:
    - normalize when `abs(sum-1.0) <= 0.01`
    - reject as invalid when drift exceeds `0.01`
- Next focus:
  - connect live Tier 1/Tier 2 model runners to `persist_tier1_result` / `persist_tier2_result`
  - add signal emission persistence from `signal_engine` into `signals`

## Workstream Layout

## WT-A: Data Contracts + Schema

Branch: `codex/resolution-data-contracts`

Owns:
- `src/data/database.py`
- dispute schema migration helpers

Delivers:
- New tables:
  - `analysis_runs`
  - `analysis_outputs_t1`
  - `analysis_outputs_t2`
  - `signals`
  - `market_outcomes`
  - `calibration_metrics`
- Backward-compatible reads for existing `markets`/`analyses`
- DB-level constraints for probability bounds where practical

Definition of done:
- Can replay one signal deterministically from stored artifacts
- Migration path works on fresh DB and existing DB
- Unit tests for insert/retrieve/validation pass

## WT-B: Tier 1 Screen

Branch: `codex/resolution-tier1`

Owns:
- `src/strategies/dispute/tier1.py`
- Tier 1 prompt contract + schema validation

Consumes:
- WT-A tables (`analysis_runs`, `analysis_outputs_t1`)

Delivers:
- `screen_decision` (`PASS|FLAG`)
- `ambiguity_score`, `dispute_prob_prior`, `top_risks[]`
- strict JSON validation + rejection logging

Definition of done:
- Invalid model output never crashes pipeline
- Deterministic `prompt_version` and `run_id` persisted
- Tests cover PASS, FLAG, malformed output, retry fallback

## WT-C: Tier 2 Deep Analysis

Branch: `codex/resolution-tier2`

Owns:
- `src/strategies/dispute/tier2.py`
- Decision-path + probability normalization rules

Consumes:
- WT-A tables
- WT-B `FLAG` outputs

Delivers:
- `p_dispute`
- `p_yes_final`, `p_no_final`, `p_invalid_final`
- `confidence`, `decision_path`, `no_trade_reason`, `assumptions[]`
- hard checks for probability bounds and sum tolerance

Definition of done:
- Sum tolerance enforced (example: `abs(sum-1.0) <= 0.01`)
- `no_trade_reason` always populated for no-trade paths
- Contract tests validate all decision paths

## WT-D: Signal + Sizing

Branch: `codex/resolution-signals`

Owns:
- `src/strategies/dispute/signal_engine.py`
- dispute-specific sizing module (no generic risk rewrite)

Consumes:
- WT-C normalized probabilities
- market snapshot prices/liquidity/spread

Delivers:
- Actions: `ENTER_YES`, `ENTER_NO`, `EXIT`, `HOLD`, `NO_TRADE`
- Edge math after fees/slippage haircuts
- Stop-loss parameterized trigger (`STOP_LOSS_PCT`) attached to entry signals
- Fractional Kelly sizing with confidence/invalid/liquidity discounts
- Cap enforcement:
  - `MAX_POSITION_PCT`
  - `MAX_MARKET_EXPOSURE_USD`
  - `MAX_STRATEGY_EXPOSURE_USD`

Definition of done:
- Every signal has reason code + snapshot context
- No emitted size exceeds caps
- Tests cover low-liquidity rejection and invalid-risk discount

## WT-E: Evaluation + Paper Runner

Branch: `codex/resolution-eval`

Owns:
- `src/strategies/dispute/eval.py`
- report generators for daily paper-trading summaries

Consumes:
- WT-A/B/C/D artifacts

Delivers:
- Backtest metrics:
  - calibration (Brier/log loss)
  - hit rate and realized EV
  - failure buckets by reason code
- Daily paper-trade summary output

Definition of done:
- Can run on historical settled markets dataset
- Report generated from CLI without manual editing
- Includes per-model-version calibration slice

## Interface Freeze Points

Freeze these interfaces before dependent merges:

1. WT-A schema names/columns/indexes
2. WT-B output JSON schema
3. WT-C output JSON schema and decision taxonomy
4. WT-D signal event schema

If an interface changes after freeze, owning stream must:
- publish changelog note in PR description
- provide adapter or migration in same PR

## Merge Order

1. WT-A
2. WT-B and WT-C (parallel, then merge)
3. WT-D
4. WT-E

Each stream merges via repo policy:
- feature branch -> fork `main` PR (approved)
- fork `main` -> upstream `main` PR

## Test Matrix (Minimum)

- Unit:
  - probability validation
  - schema persistence
  - sizing cap enforcement
- Integration:
  - scanner -> Tier 1 -> Tier 2 -> signal path
- Regression:
  - existing scanner/database behavior unchanged for current CLI usage

## Initial Parameter Defaults (for paper mode)

- `MIN_LIQUIDITY=5000`
- `DAYS_TO_CLOSE=14`
- `DISPUTE_THRESHOLD=0.30`
- `MIN_CONFIDENCE=0.60`
- `DVM_CONFIDENCE_THRESHOLD=0.70`
- `INITIATION_CONFIDENCE=0.75`
- `KELLY_FRACTION=0.25`
- `MAX_POSITION_PCT=0.05`
- `STOP_LOSS_PCT=0.15`

All values remain config-driven and easy to tune after backtesting.
