# WT-A Data Contracts + Schema

Branch: `codex/resolution-data-contracts`

## Scope
- Extend SQLite schema for resolution advantage pipeline.
- Preserve backward compatibility for existing scanner workflows.

## First Tasks
1. Add tables: `analysis_runs`, `analysis_outputs_t1`, `analysis_outputs_t2`, `signals`, `market_outcomes`, `calibration_metrics`.
2. Add DAO methods for inserts/selects used by Tier 1/Tier 2/signal engine.
3. Add tests for probability bounds and deterministic replay.

## Exit Criteria
- Fresh DB init succeeds.
- Existing DB migration path succeeds.
- Replay of one signal from stored artifacts is deterministic.
