# WT-E Evaluation + Paper Runner

Branch: `codex/resolution-eval`

## Scope
- Backtest and paper-trade evaluation for resolution advantage.

## First Tasks
1. Create `src/strategies/dispute/eval.py`.
2. Build calibration metrics (Brier/log loss) and hit-rate/EV reporting.
3. Create daily paper-trade summary generator.

## Exit Criteria
- CLI report generation works from stored artifacts.
- Includes per-model-version calibration slices.
- Includes failure buckets by signal reason code.
