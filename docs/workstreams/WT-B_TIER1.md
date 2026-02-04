# WT-B Tier 1 Screen

Branch: `codex/resolution-tier1`

## Scope
- Implement fast Tier 1 screen with strict JSON contract.

## First Tasks
1. Create `src/strategies/dispute/tier1.py` runner interface.
2. Define schema validator for `PASS|FLAG`, scores, risks, and metadata.
3. Persist run metadata and Tier 1 output through WT-A contracts.

## Exit Criteria
- Malformed model output does not crash pipeline.
- PASS/FLAG behavior tested.
- Prompt/model/version metadata always stored.
