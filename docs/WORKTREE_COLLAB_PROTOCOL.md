# Worktree Collaboration Protocol

Use this when multiple threads/agents work in parallel on the same repo.

## Baseline Rules

1. Keep root workspace as a control surface (status checks only).
2. Every stream gets:
   - dedicated branch
   - dedicated worktree
3. Do not mix unrelated changes in one stream.
4. Merge dependency-first.
5. Rebase each stream onto `main` after dependency merges.

## Branch/Worktree Map

- `codex/resolution-data-contracts` -> `.worktrees/wt-a`
- `codex/resolution-tier1` -> `.worktrees/wt-b`
- `codex/resolution-tier2` -> `.worktrees/wt-c`
- `codex/resolution-signals` -> `.worktrees/wt-d`
- `codex/resolution-eval` -> `.worktrees/wt-e`
- `codex/resolution-baseline` -> `.worktrees/wt-baseline` (clean integration lane)

## Merge Order

1. data contracts/schema
2. tier1 + tier2
3. signal engine/sizing
4. eval/reporting

## Copyable Status Snippet (for other thread)

```text
Parallel execution setup is active.

We are preserving all existing work and isolating streams with dedicated branches/worktrees:
- wt-a: codex/resolution-data-contracts
- wt-b: codex/resolution-tier1
- wt-c: codex/resolution-tier2
- wt-d: codex/resolution-signals
- wt-e: codex/resolution-eval
- wt-baseline: codex/resolution-baseline (clean lane from origin/main)

Dependency merge order:
1) data contracts
2) tier1 + tier2
3) signals
4) eval

Please do not work in the root workspace; use your assigned worktree/branch only.
```
