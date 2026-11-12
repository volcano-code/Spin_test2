# M80 Large-Training Release Gate

A model is not called large-training-ready until every applicable gate passes.

## Gate 1 — Source and CI
- Clean, reproducible branch and pinned source fingerprints.
- Full CPU mathematical/data tests pass.
- Legacy M80.1 checkpoints load unchanged.
- New multiscale/GNO/Transolver checkpoint round-trip passes.
- Training CLI and exact-resume contract pass.

## Gate 2 — Formal data
- Frozen formal manifest and audit receipt.
- Correct fingerprint and file SHA-256 for every record.
- Independent source-group split with no leakage.
- At least 256 diverse, non-equivalent queries per qualified record.
- Derivative-valid labels independently audited.
- Event/invalid labels retained but masked from Jacobian regression.

## Gate 3 — GPU preflight
On the largest actual training geometry and the selected preset:
- forward and full Jacobian backward;
- AdamW update;
- BF16 and sensitive FP32 paths;
- peak allocated/reserved memory;
- checkpoint save/reload;
- exact resume;
- finite gradients and outputs.

## Gate 4 — Pilot
- 200–500 step engineering run;
- per-geometry validation tails;
- at least three seeds for promotion;
- no test-set-driven tuning;
- no unexplained event/derivative failures.

## Gate 5 — Long training
- freeze code, data, config, environment and seeds;
- run 10k pilot;
- inspect learning curves and validation tails;
- only then extend to 40k–100k steps or a fixed GPU-hour budget.

## Gate 6 — Solver and physical acceptance
Required before claiming project success, but not a prerequisite for a standalone operator training pilot:
- native Newton bridge parity;
- event-safe derivative admission;
- exact correction and fallback;
- independent final moment/constraint/material audit;
- end-to-end time and exact-call accounting.

Current status: M80.4 multi-scale and topology-aware branches are under CPU validation; their GPU preflight and trained-quality comparisons are pending. Formal 256-query corpus and Newton bridge are not yet complete. Do not start multi-day training.
