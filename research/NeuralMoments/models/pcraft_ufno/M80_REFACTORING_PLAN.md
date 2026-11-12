# M80 Refactoring Plan — Spin Neural Moments

Date: 2026-11-13  
Repository: \`volcano-code/Spin_test2\`  
Working branch: \`research/m80-pnmo-large-train-20261112\`

## 1. Research objective

Use a P-CRAFT-dominant, geometry-aware physical neural operator to accelerate the native Spin-It Faster computation while preserving the original task, geometry, rotation axis, material policy, contact/fixed-part conditions and terminal physical acceptance. The neural model is a proposal/prediction component; the native exact evaluator and solver remain the correctness authority.

The two distinct research benefits must be reported separately:

1. **Same accuracy, lower total cost:** lower end-to-end time at the same terminal constraints and acceptance tolerance.
2. **Same budget, better feasible design:** lower material mass at the same constraints and a fixed compute/teacher-call budget.

Neither benefit is established by a neural preflight alone.

## 2. Current implementation snapshot

| Component | Current state | Evidence boundary |
|---|---|---|
| P-CRAFT single-scale Native operator | Implemented; permanent baseline | Large 512 GPU preflight passed |
| M72 native RLE interval path | Implemented | 194,275-interval object passed the reported preflight |
| GNO local branch | Implemented, optional | Single-scale GPU preflight passed; quality benefit unproven |
| Transolver++-style physical slicing | Implemented, optional | Single-scale GPU preflight and CPU contracts passed; quality benefit unproven |
| GNO + Transolver | Implemented, optional | GPU resource preflight passed; training benefit unproven |
| Shared-weight 32/64/128 P-CRAFT pyramid | Implemented as opt-in M80.4 presets | CPU CI passed at the latest recorded successful run; GPU preflight pending |
| 6-connected GNO topology filter | Implemented as opt-in for new multiscale-GNO presets | CPU contracts included in the M80.4 test run; GPU impact pending |
| Validation P50/P95/P99 | Implemented in training validation reports | CPU CI passed; local report not yet returned |
| Optional event/derivative audit label fields | Reader support implemented | Final CI rerun pending at time of this document |
| Convex moment potential head | Not implemented | Research design only |
| Quadrature / QuadrANN head | Not production-integrated | Prior adapter requires re-audit |
| UQ calibration / active learning | Not implemented | Research design only |
| Spin-It Faster Newton bridge | Not implemented | Must inspect and bind to native solver |
| Adaptive exact correction / fallback | Not implemented | Specification only |
| Formal 67 × 256 query corpus | Not confirmed | Existing corpus requires fresh audit/expansion |
| Multi-day training | Not started | Explicitly blocked |

The four previously supplied single-scale Large preflights are resource checks after one optimizer update, not trained-quality comparisons.

## 3. Architecture decisions

- Keep M80.1 single-scale P-CRAFT + M72 Native as a permanent baseline.
- Keep GNO and Transolver independently switchable.
- Do not promote GNO or Transolver from preflight cost alone.
- M80.4 adds a shared-weight P-CRAFT multi-resolution path at 32/64/128 and an optional topology-aware GNO path.
- Do not yet add GeoTransolver, FNOGNO, DoMINO or FIGConvNet to the production model. Compare them only after the current P-CRAFT family has a trained-quality baseline.
- Do not merge the convex-potential, QuadrANN and UQ work into one head before each has independent mathematical tests.

## 4. Milestones

### M80.4 — Architecture and evaluation hardening
1. Freeze the single-scale baseline and checkpoint compatibility.
2. Implement shared-weight multi-resolution P-CRAFT feature extraction.
3. Add learned global-scale and native-interval scale gates.
4. Add 6-connected native RLE component filtering for GNO neighborhoods.
5. Report per-query Moment/Jacobian tails, not only aggregate MSE.
6. Correct status and candidate documents from historical W6 state to actual M80 evidence.
7. Run CPU CI, then local 5090D preflight for the new presets.

Exit: CPU tests pass; all selected multi-scale presets pass 5090D preflight; no baseline regression.

### M80.5 — Physical readout research
1. Keep the current bounded odd M72 head as the control.
2. Implement a separate convex-potential candidate with explicit coefficient scaling and complement handling.
3. Compare AD Jacobian symmetry/PSD only on certified smooth neighborhoods.
4. Re-audit the previous QuadrANN adapter; compare fixed quadrature, learned nonnegative weights and exact-corrected learned weights.
5. Add HVP through AD from the selected potential/response function before considering a separate learned HVP head.
6. Keep Cherchi as the authoritative Spin teacher and Algoim as an independent numerical comparator.

Exit: moment, Jacobian, directional derivative and smooth-event contracts pass against independent teacher cases.

### M80.6 — Formal data V2
1. Re-audit all 67 qualified objects, label hashes and derivative-valid neighborhoods.
2. Generate at least 256 diverse, non-equivalent queries per qualified object.
3. Preserve event/invalid-derivative samples as event labels; never turn them into valid Jacobian labels.
4. Add provenance, teacher precision, event type/margin and derivative audit metadata.
5. Expand independent geometry groups beyond the current 67-object baseline after the first frozen R2 corpus passes.

Exit: frozen manifest, audit receipt, no source-group leakage and complete per-record query coverage.

### M80.7 — UQ and active teacher selection
1. Calibrate UQNO/ensemble uncertainty on held-out geometries and queries.
2. Include event margin and derivative validity as separate safety signals.
3. Implement teacher-budget-aware active selection.
4. Compare active selection with random and stratified sampling at equal teacher-call budgets.

Exit: risk-coverage and teacher-call savings reported without false derivative acceptance.

### M80.8 — Native Newton bridge
1. Inspect and pin the exact Spin-It Faster native solver source and its coefficient/derivative conventions.
2. Build a parity adapter for native moments, objective, constraints, Jacobian and stopping rules.
3. Test learned initialization separately from learned moment/Jacobian substitution.
4. Add residual, trust-region, event, derivative and uncertainty gates.
5. Count every exact correction and fallback in online time.
6. Preserve the native exact solver as the fallback and final authority.

Exit: same-task replay passes independent terminal audit and reports total wall time, exact calls, Newton iterations and failure rates.

### M80.9 — Performance engineering
Evaluate fVDB, Transolver-3 and additional sparse backends only if measured RLE, memory, graph or feature-map costs justify them. Compare cold preprocessing, cached geometry encoding, warm moment query, warm Jacobian query and end-to-end solver time.

### M80.10 — Paper-grade validation
Use multiple geometry families, held-out parent objects, arbitrary SO(3) orientations, thin-wall/disconnected cases, 512+ native resolution and fixed physical constraints. Report failures and all ablations.

## 5. Non-negotiable guardrails

- No long training until formal data, GPU, checkpoint/resume and validation gates pass.
- No unlicensed source or weights are copied.
- No test split is used for model selection.
- No learned derivative is admitted near an unknown/non-smooth event.
- No exact correction or fallback is omitted from online timing.
- No claim of CCF-A readiness is made from architecture size, preflight, or training loss alone.


## M80.5 implementation update — 2026-11-13

A separate opt-in convex homogeneous moment potential head has been implemented as a research candidate. It uses a positive sum of geometry-conditioned SPD norms, derives editable moments by AD, preserves the q/-q complement and positive-scale invariance, and bounds its response by the native absolute-moment envelope. The existing bounded odd M72 head remains the default/control.

CPU tests for Jacobian symmetry/PSD, finite differences, complement, scale invariance, envelope and checkpoint round-trip passed:
https://github.com/volcano-code/Spin_test2/actions/runs/43277790530

The candidate has not yet passed a 5090D preflight, trained-quality comparison or exact-teacher validation. QuadrANN remains a separate next experiment; it is not merged into this head.


## M80.8 bridge implementation update — 2026-11-13

Repository inspection found an existing strict native external-moment-evaluator callback in python/make_it_stand/spin.py and bindings/pybind_module.cpp. An initial M80 callback adapter has been added at spin_pino/neural_solver_bridge.py. It applies explicit coefficient/moment transforms, checks cache identity/version, requires an admission callback, and falls back to the exact evaluator by default.

CPU tests pass:
https://github.com/volcano-code/Spin_test2/actions/runs/43279718463

This is an adapter prototype only. It has not yet been bound to a real SpinItFasterRequest/source mesh, target-frame transforms have not been validated on a nontrivial frame, and no real geometry Newton replay or final material audit has been completed.
