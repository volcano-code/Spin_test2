# M80 implementation plan — P-CRAFT-dominant native hybrid

## Objective

Create the strongest empirically selected operator for accelerating Spin-It Faster while preserving the Spin-It task contract. P-CRAFT is the dominant architectural foundation, but module selection is evidence-driven. The target is not a larger standalone moment regressor; it is a safe learned moment/Jacobian service inside the analytic solver loop.

## Architecture contract

### Geometry encoder

- Primary: P-CRAFT factorized Fourier U-shaped operator.
- Native detail: lossless interval/segment representation from the authoritative native geometry package; no replacement by an aggressively pooled 16^3/32^3 proxy.
- Fusion: multiscale P-CRAFT features query/condition interval tokens; interval-to-grid aggregation must be conservative and its error measured. A second residual route retains interval details.
- Cache: geometry-only encoding may be cached only for an immutable geometry hash + model-weight hash + preprocessing contract. Any optimizer update, geometry/material change or transform invalidates the cache.

### Optional experts (each independently switchable)

- GNO/local geometry interaction: only if it improves boundary/thin-wall held-out metrics at acceptable cost.
- Volume-weighted physical-state attention: optional; M8 Transolver++ had negligible gain with materially higher cost in the restricted P0 study, so no default promotion without a new matched result.
- Quadrature proposal head: optional; positive weights and exact-reference checks required. It must not self-authorize its own labels.
- UQ head: optional; calibrate on held-out data for the exact predictor version. UQ score is not a derivative-validity certificate.
- DensiCrafter: separate offline density-design teacher, not part of the production geometry encoder.
- TRELLIS/TRELLIS.2: geometry source only; generated assets require provenance, mesh checks, canonicalization, and native teacher labels before admission.

## Physical response contract

Use one canonical response service:
- variable-domain raw moments `r_D(q) in R^10`;
- Jacobian `J_D(q) = dr_D/dq in R^(10x10)` only in certified smooth neighborhoods;
- fixed material moments `r_F` are separately computed/cached;
- total moments are composed under the native task contract, not by changing the meaning of `r_D`;
- coefficient transforms, moment transforms and chain rules are explicit and unit-tested;
- unknown, singular, tangent, topological-change or otherwise uncertified derivative neighborhoods withhold the learned Jacobian and trigger exact evaluation.

Preferred design: a differentiable shared moment function whose AD Jacobian is the default learned derivative. Any separate derivative head is only an auxiliary proposal and must be checked against AD, finite differences and exact teacher labels. Newton's required second-order information must be derived from the actual selected optimality equations; do not assume a first derivative is sufficient.

## Solver bridge

- Keep the authoritative exact native evaluator as teacher and fallback.
- Preserve the existing Spin-It Faster equality-Newton implementation as a separately named path.
- Keep general inequality/active-set constraints in a general constrained solver until the Newton extension is independently validated.
- Add residual, step, event, derivative-neighborhood and trust-region gates.
- Any correction needed during optimization counts in online wall time.
- Final independent audit remains mandatory and is reported separately.

## Required matched experiments

1. Native States baseline vs P-CRAFT-only.
2. P-CRAFT + native interval detail.
3. P-CRAFT + GNO (optional).
4. P-CRAFT + physical-state attention (optional).
5. Best combination selected on validation only.

Same geometry splits, teacher-query budget, optimizer updates, seeds and validation policy. Report parameter count, FLOPs or measured throughput, peak allocated/reserved memory, moment error, Jacobian directional error, event false-accept rate, and downstream solver success/time/material quality.

Do not promote a module based on a small analytic P0 score alone.

## Data admission

Keep the existing Spin_test2 analytic P0 dataset as a method-development benchmark. For formal training, ingest native teacher records through a new versioned adapter with explicit provenance. Preserve train/validation/test geometry-group separation. Expand both query coverage and independent geometry count. A query expansion does not create new independent geometries.

Every record must preserve source geometry hash, task hash, coordinate frame, material policy, coefficient basis/version, moment order/version, teacher executable/source hash, precision, quadrature/integration settings, derivative audit status and split group.

## Training gates

### Gate A — CPU mathematical parity

- canonical geometry and coefficient transforms;
- moment ordering and fixed/variable material composition;
- AD Jacobian vs finite differences in smooth analytic cases;
- rotation/coordinate transform tests;
- event/invalid derivative rejection;
- positive-measure/support tests.

### Gate B — native-data adapter

- read actual Spin_test2-compatible native records;
- reproduce oracle moments and labels before training;
- verify all referenced assets and provenance;
- demonstrate no split leakage;
- no skipped/relabeled failed derivatives.

### Gate C — RTX 5090D preflight

On the largest actual training geometry:
- forward;
- moment loss;
- Jacobian loss and required higher-order graph;
- complete backward;
- AdamW step;
- peak allocated/reserved memory;
- checkpoint save/reload and exact output parity;
- repeat with configured AMP and FP32 sensitive paths.

### Gate D — staged pilot

- 8-geometry engineering pilot first;
- then formal corpus pilot;
- 200–500 steps with full validation;
- verify validation moment/Jacobian tails and downstream solver replay.

### Gate E — long training

Only after A–D:
- freeze code/data/config hashes;
- compare XL/XXL under matched budgets;
- run multiple seeds;
- start 10k-step pilot, then extend to 40k–100k steps or GPU-hour budget based on measured learning curves.

## Exit criteria for "large-training ready"

A release may be called large-training ready only when:
1. all required tests pass;
2. the formal native data adapter and manifest audit pass;
3. native-512 backward + AdamW succeeds on the largest real object;
4. checkpoint/resume is verified;
5. validation and event/derivative audit are complete;
6. the run contract pins source, data, config, environment and seeds.

Until then it is a development branch, not a training-ready release.


## Implementation update — 2026-11-12

The first M80 code slice is now present in this Spin_test2 branch:

- `spin_pino/pino_backbone.py`: additive `forward_with_map()` returns the legacy pooled P-CRAFT latent plus the final spatial decoder map. It adds no parameters and the legacy `forward()` delegates to the pooled result.
- `spin_pino/native_pcraft.py`: P-CRAFT map/global features are sampled/fused at every M72 native interval; the coarse 32^3/64^3/128^3 context grid is exact RLE volume aggregation when it divides the native 512 lattice. Native intervals remain intact.
- `spin_pino/native_train.py`: M72 formal manifest/audit reader, geometry/label SHA checks, derivative-valid masking, largest-object GPU preflight, AdamW/checkpoint round-trip, group-balanced training and exact resume contract.
- Tests: `tests/test_native_pcraft.py`, `tests/test_native_train.py`.
- CI: `.github/workflows/m80-pcraft-native.yml`.

The model keeps the M72 bounded odd moment response and derives its Jacobian from the same differentiable response. The M72 fixed/variable moment split, raw ten-moment order and native quadric coefficient order are retained.

### Current gate status

- Source integration: implemented in this branch.
- CPU mathematical and M72 adapter tests: initial CI passed; the new end-to-end CPU preflight test exposed two small trainer bugs (cache-hit return shape and missing return value). Both fixes are committed; the latest CI run is pending.
- RTX 5090D full backward/AdamW/memory acceptance: not yet run.
- M72 formal 256-query corpus: not yet confirmed complete; long training defaults to 256 queries per qualified record and blocks smaller banks.
- Spin-It Faster Newton bridge and adaptive exact correction: not yet implemented.
- No multi-day training has been started.

Do not label this branch large-training-ready until the current CI rerun passes, then the user's local 5090D preflight passes on the largest real training object and the formal 256-query audit is ready.


## M80.2 implementation update — 2026-11-12

An optional native-interval GNO local integral branch has now been implemented in `spin_pino/native_pcraft.py`.

It builds a self-inclusive kNN graph over the original M72 interval centers, uses relative center coordinates, interval extent ratios and mass ratios as edge descriptors, applies a positive learned kernel weighted by native interval mass, and performs a residual local feature update before the existing mass-weighted global reduction.

The GNO branch is controlled by `gno_enabled`; the original `large`/`xl`/`xxl` configurations remain the non-GNO controls. New `large_gno` and `xl_gno` presets enable the branch. Geometry-only graph caches are bounded, stored as int32, and keyed by geometry hash, neighbor count and device.

CPU CI now covers graph integrity, GNO moment complement/scale invariance, AD Jacobian finite-difference agreement, second-order backward, cache reuse and base-vs-GNO preset separation. This is code/CPU evidence only. GPU resource and quality comparisons remain mandatory before promoting GNO.

See `M80.2_GNO_ABLATION.md` and the paired commands in `M80_LOCAL_RUN_GUIDE_CN.md`.


## M80.4 implementation update — 2026-11-13

The refactor now includes:
- shared-weight P-CRAFT execution at exact conservative 32/64/128 context grids;
- a geometry-conditioned global scale gate and per-native-interval local scale gate;
- optional 6-connected RLE component filtering for GNO kNN, preventing edges between disconnected nearby layers;
- per-query Moment/Jacobian RMSE P50/P95/P99/max validation summaries;
- best-checkpoint validation and an explicit one-time held-out test option;
- optional event/derivative audit arrays accepted by the M72 reader, while the current loss still uses only required moment/Jacobian arrays.

New multiscale presets are opt-in. Existing single-scale presets and the M80.1 baseline remain available. The single-scale GNO preset preserves the earlier Euclidean-kNN behavior; new multiscale GNO presets enable topology filtering.

CPU evidence: M80.4 multi-resolution, topology, validation-tail and optional-label contracts passed in CI:
https://github.com/volcano-code/Spin_test2/actions/runs/43275986716

GPU boundary: the supplied four single-scale Large factorial preflights passed. No M80.4 multiscale GPU preflight or trained-quality promotion has yet been reported.
