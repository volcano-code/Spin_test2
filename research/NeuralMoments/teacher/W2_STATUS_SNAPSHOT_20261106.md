# W2 status snapshot — 2026-11-06

Later on 2026-11-06 a fresh Spin_test2 evaluation preserved the full attachment; the new result is recorded under its own run identity. [M23 evidence](M23_HEART_POINT000_FRESH.md)

**Repository:** `volcano-code/Spin_test`  
**W2 branch:** `research/neuralmoments-w2-teacher-20261019`  
**Snapshot parent HEAD:** `ab4388af59da20da6826b0d2e3ebad036aee508e`  
**W0 dependency snapshot:** `5bab2eb2616df306906a6887a273f68acfa49320`  
**W3 dependency snapshot:** `eb61baa4722d0cfb1000f10cbda19aa222fe7977`

## Executive status

W2 has a fresh, real Frozen Heart Cherchi point evaluation with ten physical moments independently matching the predeclared exact-rational reference. M22-P1 audited the original run archive and M22-P2 independently replayed the matching historical point-000 boundary, reintegrated its ten exact moments, and checked the embedded surface with the pinned W1 exact triangle-intersection checker.

This remains diagnostic-only. The M21 full same-call material digest and evaluation-map identity do not match the historical point-000 material; the exact boundary and all ten exact moments do match. No full canonical same-call object was recovered. `J_D=null`, `split=excluded`, and training samples remain zero.

## Completed with evidence

### Heart

- One fresh local native Cherchi call: `heart-pilot9-flex-20261105T083338329274558-3159689`.
- Independent ten-moment comparison: 10/10 within per-component `1e-10`; maximum reported binary64 absolute error 0.
- Original M21 archive: SHA-256 `259b9f288b629415bafa42e4e09d6503e4f474e0e2253e6cc9eb1cd775a64dfd`; all 12 original checksum entries and cross-file identities passed.
- Historical H8f point-000 boundary: 8,455 vertices, 16,910 oriented faces; exact boundary digest matches M21.
- Historical point-000 exact-rational reintegration: all ten exact fractions equal the M21 saved reference; max exact-vs-reported-binary64 difference approximately `2.4071727477604756e-17`.
- Exact surface audit: 161,702 candidate pairs checked; 49,727 non-adjacent pairs; no illegal self-intersection; closed, consistently oriented, one connected component.
- The surface audit proves the embedded-boundary property of this mesh only. It does not prove Boolean region set equality or derivative continuity.

### Noodles V7

- 94/94 source parts reconstructed/reverified; 188 model/SI OBJ hashes match the frozen manifest.
- 940 exact rational ten-moment component calculations exist for the individual source parts.
- The 96-operation CSG DAG is reconstructed and source-bound.
- The six exact rational CSG output packets (`design`, `envelope`, `fixed`, `fixed-inside-envelope`, `fixed-outside-envelope`, `initial`) remain missing.
- The historical receipt records 96 old CGAL operations; new native CGAL calls in the M20 recovery milestone were zero. Individual source-part moments cannot be summed to claim the overlapping fixed union F.

## Current blockers

1. **Heart material provenance:** M21 attachment descriptor hash `b136cbcc09740915cd70615d73fab3ab2aa0b680cdb823cfed12375bed6c20e8` differs from historical point-000 material hash `6be4faf58ef2fc31b6a1e6d67e601417150d0b43dbd37104b61565a2ca6708c1`; evaluation-map identity also differs. Boundary and moments match, but full same-call serialized object is absent.
2. **Native reproducibility:** the M21 archive contains module identity hashes but not the three compiled native shared objects or external module manifest; the external budget file is also absent (the ten tolerances are embedded in the spec).
3. **Boolean semantics:** no independent proof yet that the Heart output region equals the intended Boolean set, despite the point-000 boundary being closed and self-intersection-free.
4. **Noodles F/D/A:** six exact rational CSG packets and a validated separate F/D/A partition remain missing; no fresh Noodles Cherchi teacher call.
5. **Derivatives:** W2 labels `derivative_status=UNKNOWN`, `J_D=null`. No trusted continuous smooth-neighborhood certificate, no admitted 10x10 Jacobian, and no Newton acceptance.
6. **Training:** `train`, `val`, and `test` remain empty; training sample count 0.

## Cross-window dependency snapshot

- W0 latest commit pins W2 M21/M22 receipts and provides a structural/read-only consumer gate. Its own code explicitly says `raw_diagnostic_archive_verified_by_W0=false`, `W0_full_scientific_admission_certified=false`, `Newton_accepted=false`, and no training release. This is provenance/contract integration, not scientific admission.
- W3 latest M21.3 decomposes/cross-checks LPI motion for three constructions across six saved packets. Its report records 9 new synthetic tests plus 24 prior synthetic regressions passing, but 0/18 new real LPI records because the real frozen inputs/source-motion evidence were not present in that run. Continuous neighborhood and Newton remain unaccepted.

## Initial-goal scorecard

| Initial W2 objective | Status |
|---|---|
| Real Cherchi teacher execution | One fresh Heart diagnostic point; no fresh Noodles teacher |
| Independent ten physical moments | Heart point-000 passes; Noodles Boolean F/D/A not admitted |
| Fixed/variable/total moment contract | Heart values bound and total relation replayed for point-000; dual-model contract not complete |
| Full same-call material provenance | Partial: descriptor and matching boundary; canonical object/evaluation identity not matched |
| Boolean solid/set semantics | Not certified |
| Full 10x10 Jacobian | Not admitted; `J_D=null` |
| Smooth neighborhood / derivative validity | Not certified by W0/W3 |
| Newton convergence | Not accepted |
| Dual-model Heart + Noodles teacher set | Not achieved |
| Training-ready train/val/test data | 0 samples |

## Next work

### M22-P3 — close Heart solid/material semantics

- Preserve the current point-000 result as excluded; do not rewrite historical identities.
- Determine whether the full M21 same-call material payload can be recovered from the native producer's material cache/output path. If it cannot, explicitly retain `BOUNDARY_ONLY` provenance and create a new uniquely identified capture only if a fresh run is later approved.
- Independently verify the intended Boolean region using the authenticated Heart operands and exact boundary, not just surface self-intersection.
- Keep the original H8f workflow failure unchanged.

### M23 — Noodles V7 exact F/D/A

- Recover the six historical rational packets by exact SHA, or rebuild all 96 CSG operations with the pinned backend under a new evidence identity.
- Validate closure, orientation, embedding, set partition and exact ten-moment accounting. Do not sum overlapping fixed components.
- Then run a fresh Noodles Cherchi teacher on the authenticated F/D/A request.

### M24 — derivative and teacher admission

- Coordinate with W0/W3 for independently authenticated same-side/four-side neighborhood probes and full 10x10 Jacobian checks.
- Require Newton convergence evidence before promoting any sample.
- Until then keep all labels excluded and `J_D=null`.

## Execution location

- Cloud: parsing, hash verification, exact rational re-integration tools, geometry audit tooling, test harnesses, manifest/report updates.
- Local qualified native host: real Cherchi calls, 96-op exact CSG if resource/build prerequisites are met, source-motion capture, and final native Jacobian/Newton evidence.
- The 4 GiB cloud cgroup is not qualified for the M16 24 GiB native protocol. The successful 9 GiB M21 diagnostic profile is not a blanket memory qualification for the larger Noodles CSG workload.

**Overall verdict:** Heart has passed one meaningful physical-moment and point-000 embedded-boundary diagnostic. W2 has not yet delivered the initial dual-model, derivative-certified, Newton-validated, training-ready Cherchi teacher system.