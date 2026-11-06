# Spin-PINO M5 logistic realizable-atom head tournament

Date: 2026-11-06
Workflow: [42195661322](https://github.com/volcano-code/Spin_test/actions/runs/42195661322)
Implementation commit: 0931f2442f50c4572912d27812ffde70c8e38fff
Artifact ID: 13491169764 (`spin-pino-m5-logistic-head-tournament.zip`)

## What changed

Borrowed only W6 M7's logistic realizable-atom moment and analytic native-coefficient Jacobian formula into the P-CRAFT PINO head. The P-CRAFT UFNO3d factorized-Fourier U-shaped trunk, geometry encoder, positive learned gate, density/occupancy support mask and PINO training loss remain in place. The smooth-hinge head remains available as an ablation.

Logistic mode uses tangent-projected q-space Jacobian symmetry/PSD residuals. It is not described as a scalar convex potential; the tangent PSD structure is not a continuum/CAD derivative certificate.

## Same-data three-seed results

| Model | Parameters | Held-out r MAE | r relative L2 | J MAE | J relative L2 | CPU train seconds | Combined score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Smooth PINO 4-node | 14,440 | 0.02436 | 0.41911 | 0.02163 | 1.15157 | 2.683 | 0.78534 |
| Smooth PINO midpoint 6-node | 21,888 | 0.01296 | 0.23204 | 0.01865 | 0.89120 | 2.702 | 0.56162 |
| Smooth PINO Gauss 6-node | 21,888 | 0.02736 | 0.47271 | 0.02252 | 1.20223 | 2.642 | 0.83747 |
| **PINO logistic-atom 6-node** | **21,888** | **0.01158** | **0.20602** | **0.00753** | **0.31466** | 2.587 | **0.26034** |
| W6 M7 Realizable U-FNO | 13,448 | 0.02550 | 0.43717 | 0.01134 | 0.49310 | 0.796 | 0.46513 |
| W5 voxel DeepONet | 13,939 | 0.05706 | 0.75109 | 0.02167 | 0.69941 | 0.764 | 0.72525 |

Combined score is mean(r relative L2, J relative L2). All rows use seeds 131/137/139, 120 steps, batch 8, identical analytic box+sphere fit/validation and held-out ellipsoid test families, same four-channel voxel input and analytic r/J labels.

## Interpretation

1. **M5 is the strongest result so far.** Logistic-atom PINO improves held-out r relative L2 from 0.2320 to 0.2060 and J relative L2 from 0.8912 to 0.3147 versus the smooth-hinge midpoint PINO.
2. The logistic PINO also beats the W6 M7 comparator on this same analytic test family (combined 0.2603 vs 0.4651), while retaining the P-CRAFT PINO trunk.
3. The result is consistent across all three seeds: r/J relative L2 are 0.1978/0.3242, 0.2231/0.3221 and 0.1972/0.2977.
4. The logistic head is slower to train than the compact M7 baseline (2.59 s vs 0.80 s CPU in this tiny setup) and has more parameters (21,888 vs 13,448). Runtime and memory must be measured separately before making a speed claim.

## Decision

Promote logistic-atom as the current best Spin-PINO research head candidate, while retaining smooth-hinge as a baseline. Keep P-CRAFT PINO as the primary backbone. Next test broader geometry families, orientation changes, near-tangent/event cases, inference latency and stability under increased resolution.

## Scope limits

All labels are analytic continuum vertical-plane cuts only, not Cherchi. The model is not Newton-admitted, not certified for arbitrary 10D quadrics or CAD Boolean events, and not yet validated on Heart/Teapot/Noodles. Do not infer real Spin acceleration from this synthetic benchmark.
