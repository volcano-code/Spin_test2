# Spin-PINO M2b support-masked sweep report

Date: 2026-11-06
Branch: research/neuralmoments-w6-pcraft-20261019
Implementation commit: 0a728ce93f51e6d79ab041332b8c98c7b518be9e
Workflow: [42191687667](https://github.com/volcano-code/Spin_test/actions/runs/42191687667)
Artifact ID: 13489614384 (`spin-pino-m2b-support-masked.zip`)

## Why M2b was required

The original positive quadrature head did not explicitly mask learned atoms by the input density/occupancy. This allowed learned discrete mass in sampled empty space. M2b changed the atom weights to

`w_k = (8/N^3) * rho(x_k) * occupancy(x_k) * sigmoid(g_theta(D)_k)`.

This enforces sampled discrete support and density. It is not a continuum or CAD certificate.

## Same-split, three-seed results

| Configuration | Parameters | Held-out r MAE | Held-out J MAE | r relative L2 | J relative L2 | CPU train seconds | Combined score |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8^3 / 4^3 / eps 0.04 | 14,440 | 0.02588 | 0.02247 | 0.42333 | 1.20179 | 1.72 | 0.81256 |
| 8^3 / 6^3 / eps 0.02 | 21,888 | 0.01693 | 0.01891 | **0.25611** | **0.89719** | 1.75 | **0.57665** |
| 16^3 / 4^3 / eps 0.02 | 53,088 | 0.02010 | 0.04052 | 0.29955 | 3.19076 | 9.31 | 1.74516 |

Combined score is the mean of held-out r and J relative L2. Lower is better. All rows use identical geometry IDs/splits and seeds 131, 137, 139; train families boxes+spheres, held-out test family ellipsoid; 120 steps.

## Comparison with pre-mask M2

| Configuration | Pre-mask r rel L2 | Support-masked r rel L2 | Pre-mask J rel L2 | Support-masked J rel L2 |
|---|---:|---:|---:|---:|
| 8^3 / 4^3 / eps 0.04 | 0.42496 | 0.42333 | 1.58025 | 1.20179 |
| 8^3 / 6^3 / eps 0.02 | 0.51117 | **0.25611** | 0.96144 | **0.89719** |
| 16^3 / 4^3 / eps 0.02 | 0.43395 | 0.29955 | 2.49809 | 3.19076 |

Support masking materially improved the selected 6-node configuration's moment generalization and modestly improved its Jacobian relative error. The 16^3 configuration remains poor for J and much slower; do not promote it.

## Decision

Promote `grid_size=8`, `points_per_axis=6`, `smoothing=0.02` as the current Spin-PINO research configuration for M3 comparison, not as a production/Newton-ready model.

## Limits

- Exact labels are analytic continuum vertical-plane cuts only, not W2/Cherchi labels.
- Test family is ellipsoid; no arbitrary quadratic cut, CAD Boolean, or native Spin validation.
- Jacobian relative L2 0.897 is still too large for Newton admission.
- Model has 21,888 parameters; compare capacity and runtime against baselines explicitly.
- The same workflow's PINO unit-test job initially had one faulty density-scaling test assumption (the learned gate also depends on density); that test was corrected separately. The M2b training job itself passed.
