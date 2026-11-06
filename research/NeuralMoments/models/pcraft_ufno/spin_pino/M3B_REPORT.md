# Spin-PINO M3b Gauss-Legendre quadrature ablation

Date: 2026-11-06
Workflow: [42193467406](https://github.com/volcano-code/Spin_test/actions/runs/42193467406)
Implementation commit: ffbf18ecdbb6cd22ae80fa5429b3106e1cfaa79c
Artifact ID: 13490493357 (`spin-pino-m3b-gauss-tournament.zip`)

## Change under test

Transferred only the positive tensor-product Gauss-Legendre node/weight construction from W6 M7 into the P-CRAFT PINO head as a selectable `quadrature_rule='gauss'`. The P-CRAFT UFNO trunk, PINO moment potential, physics loss and density/occupancy support mask remain unchanged. The `grid_sample` coordinate port for [B,C,X,Y,Z] input was corrected to use (z,y,x) at the sampling interface.

## Same-data results

| Model | r relative L2 | J relative L2 | Combined score |
|---|---:|---:|---:|
| Spin-PINO midpoint 6-node | **0.23204** | **0.89120** | **0.56162** |
| Spin-PINO Gauss 6-node | 0.47271 | 1.20223 | 0.83747 |
| W6 M7 Realizable U-FNO | 0.43717 | **0.49310** | **0.46513** |
| W5 voxel DeepONet | 0.75109 | 0.69941 | 0.72525 |

All use seeds 131/137/139, 120 steps and identical analytic geometry splits. Parameter counts: both PINO variants 21,888; M7 13,448; W5 13,939.

## Decision

Do not promote Gauss-Legendre nodes into the PINO candidate. In this voxelized occupancy-mask and learned-gate setting, midpoint quadrature is materially better for both r and J. Retain Gauss as a negative ablation. This demonstrates that M7's positive-measure idea transfers, but its particular node distribution does not transfer automatically.

## Next research step

Investigate PINO training conditioning and gradient conflict, borrowing PI-CViT's PINO training diagnostics and TensorPILS's preconditioned residual principle. Implement only mathematically valid Spin moment residual conditioning; do not import their unrelated PDE residual equations or replace the P-CRAFT PINO trunk.

## Limits

Analytic vertical-plane P0 only; no Cherchi labels, arbitrary quadric certification, Newton admission or real Spin end-to-end result.
