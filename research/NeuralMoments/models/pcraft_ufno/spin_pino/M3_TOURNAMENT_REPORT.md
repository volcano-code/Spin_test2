# Spin-PINO M3 same-data architecture tournament

Date: 2026-11-06
Workflow: [42192701882](https://github.com/volcano-code/Spin_test/actions/runs/42192701882)
Implementation commit: e8f30feb868820d5bcf8808b20794ded9769c5c9
Artifact ID: 13490785616 (`spin-pino-m3-same-data-tournament.zip`)

## Protocol

- Same deterministic analytic dataset and geometry-family splits for every model.
- Fit/validation families: boxes and spheres; held-out family: ellipsoids.
- Seeds 131, 137, 139; 120 Adam steps; batch size 8; same four-channel voxel D and native coefficient c.
- Labels: exact analytic continuum ten moments and Jacobians for strictly interior vertical-plane cuts only; no W2/Cherchi labels.
- Models: Spin-PINO 4-node parameter-matched, support-masked Spin-PINO 6-node M2b config, W6 M7 Realizable U-FNO, W5 voxel DeepONet.
- Parameter counts and training time reported; not forced equal.

## Three-seed held-out results

| Model | Parameters | r MAE | r relative L2 | J MAE | J relative L2 | CPU training seconds | Combined relative score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spin-PINO 4-node | 14,440 | 0.02588 | 0.42333 | 0.02247 | 1.20179 | 2.716 | 0.81256 |
| **Spin-PINO 6-node support-masked** | **21,888** | **0.01693** | **0.25612** | 0.01891 | 0.89719 | 2.916 | 0.57666 |
| W6 M7 Realizable U-FNO | 13,448 | 0.02550 | 0.43717 | **0.01134** | **0.49310** | **0.788** | **0.46513** |
| W5 voxel DeepONet | 13,939 | 0.05706 | 0.75109 | 0.02167 | 0.69941 | **0.713** | 0.72525 |

Combined score = mean(r relative L2, J relative L2). Lower is better.

## Interpretation

1. **PINO remains primary.** The selected support-masked Spin-PINO has the best moment r relative L2 (0.2561) of the four candidates. It is not replaced by M7 or W5.
2. **W6 M7 is the strongest derivative baseline.** It has the best combined score (0.4651), best J relative L2 (0.4931), lowest J MAE (0.01134), fewer parameters than Spin-PINO 6-node and about 3.7x lower CPU training time.
3. **Spin-PINO still has a derivative weakness.** Its J relative L2 (0.8972) is materially worse than M7's 0.4931. This is the most important architecture gap to address next.
4. **W5 DeepONet is not the best overall model on this voxel/analytic task.** It has a competitive J relative error but poor moment r generalization (0.7511 relative L2).
5. The M7 advantage suggests transplanting its positive Gauss-Legendre quadrature nodes/weights and realizable atomic-measure design into the P-CRAFT PINO head, while keeping the PINO U-shaped F-FNO trunk and PINO moment-physics loss unchanged.

## Decision and next step

- Keep P-CRAFT PINO as the only production backbone.
- Do not promote M7 as a replacement; treat it as the strongest current teacher/algorithmic comparator.
- Next M3b: add Gauss-Legendre quadrature nodes and tensor-product positive weights as a selectable PINO head rule; retain sampled density/occupancy masking. Compare midpoint vs Gauss within the same PINO trunk and same splits.
- After M3b, test TensorPILS-inspired preconditioned moment residual/loss balancing and PI-CViT-inspired gradient conflict/collocation strategy.

## Limits

Analytic vertical-plane P0 only; no arbitrary quadric, no W2/Cherchi training, no native Newton admission, no CAD Boolean certification, no real Heart/Teapot/Noodles timing.
