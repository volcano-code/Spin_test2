# Spin-PINO M1 analytic training report

Date: 2026-11-06
Branch: research/neuralmoments-w6-pcraft-20261019
Training workflow: [42188044841](https://github.com/volcano-code/Spin_test/actions/runs/42188044841)
Workflow commit: 00c17bdfc4c5dff4b0862b428a1c08246c5dc924
Artifact: `spin-pino-m1-analytic-three-seed.zip` (GitHub artifact ID 13488726149)
Artifact SHA-256: `3be4a0ffb41345bcef6c5f682d52e5c108bd5a77964b14754c6cbf3a790decbe`

## Dataset and training protocol

- Source: existing `analytic_cut.py`, `ANALYTIC_CONTINUUM_P0_NOT_W2_CHERCHI`.
- Fit families: axis-aligned boxes and spheres.
- Validation: same two families, but disjoint geometry IDs.
- Held-out test: ellipsoid family, entirely absent from fit/validation.
- Per fit family: 16 geometries × 2 vertical-plane queries.
- Fit: 52 samples; validation: 12 samples; held-out test: 16 samples.
- Geometry input: four channels at 8^3.
- Each model: 14,440 trainable parameters; Adam, 120 steps, batch size 8.
- Three seeds: 131, 137, 139.
- Supervised Jacobians are exact analytic derivatives for strictly interior vertical-plane cuts only. No W2/Cherchi labels were used.

## Three-seed results

| Seed | Selected step | Fit r MAE | Validation r MAE | Held-out ellipsoid r MAE | Held-out ellipsoid J MAE |
|---:|---:|---:|---:|---:|---:|
| 131 | 110 | 0.04856 | 0.06154 | 0.02991 | 0.03600 |
| 137 | 120 | 0.05035 | 0.06262 | 0.04996 | 0.03448 |
| 139 | 110 | 0.05041 | 0.05108 | 0.04221 | 0.03333 |
| Mean | — | 0.04977 | 0.05841 | 0.04069 | 0.03461 |

Mean held-out ellipsoid relative L2: r = **0.4250**, J = **1.5802**. Held-out negative-mass fraction was 0 for all seeds.

## Interpretation

- The PINO-centered model and analytic training loop are operational and reproducible on cloud CPU.
- The entire ellipsoid family was held out, so this is a useful first geometry-family generalization check.
- The current errors are not yet good enough to claim a useful Spin surrogate: the held-out Jacobian relative L2 is particularly weak (>1), and no fair U-FNO/DeepONet/GINO baseline comparison has yet been run.
- The 8^3 input and 4^3 learned quadrature nodes are deliberately small smoke settings, not a final resolution choice.
- The positive-mass bound and analytic potential identities passed their unit tests; this does not establish accuracy of the learned integral.

## Required M2 follow-up

1. Run a controlled smoothing/quad-node/grid-resolution sweep (smoothing 0.01/0.02/0.04; 4^3/6^3 nodes; 8^3/16^3 input) on the same fixed geometry splits.
2. Compare Spin-PINO against the frozen W6 M13 U-FNO model under identical data, steps, seeds and parameter reporting.
3. Add per-component error analysis, especially mass and z/zz terms, and Jacobian directional error.
4. Keep the held-out ellipsoid family untouched during hyperparameter selection; rotate family-held-out tests in a second evaluation.
5. Only promote a configuration if it improves held-out error without violating the moment-physics identities.

## Scope limits

This dataset covers axis-aligned box/sphere/ellipsoid solids cut by strictly interior vertical planes. It does not validate arbitrary 10D quadrics, general CAD Boolean topology, Cherchi teacher transfer, native Newton use, 512^3 performance or real Heart/Teapot/Noodles acceleration.
