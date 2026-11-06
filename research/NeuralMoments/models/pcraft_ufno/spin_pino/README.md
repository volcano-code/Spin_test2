# Spin-PINO v1 — PINO-first model

Primary model: P-CRAFT UFNO3d / F-FNO factorized Fourier backbone, adapted to Spin's ten raw moments.

- `pino_backbone.py`: source-pinned P-CRAFT UFNO3d-style factorized spectral U-shaped encoder.
- `spin_pino.py`: geometry-conditioned positive quadrature weights, smooth convex moment potential, ten moments and analytic coefficient Jacobian.
- `SpinPINOLoss`: data term plus moment-specific physics identities. The original elasticity equilibrium/constitutive PINO losses are not reused because they do not describe Spin's moment integral.
- `tests/test_spin_pino.py`: CPU smoke and mathematical identity tests.

Borrowed GINO, Transolver, QuadrANN, UQNO, NQR, fVDB and DensiCrafter components are optional plug-ins or baselines. None replaces the PINO trunk.

Source attribution is recorded in [the core source audit](../PINO_CORE_SOURCE_AUDIT.md). The historical training and milestone reports below describe the original Spin_test branch.

The Spin_test2 import and current analytic replay are documented in [SPIN_TEST2_REPLAY.md](SPIN_TEST2_REPLAY.md).

## M0 CPU validation

GitHub Actions run [42186600269](https://github.com/volcano-code/Spin_test/actions/runs/42186600269) passed: Spin-PINO 9/9 tests in 1.44 s; the existing GINO/QuadrANN smoke 8/8; Algoim C++ 1/1 with max absolute error 7.50e-11. This validates the initial architecture and identities only; no Spin/Cherchi training or Newton admission is claimed. See `VALIDATION_REPORT.md`.

## M1 analytic P0 data and training

The dataset builder uses the existing exact continuum vertical-plane references for boxes, spheres and ellipsoids. Fit/validation are split by geometry identity (boxes+spheres); the entire ellipsoid family is held out for testing. Labels are explicitly tagged `ANALYTIC_CONTINUUM_P0_NOT_W2_CHERCHI`; the analytic Jacobian is only valid for strictly interior vertical-plane cuts.

Small CPU smoke:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q \
  research/NeuralMoments/models/pcraft_ufno/spin_pino/tests/test_spin_pino.py \
  research/NeuralMoments/models/pcraft_ufno/spin_pino/tests/test_dataset.py
```

Three-seed training (CPU prototype):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.train \
  --seeds 131,137,139 --train-geometries-per-family 16 \
  --test-ellipsoid-geometries 8 --queries-per-geometry 2 \
  --steps 120 --grid-size 8 --device cpu \
  --output /tmp/spin_pino_m1_three_seed.json --save-checkpoints
```

These are synthetic analytic labels, not Cherchi supervision. The held-out family tests geometry generalization only within this restricted P0 vertical-plane task; no arbitrary quadric or native Newton claim is implied.

## M1 three-seed analytic training result

GitHub Actions run [42188044841](https://github.com/volcano-code/Spin_test/actions/runs/42188044841) passed all jobs. The 3-seed CPU run used 52 fit / 12 validation / 16 held-out ellipsoid samples, 8^3 geometry inputs and 120 steps. Mean held-out ellipsoid r MAE was 0.04069 and J MAE 0.03461; relative L2 errors were 0.4250 and 1.5802. This is a functional but preliminary baseline, not yet good enough for Newton use. Full metrics and caveats are in `M1_TRAINING_REPORT.md`; checkpoint/report artifact ID 13488726149.

## M2 controlled sweep

`sweep.py` runs three matched configurations on the same geometry-family splits and three seeds:

- baseline: 8^3 input, 4^3 positive nodes, smoothing 0.04;
- quadrature refinement: 8^3 input, 6^3 positive nodes, smoothing 0.02;
- resolution refinement: 16^3 input, 4^3 positive nodes, smoothing 0.02.

The selection metric is the mean of held-out ellipsoid r and J relative L2, on the previously reused model-development benchmark. Run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.sweep \
  --seeds 131,137,139 --steps 120 --batch-size 8 --device cpu \
  --output /tmp/spin_pino_m2_sweep.json
```

This sweep is designed to identify whether low M1 accuracy is caused by coarse geometry input, insufficient positive quadrature nodes or potential smoothing. It does not by itself prove performance on general quadrics or Cherchi geometry.

## M2b: support-masked quadrature measure

The initial M2 sweep showed a trade-off: increasing from 4^3 to 6^3 nodes reduced held-out Jacobian relative L2 from 1.580 to 0.961, but worsened moment relative L2 from 0.425 to 0.511. Increasing the voxel input from 8^3 to 16^3 did not improve the combined score.

Code review identified a missing physical constraint in the first Spin-PINO head: its positive learned quadrature weights were not explicitly masked by the input density/occupancy at the quadrature nodes. M2b fixes this. Each node weight is now

`w_k = (8/N^3) * rho(x_k) * occupancy(x_k) * sigmoid(g_theta(D)_k)`.

This prevents the network from creating discrete material atoms in sampled empty space. It is a discrete sampled-support constraint, not a continuum/CAD certificate. The M2b three-seed sweep reruns the same three configurations and is tracked separately from the pre-mask M2 results.

## M2b support-masked result

The support-masked 8^3 input / 6^3 quadrature / smoothing 0.02 configuration is the current research setting. Mean held-out ellipsoid r relative L2 = 0.2561 and J relative L2 = 0.8972 across seeds 131/137/139. This improves on the pre-mask 6-node run, but is not Newton-ready. See `M2B_REPORT.md`.

## M3 same-data tournament

`tournament.py` compares two Spin-PINO configurations with W6 M7 Realizable U-FNO and W5 voxel DeepONet using identical analytic dataset splits, labels, seeds and training steps. It reports capacity differences rather than hiding them.


## M3 same-data tournament

Run [42192701882](https://github.com/volcano-code/Spin_test/actions/runs/42192701882) passed. On the same analytic split, support-masked Spin-PINO 6-node achieved held-out r relative L2 0.2561 / J relative L2 0.8972; W6 M7 Realizable U-FNO achieved 0.4372 / 0.4931; W5 voxel DeepONet achieved 0.7511 / 0.6994. M7 is the strongest combined/J baseline, while PINO remains the primary backbone and has the best r accuracy. See `M3_TOURNAMENT_REPORT.md`.


## M3b Gauss-Legendre quadrature borrowing

`SpinPINO(quadrature_rule='gauss')` uses positive tensor-product Gauss-Legendre weights from the W6 M7 design while retaining the P-CRAFT PINO trunk and SpinPINOLoss. `midpoint` remains available for ablation. The support mask is applied at the correct XYZ voxel coordinates. Tests verify positive weights, total volume 8 and asymmetric X-axis support sampling. The next tournament includes both midpoint and Gauss PINO variants.


## M3b Gauss rule ablation result

The Gauss-Legendre variant passed implementation tests but did not improve held-out accuracy: r/J relative L2 0.4727/1.2022 vs midpoint PINO 0.2320/0.8912. It remains a negative ablation and is not promoted. See `M3B_REPORT.md`.


## M4 loss-conditioning ablation

`loss_conditioning.py` provides a fit-split-only diagonal residual preconditioner and deterministic PCGrad-style gradient projection. `conditioning_sweep.py` compares baseline, preconditioned loss, and preconditioned loss + PCGrad using the selected support-masked PINO midpoint 6-node configuration. This adapts PI-CViT/TensorPILS training ideas while preserving P-CRAFT PINO as the backbone and Spin-specific physics losses.


## M4 loss-conditioning result

The M4 three-seed sweep did not improve over the baseline: r/J relative L2 0.2320/0.8912 baseline, 0.3101/0.8963 diagonal preconditioning, 0.2898/0.8907 preconditioning+PCGrad. PCGrad increased CPU time. Keep these as optional ablations; do not change default PINO loss. Next investigate a PINO-preserving M7 logistic realizable-atom head transplant for Jacobian accuracy. See `M4_LOSS_CONDITIONING_REPORT.md`.


## M5 M7 logistic-head transplant

A new optional `moment_parameterization='logistic_atom'` borrows W6 M7's projective sigmoid-atom moment/Jacobian formula while keeping the P-CRAFT PINO trunk, geometry-conditioned positive weights and density/occupancy support mask. The original `smooth_hinge` potential remains the default. Logistic mode uses tangent-projected q-space Jacobian symmetry/PSD constraints and does not claim to be a scalar convex potential. Tests compare its analytic Jacobian to finite differences.


## M5 result — logistic-atom head

The same-data three-seed tournament shows the PINO logistic-atom head achieves held-out r/J relative L2 0.2060/0.3147, beating the smooth-hinge PINO 0.2320/0.8912 and W6 M7 0.4372/0.4931 on the analytic vertical-plane P0 task. The P-CRAFT PINO trunk remains primary. Logistic mode is not a scalar convex potential and is not Newton/Cherchi certified. See `M5_LOGISTIC_HEAD_REPORT.md`.


## M6 rotation covariance

`equivariance.py` implements exact rotation transforms for native coefficients, ten raw moments and Jacobians, plus voxel resampling and the 24 proper cube rotations. `augment_with_cube_rotations` adds exact transformed analytic labels to the fit split only. `rotation_sweep.py` compares logistic PINO with and without rotation augmentation and reports both unrotated and rotated held-out ellipsoid metrics.

## M7 arbitrary SO(3) orientation augmentation

`equivariance.py` now includes seeded Haar-uniform SO(3) rotations generated from unit quaternions. Coefficients, ten raw moments and Jacobians are transformed exactly with the native-coordinate maps; voxel channels are trilinearly resampled on the fixed cube. This is deliberately distinguished from the 24 exact cube-preserving signed-permutation rotations: arbitrary SO(3) introduces voxel interpolation error, which must be measured separately.

Use the paired Spin_test2 test panels for the augmentation comparison. Historical M6/M7 mixed-panel reports used the training arm to select the evaluation rotations.
