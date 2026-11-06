# Spin-PINO M0 validation report

Date: 2026-11-06
Branch: research/neuralmoments-w6-pcraft-20261019
Implementation commit: d1ee418a006e7dc41f5260dea19aa02b9f74f1ef
CI-fix commit: 3c366be7b0b3476d8f1db7a0e6ab3dbb2ee23ec1

## CI result

- GitHub Actions run: [42186600269](https://github.com/volcano-code/Spin_test/actions/runs/42186600269)
- Overall workflow: success (3/3 jobs).
- Spin-PINO CPU job: **9 passed in 1.44 s**.
- Existing NeuralOperator/QuadrANN CPU smoke: **8 passed in 2.49 s**.
- Algoim C++ job: official pinned source compiled; CTest **1/1 passed**.
- Algoim unit-sphere ten-moment maximum absolute error: **7.5044859215722681e-11**.

## What the Spin-PINO tests establish

1. P-CRAFT UFNO3d-style factorized-Fourier U-shaped encoder executes on CPU and returns a geometry latent.
2. The Spin task head returns ten raw moments and a 10x10 coefficient Jacobian.
3. Positive projective coefficient scaling leaves predicted moments invariant.
4. The smooth convex moment potential satisfies the implemented complement identity within test tolerance.
5. The Jacobian is symmetric and positive-semidefinite in monomial coefficient coordinates within floating-point tolerance.
6. The radial identity J_c c = 0 holds within tolerance.
7. SpinPINOLoss backpropagates through the PINO geometry trunk and moment head.
8. Diagnostic/unadmitted Jacobian labels are refused by default.

## Fixes and limitations

- First CI attempt failed because pytest was requested from the PyTorch CPU wheel index. The workflow was corrected to install torch from the PyTorch index and pytest from PyPI; this was a CI dependency configuration error, not a model failure.
- A PyTorch warning reported NumPy was not installed in the minimal Spin-PINO job. The tests do not require NumPy; add NumPy if future data-generation tests need it.
- The source P-CRAFT repository has no root LICENSE file in the audited tree. Current source reuse is recorded as internal project reuse pending explicit ownership/license clearance; do not redistribute externally until resolved.
- No Spin-PINO training run has been performed yet. The tests validate architecture and mathematical identities only.
- No W2/Cherchi labels are used; no J_D is admitted to Newton; no real Heart/Teapot/Noodles speedup is claimed.

## Next milestone

M1 (Nov 9–12): create deterministic geometry-family datasets with analytic plane/box labels and Algoim reference labels, record source hashes, and implement a reproducible PINO training script with data/physics loss traces. Keep the current W6 M13 U-FNO and W5 DeepONet as frozen comparison baselines.
