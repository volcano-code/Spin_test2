# M80 baseline audit — Spin_test2

Date: 2026-11-12  
Target repository: `volcano-code/Spin_test2`  
Target branch: `research/m80-pnmo-large-train-20261112`  
Base: `research/neuralmoments-w6-pcraft-20261019`

## Repository correction

All M80 development belongs in **Spin_test2**. The similarly named `Spin_test` repository is not the target project. It may be consulted only as a source of later M69–M72 Python work, and no files, manifests, checkpoints, or data are to be copied wholesale without compatibility review.

## What is actually present in this Spin_test2 base

- The W6 `spin_pino/` package contains a P-CRAFT-derived factorized Fourier U-shaped encoder (`pino_backbone.py`), Spin-specific moment model (`spin_pino.py`), analytic P0 dataset builder (`dataset.py`), and CPU-first analytic trainer (`train.py`).
- The P-CRAFT encoder currently consumes a dense four-channel tensor `[B,4,X,Y,Z]`; its `grid_size` path uses adaptive pooling/interpolation to a configured cubic grid. The historical training examples used very small grids (8/16), not native 512 interval inputs.
- The SpinPINO output head currently predicts positive weights at a fixed tensor-product quadrature rule (default four nodes per axis), then derives moments and coefficient Jacobian from its own moment expression.
- The training dataset is explicitly analytic P0 (boxes/spheres/ellipsoids cut by strictly interior vertical planes). It is **not** the formal Cherchi/native CAD teacher dataset and is not general-quadric/Newton admission.
- M8 Transolver++ and M9/M10 experiments contain useful ablations, including negative results. They must remain optional, not be silently promoted.
- The branch contains Spin-It C++ code for native moments, quadric coefficients, solvers and voxel methods. That native code is the authoritative project context; Python surrogate work must preserve its coefficient, moment, coordinate and task contracts.

## What is not yet present / not proven

- No verified end-to-end adapter from the M69/M71 native interval manifest and labels into this P-CRAFT SpinPINO input/output contract.
- No verified native-512 P-CRAFT forward/backward/memory acceptance on the RTX 5090D.
- No demonstrated training on admitted native Cherchi moment/Jacobian labels.
- No validated neural-query-to-Spin-It-Faster Newton bridge with event/derivative gating and exact fallback.
- No measured end-to-end speedup or improved final material mass on the same native Spin task.
- No formal TRELLIS-generated geometry admitted to the training corpus.

## Critical mathematical / representation mismatch

The P-CRAFT SpinPINO head currently learns a positive discrete quadrature measure at a fixed set of points. The M69/M71 query system predicts a variable-domain ten-moment response for a native quadric/material query, with its own coefficient basis, raw-moment ordering, fixed/variable material split, labels, derivative-validity states and event rules.

These are not interchangeable tensors. A migration adapter must explicitly reconcile:
1. coefficient ordering and cross-term factors;
2. raw ten-moment ordering and coordinate frame;
3. fixed material moments versus variable-domain moments;
4. material mask / occupied interval semantics;
5. query normalization and Jacobian chain rule;
6. derivative validity and non-smooth event withholding;
7. exact teacher source and audit/provenance metadata.

Do not feed M69 labels into the existing four-channel/fixed-node head by reshaping or truncating. That would be an invalid supervision contract.

## M80 decision

Build a **P-CRAFT-dominant hybrid** in Spin_test2, but preserve the M71 Native States model as a matched baseline. The P-CRAFT encoder must not be reduced to a decorative side branch. Native interval features must enter the main representation without lossy conversion to a small dense grid. Local geometry / physical-state modules are optional ablations.

The first training-ready release is blocked until the data adapter, mathematical parity tests, full native-512 backward + AdamW memory test, and exact-solver safety bridge all pass.

## Preservation rules

- Do not overwrite W6 baseline files or old checkpoints.
- Do not import the Spin_test M69–M72 package wholesale.
- Do not change derivative audit thresholds to force a pass.
- Do not call analytic P0 labels "Cherchi" or "native CAD" labels.
- Do not call a Newton equality core a general constrained solver.
- Do not start a multi-day training run before the M80 readiness gate is satisfied.


## M80 implementation delta — 2026-11-12

The earlier "not yet present" items have been partially addressed in this branch:
- A versioned M72 native RLE-to-P-CRAFT adapter now exists in `spin_pino/native_pcraft.py`.
- The P-CRAFT encoder now exposes its final spatial feature map without changing its parameter/state-dict contract.
- A strict M72 manifest/audit and label loader plus GPU preflight/training entry point now exists in `spin_pino/native_train.py`.

These changes do **not** yet close the native 512 GPU acceptance, formal 256-query dataset, derivative audit, or Spin-It Faster Newton bridge gates. The earlier caveats about not claiming end-to-end speedup remain in force.
