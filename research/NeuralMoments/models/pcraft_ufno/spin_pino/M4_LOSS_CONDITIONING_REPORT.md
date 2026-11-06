# Spin-PINO M4 loss-conditioning ablation

Date: 2026-11-06
Workflow: [42194697503](https://github.com/volcano-code/Spin_test/actions/runs/42194697503)
Artifact ID: 13491751590 (`spin-pino-m4-loss-conditioning.zip`)
Model: support-masked PINO midpoint 6-node, 8^3 grid, smoothing 0.02, 21,888 parameters.

## Same-data, three-seed results

| Training strategy | r relative L2 | J relative L2 | Combined score | CPU train seconds |
|---|---:|---:|---:|---:|
| Baseline M2b | **0.23204** | 0.89120 | **0.56162** | 2.53 |
| Diagonal fit-split residual preconditioner | 0.31015 | 0.89630 | 0.60322 | 2.51 |
| Diagonal preconditioner + PCGrad | 0.28979 | **0.89073** | 0.59026 | 4.56 |

Combined score is mean held-out r and J relative L2. All strategies use identical fit/validation/test geometry splits, seeds, labels, model, optimizer and steps. The diagonal scales are fit using the training split only.

## Gradient diagnostics

On the first logged PCGrad batch, raw data-vs-physics cosine was approximately -2.31e-6, data-vs-J cosine 0.3835, and physics-vs-J cosine 2.44e-5. The physics gradient norm was approximately 9.13e-17 because the current smooth-potential parameterization already satisfies its implemented moment identities to numerical precision. There was therefore almost no meaningful conflict for PCGrad to resolve.

## Decision

Do not promote the diagonal preconditioner or PCGrad to the default trainer for this task: neither beats the unconditioned baseline on the held-out combined score, and PCGrad roughly increases CPU training time. Keep both as optional ablations and retain their code because other geometry/label regimes may behave differently.

Interpretation: the main current limitation is not simply loss-scale imbalance. The smooth potential's physics identities are structurally satisfied, but the predicted Jacobian still has substantial held-out error. The next experiment should change the moment parameterization/head while preserving the P-CRAFT PINO trunk, using the W6 M7 logistic realizable-atom derivative as a borrowed head candidate.

## Important scope

Labels remain analytic continuum vertical-plane P0 only, not Cherchi. No Newton admission, arbitrary quadric certification or real Spin end-to-end speedup is claimed. The workflow's separate PINO unit-test job failed because the test file lacked imports for `nn` and `pcgrad_backward`; that test-import issue was corrected in the subsequent commit and rerun separately. The M4 training/conditioning job itself passed.
