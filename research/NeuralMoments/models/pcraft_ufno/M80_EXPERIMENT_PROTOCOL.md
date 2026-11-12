# M80 Experiment Protocol

## 1. Preflight is not model-quality evidence

A preflight proves that the selected model/data path can complete forward, Jacobian backward, optimizer update and checkpoint round-trip on one large object. It does not prove trained accuracy, generalization, solver acceleration or material reduction.

The four supplied single-scale Large preflights passed:
- P-CRAFT baseline: 457,130 parameters, 0.6689 s optimizer step, 123.3 MiB peak allocated.
- P-CRAFT + GNO: 522,747 parameters, 1.1437 s, 687.5 MiB.
- P-CRAFT + Transolver: 552,895 parameters, 0.6952 s, 1,258.4 MiB.
- P-CRAFT + GNO + Transolver: 618,512 parameters, 1.1337 s, 1,693.3 MiB.

The largest object had 194,275 native intervals at resolution 512. These were one-step random-initialization resource checks; the reported losses are not trained-quality comparisons.

## 2. Architecture factorial

Single-scale M80.3 2x2:
1. P-CRAFT Native.
2. P-CRAFT Native + GNO.
3. P-CRAFT Native + Transolver.
4. P-CRAFT Native + GNO + Transolver.

M80.4 follow-up:
- Repeat the same 2x2 under shared-weight 32/64/128 P-CRAFT.
- Add a separate GNO graph-policy comparison: global Euclidean kNN vs 6-connected-component-filtered kNN.
- Compare GeoTransolver and FNOGNO only after the best current encoder is selected; do not add them all to one model.

## 3. Matched budgets

For each comparison, pin:
- exact manifest and audit fingerprints;
- geometry-group split;
- seed schedule;
- optimizer and learning-rate schedule;
- query batch and query sampling policy;
- chunk size and AMP policy;
- parameter count and wall-clock budget.

Report both:
- same optimizer update count;
- same GPU wall-clock budget.

Use at least three seeds for promotion decisions. A 200–500 step run is an engineering pilot only; 2k–5k is a screening run; 10k is a formal pilot; 40k–100k is a long-training candidate after gates pass.

## 4. Required metrics

### Prediction
- ten raw-moment component errors;
- scaled Moment MSE and per-query RMSE P50/P95/P99/max;
- Jacobian component and directional errors;
- per-query Jacobian RMSE P50/P95/P99/max;
- Jacobian symmetry/PSD diagnostics only on certified smooth neighborhoods;
- HVP error when a validated HVP target/service exists.

### Reliability
- derivative-valid fraction and reason distribution;
- event classification recall/precision when event labels exist;
- uncertainty calibration and risk-coverage;
- false derivative acceptance (target zero).

### Cost
- parameter count;
- cold geometry preprocessing and encoding;
- warm moment query and warm Jacobian query;
- optimizer step and total training throughput;
- peak allocated/reserved GPU memory;
- graph build/cache costs;
- exact teacher calls and fallback rate.

### Downstream
- Newton iteration count;
- exact correction calls;
- strict terminal success/failure;
- final material mass under the same constraints;
- full cold-start and warm-start end-to-end time.

## 5. Test split policy

Validation selects architecture and hyperparameters. Test is evaluated once after the model/config/data contract is frozen. Do not use test metrics to decide which module to add or which checkpoint to promote.

## 6. Promotion rule

A module is promoted only if it improves held-out geometry tails or downstream solver outcomes at acceptable total cost. A faster preflight, lower parameter count or lower training loss alone is insufficient.
