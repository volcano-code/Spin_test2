# Spin_test2 W6 analytic replay

This package ports the Spin-PINO training and evaluation code from `volcano-code/Spin_test` at `6bdaca7cc72cf56a1588549e9c148fc16fb0cc23` onto the Spin_test2 W6 branch at `a3fe82ddcc5b418630411017b90cc8502de1ae18`. The target’s independent density-gradient regression is retained. The P-CRAFT encoder pins and same-project reuse status are recorded in the [source audit](../PINO_CORE_SOURCE_AUDIT.md). M0–M5 status records and linked runs refer to Spin_test; its candidate adapters are separate from this training package.

The three-seed analytic M5 tournament was rerun with the six source configurations. For the logistic-atom model, the mean ellipsoid-test relative L2 errors were 0.20602048 for the ten raw moments and 0.31465573 for the Jacobian. The same-size smooth midpoint baseline gave 0.23204 and 0.89120. The held-out ellipsoid family has previously been used to choose models. Independent new geometries are needed to measure generalization after model selection.

The orientation study uses the same 16 held-out queries per seed in each training arm, 8 ellipsoids per seed, and 3 seeds (131, 137, 139). Every score uses 120 training steps at batch size 8 and voxel inputs of size 8³. Each transformed-only panel contains one rotation per query. The Haar direct panel resamples the continuous ellipsoid at voxel nodes with the same rotation and labels as the trilinear panel. The run checks the geometry IDs against the frozen generator; panel hashes cover all inputs, coefficients, labels and query metadata. The table shows three-seed mean r/J relative L2.

| Fit augmentation | Original | Cube only | Haar only, trilinear | Same Haar, direct voxels |
|---|---|---|---|---|
| None | 0.20602 / 0.31466 | 0.20600 / 0.31455 | 0.37123 / 0.44952 | 0.19277 / 0.31782 |
| Cube | 0.20556 / 0.31432 | 0.20562 / 0.31430 | 0.37072 / 0.44900 | 0.19262 / 0.31761 |
| Haar SO(3) | 0.20557 / 0.31428 | 0.20554 / 0.31427 | 0.37066 / 0.44891 | 0.19263 / 0.31764 |

The Haar interpolation-versus-direct occupancy relative L2 was 0.416–0.421 across seeds. The source task consists of analytic interior plane cuts; these results measure neither true CAD cut events nor native Newton speed. The unchanged M5 support weighting is the baseline for the subsequent density/occupancy ablation.

## Replay

Use the project root and a CPU environment with PyTorch, NumPy and pytest. The W6 workflow pins a PyTorch CPU wheel and publishes both raw JSON outputs. Its M5 and paired mean regression tolerance is 0.001 absolute in relative L2 units. The [local receipt](SPIN_TEST2_ANALYTIC_REPLAY.json) includes the source pins, runtime version and panel digests. Training times are machine dependent.

```bash
python -m pytest -q research/NeuralMoments/models/pcraft_ufno/spin_pino/tests
python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.tournament --output /tmp/w6-m5.json
python -m research.NeuralMoments.models.pcraft_ufno.spin_pino.rotation_sweep --output /tmp/w6-paired.json
```
