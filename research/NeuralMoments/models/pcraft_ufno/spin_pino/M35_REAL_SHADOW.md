# Frozen PINO on the discrete noodles teacher

This offline check uses one original D′ mesh and three preselected coefficient queries. The variable target is the positive material cut of D′ in the original tip frame. W2 supplied each libigl/CGAL result, W1 supplied independent moment evidence and W3 supplied the discrete shape derivative. The JSON fixture records the actual q, clipped-output SHA, ten target values, full native Jacobian and the 8³ point occupancy. The fresh stock hyperboloid and the two ellipsoid iterates probe different cut sizes on the same parent. The two ellipsoid cases share a trajectory.

All three models replay the analytic-only single_physics arm. Each uses effective density, occupancy, two zero channels, 216 midpoint logistic atoms and temperature 0.12. The analytic-only training and validation splits, 120-step fit, physics weight 0.05 and seeds 131/137/139 match the prior experiment. Their analytic validation checkpoints are steps 20, 20 and 60. The real queries enter only after the checkpoint is frozen.

## Relative error in the tip frame

The relative ten-vector and full 10×10 matrix errors are averaged across the three frozen seeds per query. Values are ratios.

| Query | True variable mass | Range of predicted mass | Ten relative L2 | J relative Frobenius |
|---|---:|---:|---:|---:|
| Ellipsoid, evaluation 1 | 0.00073604 | 0.06404–0.07562 | 70.1984 | 0.59891 |
| Ellipsoid, evaluation 6 | 0.00074981 | 0.07399–0.08842 | 79.2029 | 0.62384 |
| Original stock hyperboloid, evaluation 1 | 0.13331743 | 0.07308–0.09379 | 0.46893 | 0.74261 |

Pooling the three predictions per seed before forming relative norms gives mean errors of 0.90974 (ten) and 0.61512 (J). The larger hyperboloid moments dominate the pooled ten denominator. The JSON keeps each predicted vector and full matrix. These three related queries carry no geometry-population confidence interval.

## Why fitting the encoder alone cannot fix the small-cut mass

Sixteen of the 512 point-grid samples contain material. Only 25 of 216 midpoint atoms have positive interpolated density. For the fixed geometry, the PINO predicts one nonnegative weight per supported atom, shared across all coefficient queries. Write αᵢⱼ for the activation at query i and atom j. Because all reference activations are positive, the predicted mass ratio to query k is a weighted average of αᵢⱼ/αₖⱼ. It must lie above the smallest supported ratio.

For ellipsoid 1 versus the stock hyperboloid, the true ratio is 0.005521. The smallest supported activation ratio is 0.0671913. For any choice of nonnegative weights, at least one of these two masses must therefore have relative error of at least 84.8142%. The ellipsoid 6 comparison gives 62.1603%. This is a post hoc capacity check on the binary32 activations of this head and grid, evaluated as rational numbers before final floating-point summation. The choice of temperature, sampling lattice and density representation all enter the bound. Changing them requires a new held-out experiment.

## Coordinate check and reproduction

The unit half-width cube is centered at d=(0,0,1) in the original tip frame, with grid coordinates y=x−d. Define S by m(x)=S m(y), where m denotes the ten ordered monomials, and let B=diag(1,1,1,1,1,1,1,2,2,2). Network inputs are raw coefficients q_grid=T q_tip, with T=B⁻¹ Sᵀ B. Network outputs in the tip frame are S r_grid and S J_grid T. The internal temperature includes the native coefficient norm; there is no explicit renormalization in the adapter. The unit tests compare all 100 transported derivative entries against autograd and two-sided finite differences of the translated prediction.

```bash
pkg=research.NeuralMoments.models.pcraft_ufno.spin_pino
python -m "$pkg.real_cgal_shadow" --output real-shadow.json
python -m unittest "$pkg.tests.test_real_cgal_shadow" -v
```

The result supports testing a refined quadrature/support and narrower or adaptive transition on held-out cuts before solver insertion. This panel contains no real-label adaptation, neural Newton evaluation, original Cherchi timing, terminal result or mass-optimization comparison.
