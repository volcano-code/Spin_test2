# Heart point-000 as a strict Newton initial guess

The [receipt](H9_POINT000_STRICT_INIT.json) tests what the existing Heart request would do **if its point-000 exploratory sphere were supplied as an initial guess**. The saved target-frame total moments and native coefficients come from the SHA-pinned M23 capture. The coefficient vector equals the registered H8 exploratory sphere byte for byte, including its signed zero.

The active spinning-top objective is the squared transverse inertia ratio plus the center-of-mass weight times height. The weight is zero. The four equality constraints are the first moments x and y and the central products of inertia I13 and I23. The replay uses the default full-space objective with no Spin-It-2014 override. The binding selects this objective in [pybind_module.cpp](https://github.com/volcano-code/Spin_test2/blob/25d7187c30f647f04736eeb3fd8f9b1b9daf11da/bindings/pybind_module.cpp#L2610). It passes the literal quadric through the original normalized dual-monomial transform. The C++ probe computes the production objective gradient and constraint Jacobian, and solves the same least-squares projection with Eigen's complete orthogonal decomposition. Computing the initial multiplier requires only the base moments, objective, constraints and coefficients.

The multiplier is −16.812030434192025. The strict solver rejects this initial guess before constructing a Newton direction. The registered seeded solve can still try its deterministic starts. An accepted update requires a positive-length trial with fresh native material, its own moment map and decreased raw residual. The actual solver also records topology transitions and relinearizes accepted transitions. The final equation and physical tolerances remain 10⁻¹² and 10⁻⁶. [Initial test and trial rule](https://github.com/volcano-code/Spin_test2/blob/25d7187c30f647f04736eeb3fd8f9b1b9daf11da/src/spin/QuadricHollowingSolver.cpp#L1046)

Run at repository root with Eigen headers and the saved base directory. The receipt replays recorded moments. It entails zero Cherchi calls.

```bash
dir=research/Heart_v1_20261017/Heart_H8_Asymmetric_CAD_20261018
c++ -std=c++17 -O1 -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -Iinclude -I"${EIGEN_INCLUDE_DIR:-/usr/include/eigen3}" \
  "$dir/source/h9_strict_init_multiplier.cpp" src/spin/SpinMomentProblem.cpp \
  src/spin/DualMonomialCoordinates.cpp src/spin/QuadricCoefficients.cpp -o /tmp/h9-strict-init
python "$dir/source/h9_strict_init_preflight.py" "$BASE" /tmp/h9-strict-init
```
