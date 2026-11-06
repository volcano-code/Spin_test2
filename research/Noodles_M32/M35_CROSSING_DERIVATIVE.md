# Derivative of the discrete noodles cut

The fixed design volume is the 94,150-face M32 noodles D′. A full-rank triaxial cutter crosses its boundary at the registered W2 sample. The variable material is the positive side of the original native cutter after tessellation and exact libigl/CGAL clipping. The source is the original fixed-normal tessellation with 162 vertices and 320 triangles. All admissible velocities refer to that discrete surface. The fixed shell contributes to retained moments but has zero coefficient derivative.

For each retained triangle with a cutter birth face, interpolate the native source-vertex velocities barycentrically on that parent. Integrate each degree ≤2 monomial times the outward normal velocity on the exact clipped triangle. The integrand has degree ≤3, so the four-point degree-three triangle rule integrates it exactly. Accumulate the rational contributions with outward 140-bit dyadic bounds. Reverse the parent orientation check for a retained difference. The reference order is mass, x, y, z, xx, yy, zz, yz, xz, xy; native cross coefficients multiply twice the corresponding monomial. Native columns refer to the unit tip-frame coefficient before an ambient perturbation and subsequent normalization. Use the tip translation represented by the original binary64 value −0.16.

The independent path reconstructs transverse edge/face intersection velocities from their incident input faces and differentiates signed tetrahedral volume and monomial formulas. The 243 output vertices comprise 41 fixed vertices, 65 moving cutter vertices and 137 edge/face intersections. All 100 component intervals from the two formulas overlap. Their maximum possible discrepancy for the represented input motions is 2.61×10⁻⁴⁰. An independent bordered-system solve also checks the 4,860 original native ellipsoid source-velocity scalars, with maximum scaled discrepancy 1.72×10⁻¹⁴.

## Central differences using actual independent Boolean calls

W2 evaluated thirteen exact full-design Boolean calls. W1 independently certified all thirteen retained sets and supplied exact ten integrals. Construction, oriented triangles and seam fingerprints agree at all sampled points, including the 137-segment seam loop. The base output has 217 design-parent faces and 265 cutter-parent faces. The errors below are maxima over all ten moment rows of |central difference − flux| / max(1, |flux|) in the tip frame.

| Native column | Step 2⁻¹⁹ | Step 2⁻²⁰ |
|---|---:|---:|
| 0, constant | 1.671e−7 | 4.176e−8 |
| 1, x | 1.286e−8 | 3.131e−9 |
| 8, xz | 1.326e−7 | 3.305e−8 |

All 60 scalar comparisons meet the 1e−6 sampled gate. Agreement at endpoints and center qualifies these samples. It does not bound higher derivatives over the whole segment.

The base tip-frame derivative has maximum radial residual |Jc| of about 9.25×10⁻¹⁷. In monomial coefficient coordinates, project both sides of J onto the tangent with normal B⁻¹c, the convention for W6's native logistic temperature. A symmetric tangent operator has relative Frobenius error at least 0.000470293323 on this unshifted example. This expression concerns the model initialized in the tip frame. Transporting the network to another frame also transports the normalization metric. It is distinct from the componentwise finite-difference gate.

## Supplemental checks on all ten native columns

W2 also evaluated 41 exact Boolean queries after clipping D′ to a box that contains each entire cutter strictly in its interior. On this intersection chart, associativity makes the crop exact. The raw crop triangles are mapped back to original design and cutter parents. The thirteen overlapping controls match the direct full-design geometry and parents. Their exact ten integrals also agree on every shared perturbation. W1 supplied independent ten integrals and verified the 112-bit enclosures, original parent/triangle/seam fingerprints, and strict feature tests against the full D′ for all 41 queries.

Against the original 100-entry flux Jacobian the largest scaled discrepancy over all native columns and moment rows is 7.11001×10⁻⁷ at step 2⁻¹⁹ and 1.77700×10⁻⁷ at step 2⁻²⁰, both in column 6 (zz). All 200 scalar comparisons meet the 1e−6 gate. This supplemental stencil reuses the same ellipsoid geometry, base, coefficient chart and thirteen primary parameter values. Six ancillary normalized coefficient values differ at the final floating-point bit between Python and Eigen. The stencil uses raw input perturbations, original native cutter coordinates and native derivatives from the actual source executor.

```bash
python crop_stencil.py "$CROP" "$W1_SUPPLEMENT" "$W1_CROP_TEN" M35_CROSSING_DERIVATIVE.json > all-column-crop.json
```

## Original stock seed

At the first stock hyperboloid iterate W2 used the original general source, including truncation, and independently verified no cap-owned face in the result. W1 checked the regularized retained set. The 14,092-face result has 1,678 moving faces. The independent construction recovers 5,350 fixed vertices and 1,598 transverse edge/face intersections. The two full Jacobians agree to within 2.12×10⁻³⁹ for the represented input geometry and velocities. Direct source-vertex stencils at two steps probe 2,280 source-vertex/native-column components, including cap-associated vertices. Maximum scaled source discrepancies are 5.18×10⁻⁶ and 1.30×10⁻⁶ at the two steps. They check the moving source, separate from the full Boolean/moment stencil on the registered ellipsoid. The two independent analytic matrix checks share the supplied native source velocities.

## Reproduction

Generate the real input packets with W2's teacher scripts and the W1 independent moment files with its M35 performance tools. Each sample directory contains cutter.txt, source.json, result.json and receipt.json. The stencil directory also contains protocol.json. The W1 directory contains sample.w1.json and sample.ten for each of the thirteen names. From this directory run the following commands with the paths to those materials.

```bash
python crossing_stencil.py "$STENCIL" "$DPRIME" "$W1" > ellipsoid-derivative.json
python crossing_report.py "$STOCK" --output stock-flux.json
python crossing_vertex_motion.py "$STOCK" "$DPRIME" stock-flux.json > stock-vertices.json
python generic_source_stencil.py "$STOCK" "$BOUNDS" "$BOUND_SOURCE_EXECUTABLE" --workers 4 > stock-source-stencil.json
python -m unittest discover -p 'test_crossing_*.py' -v
```

The selected stock executable must match the SHA in the original receipt. The checked JSON includes the full flux, dyadic bounds, distinct cross-checks and source identities. Source vertices are emitted in floating point by the upstream implementation. Exact integration and intervals condition on these represented inputs. The results qualify local evidence for this discrete teacher. Material error for the continuum quadric, additional models, terminal optimization accuracy and end-to-end speed require separate measurements.
