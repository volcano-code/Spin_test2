# Feasible continuous single quadric in M32

There is a single, strictly interior ellipsoidal cavity in D′ for which the remaining M32 body has zero lateral first moments and zero mixed moments involving the specified spin axis. It retains 99.9834305100% of the full object's axial inertia and removes about 0.0218840% of its mass. The proof is analytical on the registered polyhedral domain and uses the authenticated signed ten moments of the original A. The [receipt](M32_SINGLE_ELLIPSOID.json) includes outward intervals, the support/winding counts and approximate normalized coefficient vectors. These vectors can initialize future numerical studies. The exact claim applies to the analytic shape below.

## Construction

Write the ellipsoid as c+B u for ||u||≤1. Its center is (X,Y,Z); the first two columns of B are (a C,a S,0) and (−b S,b C,0) and the last is (sx,sy,h). Here (C,S) is the horizontal yaw unit direction constructed from a rational half-angle tangent. Fix the five rational values X,Z,a,h,t recorded in the receipt. The original whole-body raw moments of A determine the other parameters exactly:

- V=Lx/X, Y=Ly/V and b=3V/(4πah).
- sx=5(Qxz/V−XZ)/h, sy=5(Qyz/V−YZ)/h.

The ball's volume is 4π/3, its first moment is zero and its mean uuᵀ is Id/5. The cavity therefore has volume V and weighted moments (Lx,Ly,Qxz,Qyz) exactly equal to those of A. Subtracting this cavity gives the four desired equalities simultaneously. The mass bound uses unit density, and Izz is calculated about the original z axis through the fixed tip. A separate interval check shows the centroidal Izz exceeds the largest transverse principal inertia by at least 0.132682477714054 in model units, using a Gershgorin bound.

## Containment

The verifier encloses the full body's exact moments by outward dyadic intervals. A pair of alternating arctangent series and Machin's identity enclose π. A nominal rational center and matrix enclose the true ellipsoid to within an additive axis-aligned error box of half-width below 4.29×10⁻¹³ model units. Exact AABB tests separate 92,723 D′ boundary faces. For every other triangle, an integer vector in [the certificate](M32_ELLIPSOID_SEPARATORS.txt) separates all three exact vertices from the supremum of the ellipsoid support in that direction, including the box error. All 1,427 directions verify by rational arithmetic. An exact ray from the nominal center has winding 1; this center lies inside the true ellipsoid. Together with exact embedding and orientation of D′, this places the complete ellipsoid within its interior.

The native coefficient order is constant, x, y, z, xx, yy, zz, 2yz, 2xz, 2xy, with material retained for β≥0. Coefficient hints are Euclidean-unit native vectors in unscaled model coordinates about (0,0,0) and the tip respectively. Discrete source tessellation, data admission, printing, convergence and local or global optimality still require their own evidence.

## Replay

Use Python and the M34 Dprime.txt operand plus the W1 A.ten exact hexadecimal fraction moments. The standalone verifier uses the standard library. The integer separating directions were proposed with floating-point nearest-triangle calculations on a nominal ellipsoid; their validity is checked independently and exactly.

```bash
python -m unittest test_single_quadric_witness
python single_quadric_witness.py /path/to/Dprime.txt /path/to/A.ten > /tmp/m32-ellipsoid.json
cmp /tmp/m32-ellipsoid.json M32_SINGLE_ELLIPSOID.json
```
