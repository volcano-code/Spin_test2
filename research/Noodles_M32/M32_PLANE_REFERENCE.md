# W3 reference on M32's exact rational carve domain

D′ is the separately registered M32 carve domain. This test takes its authenticated M34 rational operand as input without a binary64 vertex conversion. It uses model coordinates at unit density and the positive side of β = −3/5 + (4/5)z = 0, i.e. z ≥ 3/4. The raw moments and ambient 10×10 Jacobian refer to this continuous implicit cut at this point. Fixed-material moments and a target-frame transform are still required for a whole-body evaluation.

The run uses 46,679 vertices and 94,150 triangles. Of the source faces, 70,244 are inside, 22,161 are outside, and 1,745 cross the plane. Every vertex has nonzero signed margin; its minimum absolute β is greater than 1.68×10⁻⁵. Perturbing the constant, x or y coefficient by ±2⁻²⁰ and ±2⁻²¹ leaves the signs of all vertices and the set of intersected source edges unchanged.

The C++ program clips the oriented input triangles with exact rational evaluations and intersections, closes each boundary edge with an oriented planar cap triangle and sums signed tetrahedra. The cap fan may overlap geometrically; its signed integral gives the winding-weighted cap. The separate derivative calculation integrates the Gram matrix of monomials through degree four over the signed xy section, divides by the z-gradient 4/5, and applies the factor two on native mixed coefficients. Each exact signed contribution is enclosed by adjacent dyadic rationals on a 2⁻¹⁶⁰ grid before summation. The [receipt](M32_PLANE_REFERENCE.json) includes enclosing intervals for all ten moments and 100 derivative entries. Finite differences reuse the volume computation and do not consume these surface derivatives.

The variable mass is approximately 0.6004750189560762. The largest absolute component errors for the three numerically checked columns are shown below; the acceptance check uses the worst distance between the exact outward enclosures with threshold 10⁻⁶.

| Ambient direction | h = 2⁻²⁰ | h = 2⁻²¹ |
|---|---:|---:|
| constant | 2.37×10⁻¹⁰ | 5.93×10⁻¹¹ |
| x | 8.17×10⁻¹² | 2.05×10⁻¹² |
| y | 7.76×10⁻¹² | 1.94×10⁻¹² |

The analytic box regression covers all 110 reported quantities, and a mesh vertex exactly on the base plane is explicitly rejected. The M32 probe uses linear-plane perturbations only; curved-cut stencils, strictly admissible native source ranks, whole-body constraints, learned labels and production Newton remain open.

## Replay

Use the M34 design operand, Python 3, a C++17 compiler and the GMP C++ development package. From this directory run:

```bash
c++ -O2 -std=c++17 exact_plane_probe.cpp -lgmpxx -lgmp -o /tmp/m32-plane
python -m unittest test_plane_probe
python plane_report.py /path/to/Dprime.txt c3633c74bc74c2be488eb22de108454276f28941f0477cfee5837f57db99d622 /tmp/m32-plane > /tmp/m32-plane-result.json
cmp M32_PLANE_REFERENCE.json /tmp/m32-plane-result.json
```
