# Frozen Heart target-axis feasibility

The W2 prepared request with SHA-256 `1d2b06063c5b2c7010fcb2764518c05c9707f013d6914bc4e20298f8516e061c` fixes the shell, design solid, positive uniform density and spin axis. The target axes are the identity and the target origin has x coordinate zero. The first original equality in the spinning-top problem requires the total raw first moment along target x to vanish. Its mass-normalized value is the x coordinate of the center of mass.

Let `D` denote the design solid and `S` any retained subset of `D`. The density in normalized coordinates is one. Retaining all material with negative x and removing all material with positive x minimizes the first moment even when arbitrary measurable subsets are allowed.

```math
C_x(S) = F_x + \int_S x\,dV
\quad\ge\quad F_x + \int_{D\cap\{x<0\}}x\,dV .
```

The locked shell has exact target first moment `F_x = 3623286431931019/18014398509481984`. Two independent rational flux calculations bound the negative design contribution between −0.109559210383829 and −0.109559210383828. The total first moment therefore remains at least **0.091573590693259**. Since the raw equation vector includes this component, its Euclidean norm exceeds the registered tolerance of `1e-12` for every admissible shape.

The shell and fully filled design together have mass below 2.035353206936568. Dividing the first moment bound by that maximum mass gives a target x center-of-mass offset of at least 0.044991498468753 normalized units. At the frozen scale of 25 mm per unit this is **at least 1.124787461718 mm**. The separate physical-relative constraint residual also exceeds the registered `1e-6` tolerance. Its numerator contains `C_x`. Its denominator is at most 18.905088562226178, from the sum of absolute fixed moments and bounding each variable monomial by its supremum over the design box times the full design mass. This gives a lower bound of **0.004843859386949**.

## Exact replay

From the W0 tree, run the following command with the frozen prepared request.

```bash
python -m research.NeuralMoments.contracts.heart_axis_feasibility prepared-request.json
```

The code interprets the 13,122 binary64 design vertices as exact dyadic rationals, checks the oriented edge closure and regenerates byte-for-byte the W1 design-boundary input with SHA-256 `eff414f5daaf52f8ae847bb851d3cbb5cfd604bce0f88a5a56f99ed6b1efb9ad`. W1's exact intersection checker examined all 218,939 candidate triangle pairs on this connected outward boundary of 26,244 faces. The replay clips the 553 faces that cross x=0, retaining triangle orientation, and integrates the divergence fields `(x²/2,0,0)` and `(0,xy,0)`. Both fields have divergence x and both give zero flux through the new cap. Their exact fluxes on the negative half agree. The positive and negative half moments additionally sum exactly to the first moment of the full solid, evaluated independently by signed tetrahedra. The displayed decimal intervals are rounded outwards.

The obstruction holds for every admissible cut under this specific frozen axis, locked mass distribution and design envelope. Other model-axis registrations, locked shells and objects require their own feasibility checks. The bound is independent of any sampled Newton trajectory or choice of neural model.
