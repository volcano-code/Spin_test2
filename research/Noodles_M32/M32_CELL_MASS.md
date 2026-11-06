# Minimum mass on two certified finite M32 design families

W1's 6,879 certified axis-aligned cells are entirely in the exact M32 carve domain. Centers have coordinates (i,j,k)/32 in the original model frame. A cavity has fixed x and y width 7/256, is centered in its own cell, and may vary in z height between zero and that width. Exact formulas use unit density in this frame. In the spaced family every center index is divisible by 3 in each axis. Each represented cavity lies inside its certified cell and the closed cavities are separated by a positive gap.

For removed volume v at center (x,y,z), the four balance contributions are v times (x,y,xz,yz). Its contribution to removed axial inertia is v times x²+y²+width²/6. Maximize the sum of v, cancel the four original whole-body moments exactly, and allow removal of at most 10% of the original axial inertia. Each variable lies between zero and width³. The published M32 equation system has the original z axis and the original tip. Reflection symmetries or an adjusted axis are not required.

HiGHS proposes a basis. The verifier inverts its 4- or 5-column system with rational arithmetic. The four right-hand sides are the exact W1 whole-body integrals; an active fifth row uses its exact inertia budget. Outward rational bounds certify all basic volumes strictly inside their bounds. Exact dual multipliers are recovered from the transposed basic system, and all reduced costs are verified. This proves the optimum for each finite family. It gives a feasible upper bound for the mass over arbitrary admissible cuts. These box cavities have a different representation from Spin-It Faster's single quadric.

| Design family | Candidate cells | Nonempty cavities | Retained mass | Retained axial inertia | Minimum grid spacing between cavity surfaces |
|---|---:|---:|---:|---:|---:|
| Full selected lattice | 6,879 | 6,811 | 90.32951% | 91.96643% | 0.09765625 mm |
| Every third cell on each axis | 242 | 226 | 99.68279% | 99.76531% | 1.66015625 mm |

From the verified four balance equalities, the registered axis is a principal axis. Conservative centroidal Gershgorin bounds put its inertia above both transverse principal moments by at least 0.1202 and 0.1322 unit-density model units respectively. The computed center-of-mass heights above the protected tip are about 26.581 and 26.568 mm. Actual spinning time remains unmeasured. The inertia inequality is slack in both optima. The finest result has an approximately 0.0214 mm partial-cavity height. The spaced result has an approximately 0.0239 mm partial-cavity height. Mechanical strength, fabrication process limits, practical support removal and physical spin measurements remain open for either family. An early coarse layout requiring all 27 neighbors in each block to pass W1's interior test provided fewer than five candidate sites and was dropped; this is a property of that conservative enumeration.

The complete retained material is the authenticated outer solid A minus the listed interior cavities. The JSON stores the inclusive index ranges of full cavities (ordered lexicographically in the family), coordinates of basic cells, the exact inverse matrix and dual multipliers. To recover the nontrivial exact volumes, multiply the stored inverse by the exact target vector minus the full-volume contributions of the indicated upper cells. All other volumes are zero. All intervals in the receipt are rounded outwards.

To replay with Python, NumPy and SciPy, supply the eligible-cell list and A.ten from the W1 M34 exact audit. Both source files are SHA-bound.
```bash
python research/Noodles_M32/cell_mass_lp.py /path/to/all-eligible-cells.txt /path/to/A.ten
python -m unittest discover -s research/Noodles_M32 -p test_cell_mass_lp.py
```

[Independent exact geometry and cell verification](https://github.com/volcano-code/Spin_test2/blob/e0ceffd2595f2adff27a97af0058c8531f6918e7/research/NeuralMoments/performance/M34_NOODLES_EXACT_PARTITION.md) · [Single-ellipsoid feasibility](M32_SINGLE_ELLIPSOID.md)
