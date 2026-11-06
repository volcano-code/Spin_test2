# Reconstructed Ellipsoid input for strict 2024 hollowing

The frozen 2014 reconstructed case has 608 occupied cells, including 352 design cells and 256 fixed cells. Its constraints can be evaluated at the recorded origin and +z axis. Three properties prevent transfer of the unchanged case to the strict hollowing and physical-contact pipeline.

| Condition | Geometric witness in original model units |
|---|---|
| Positive clearance of the entire design domain from the outer boundary | The two tracked triangle meshes share 208 voxel edges. One runs from (−5/8,−3/8,7/8) to (−5/8,−3/8,1). Their minimum distance is zero. |
| Embedded boundary of the fixed region | The exact boundary of the fixed cell union has 208 edges incident to four exposed faces apiece. |
| Material at the declared physical pivot | The lowest occupied layer starts at z=1/2 and contains the spin axis. Its horizontal footprint has area 7/16. The distance from the frozen origin (0,0,0) to the solid is exactly 1/2. |

The first two witnesses concern the registered geometry pipeline. `SpinArtifactProblem` requires a positive `minimum_outer_wall`. `check_fixed_boundary_shell` verifies that clearance for the whole design domain. Physical point contact is a separate application requirement. An abstract mathematical spin axis can pass through an origin outside the solid. The preflight pins the original voxel grid, target and OBJ bytes, computes boundaries from the input cells, and verifies that both surface OBJ files contain the same lattice edges. It does not evaluate a new 2024 trajectory.

A subsequent separately registered printable variant would need a specified contact on the spin axis, a positive wall around its full design domain, and an embedded fixed-region boundary. Its material coverage, constraints and density would need new checks before any native solve. The present fixture remains available for 2014 constraint and reconstruction tests.

From this directory run the following commands.

```bash
python m31_ellipsoid_input_preflight.py
python -m unittest test_m31_ellipsoid_input_preflight
```

The source contracts are [SpinArtifactProblem](../../../python/make_it_stand/verification/spin_problem.py), [fixed boundary shell](../../../python/make_it_stand/verification/fixed_boundary_shell.py), and the [strict material bridge](../../../python/make_it_stand/verification/faster_material.py).
