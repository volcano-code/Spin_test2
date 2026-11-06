# A feasible retained mask on the reconstructed Ellipsoid

The registered Ellipsoid case in the 2014 reproduction uses the z axis through the origin, a 16³ voxel grid and the project-generated Figure 6 structural counterpart. The archived first viewer run points to the same occupancy SHA-256 as the paper-case manifest, `6036f03ff83777a636cbb8cf67bd57675caa44b90c40b3d2e442f5fb27d85983`. Both manifests identify the asset as reconstructed. The archived run used octree level 4.

The verifier checks every oriented grid face against the recovered cell memberships. The frozen source mask has 608 occupied cells and 352 carveable cells. The viewer's `material.obj` and `outer.obj` are byte-identical outer boundaries. Its inward-oriented `cavity.obj` removes 336 cells, all in the design region. The combined boundary leaves all 256 locked cells and 16 design cells, for 272 retained cells. At side length 1/8, their exact total mass is 17/32 at unit density.

Integrating all ten monomials over these cubes gives the registered-frame center of mass (0, 0, 1). The two lateral first moments and the two central cross inertias involving the spin axis all vanish exactly. This construction demonstrates a feasible physical-equality witness for this specific voxel asset and axis. A 2024 strict Newton terminal, the corresponding quadric geometry and physical spinning measurements remain open.

```bash
python -m research.NeuralMoments.contracts.ellipsoid_mask_witness experiments/spin_it_reproduction
```
