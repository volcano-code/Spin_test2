# H8f point-000 physical surfaces

The pinned W1 C++17/GMP auditor was built from verified source and fed exact rational coordinates in the physical model frame. Inputs are the frozen request design triangles and the archived binary64 source quadric triangles, decoded as exact rationals. The saved difference uses the physical `material_proof.exact_vertices` with the recorded output winding. [Machine-readable results and exact volumes](M22_P4_POINT000_SOLID_BOUNDARIES.json)

| Surface | Vertices / edges / faces | Candidate pairs checked | Nonadjacent | Euler characteristic / genus | Signed volume, approximate |
|---|---|---:|---:|---|---:|
| Design | 13,122 / 39,366 / 26,244 | 218,939 | 60,772 | 0 / 1 | 1.0998850071299318 |
| Faceted cutter | 162 / 480 / 320 | 1,986 | 96 | 2 / 0 | 1.3881363252327446 |
| Saved difference | 8,455 / 25,365 / 16,910 | 161,702 | 49,727 | 0 / 1 | 0.4638104336177978 |

Every triangle is nondegenerate. Each edge has one incident face in each direction, the link of each vertex is connected, and the exact checker found zero illegal self-intersections. Each mesh has one connected shell. The exact signed volume from oriented tetrahedra is positive in each case, giving an outward orientation of the bounded interior. Each operand therefore has winding one inside and zero outside. The design and saved result each have a handle. For the saved output the physical-frame signed volume also equals the archived exact fraction. These conclusions pertain to the three triangulated solids at this one input.

The frozen Python producer passes design as the first operand and the physical quadric-source mesh as the second. Its `difference` branch is stored as `subtraction` in the material. The native bridge maps that operation to `SUBTRACTION`. The material digest and request digest in this report match the earlier W2 point-000 replay. [Frozen producer](https://github.com/volcano-code/Spin_test2/blob/0bcc187b3005478baa586a6d6f450e73811ca342/python/make_it_stand/verification/cherchi_inprocess.py#L182) · [Frozen bridge](https://github.com/volcano-code/Spin_test2/blob/0bcc187b3005478baa586a6d6f450e73811ca342/bindings/cherchi/InProcessBoolean.cpp#L200)

W0 can combine these input and output surface properties with the independent per-parent coverage tests to assess the regularized Boolean region. The surface audit alone checks no design-versus-cutter patch selection. This execution used zero fresh native Boolean calls, leaves the M21 attachment and evaluation-map issue open, and admits zero training samples and no Jacobian.

## Reproduce

From the W2 checkout, with a C++17 compiler, Boost headers, GMP and GMP C++ development files installed, supply the original H8f ZIP. Fetch the indicated W1 Git object from Spin_test2 if using a single-branch clone. The runner verifies the source SHA-256 before building a fresh executable. Compiler and library differences may change the executable digest.

```sh
git show 249db334303b31b8b8c2cfcb415b21a7cc2c9464:research/NeuralMoments/performance/m26_exact_intersection.cpp > /tmp/w1-m26-point0.cpp
cd research/NeuralMoments/teacher
python m22_point0_solid_boundaries.py --h8f "$H8F_ZIP" --checker-source /tmp/w1-m26-point0.cpp > point0-solids-replay.json
python -m unittest -q test_m22_point0_solid_boundaries test_m22_point0_operand_replay
```
