# Heart H8f point-000 design-side cut faces

At this archived evaluation, all 696 design triangles cut into both positive-area retained and removed regions match the saved output in relative interior, including the winding of all overlapping saved triangles. This audit examined 26,244 design triangles, 1,134 saved output triangles on the cut parents and 3,200 exact subdivision cells. Each cell ledger reports zero missing, extra and multiply covered area. The first such design face (zero-based 3456) contains saved faces 14948 and 14949 and has six cells.

## Evidence and replay

Use Python 3.10+ and NumPy. From this directory run

```bash
python -m unittest -v test_m28_cut_cell_coverage test_m29_heart_cut_faces
python m29_heart_cut_faces.py /path/to/H8f_12135528290.zip --out /tmp/heart-cut.json
cmp /tmp/heart-cut.json evidence/m29-h8f-point-000-design-cut-faces.json
```

The full archive and each consumed archive member are SHA-256 pinned. The design operand comes from the saved prepared request. The cutter and the exact model-frame boundary come from the historical point-000 packet. Use the physical model coordinates in `material_proof.exact_vertices` for this comparison; the translated target-frame coordinates used for the moment integral differ.

The program verifies that the 320 outward cutter triangles form a connected, closed, consistently oriented convex mesh, using exact half-space tests on every vertex. It clips the design parent against these half-spaces and derives the expected regular subset of the subtraction from these operands independently of the saved output. It also scans saved triangles conservatively, finds exact coplanar overlap and checks winding and containment. Projected input cuts and actual output edges partition each parent. At every open cell a rational witness is tested against the cutter half-spaces and the saved triangles. The reported areas are rational areas in the stated coordinate projection. Contacts confined to edges or points are outside this positive-area test.

Historical H8f point-000 and the W1 M28 evaluations 4/116/117/123 use different input archives. M29 does not resolve any of the 1,555 cut parents in the separate M28 archive. The W2 M21 evaluation-map identity remains distinct from H8f. This run neither checks every uncut parent nor the cutter-owned subset of the output, and therefore does not certify global Boolean set equality. It supplies no derivative, Newton, training or speed result.

## Follow-up covering every design parent

The full design-side audit subsequently passed all 26,244 parents. Their expected relative interiors split into 14,900 fully retained faces, 10,648 entirely removed faces, and the 696 cut faces above. A plane-indexed search checks every coplanar saved triangle whose conservative bounding box overlaps a parent. Positive-area crossings of a parent perimeter, wrong windings, duplicate assignment and positive-area coincidence with a cutter facet are rejected. A fully retained face must be reproduced by the identical triangle or pass an exact per-cell coverage audit. A fully removed face must have no positive-area saved coverage. The 16,034 saved design-side triangles and their individual parent IDs exactly matched the independently constructed W2 incidence map (canonical SHA-256 `c1de258442374ba9678da93b3f12e4cfc150d82dabcbefcdc966a1d0ea2e0270`).

```bash
python m29_heart_cut_faces.py /path/to/H8f_12135528290.zip \
  --all-design-ledger /tmp/design-parents.jsonl --out /tmp/all-design.json
cmp /tmp/all-design.json evidence/m29-h8f-point-000-all-design.json
sha256sum /tmp/design-parents.jsonl
```

Each JSONL row stores the design face index, classification and all overlapping saved face indices. The expected digest is in the committed summary. The 876 cutter-owned output triangles still require coverage against the intended subset of the 320 cutter faces. This follow-up closes the design-face relative-interior obligation on historical H8f point-000 only.
