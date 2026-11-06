# H8f point-000 cutter-side coverage

The archived subtraction has 320 cutter input triangles. Every positive-area region of each triangle was compared against the saved boundary using exact rational arithmetic. The replay found 48 whole triangles inside the Heart design, 187 wholly outside and 85 partially inside. The 784 input-derived cut segments and the saved triangulations subdivided the parents into 4,763 tested cells. All 876 cutter-side output triangles have subtraction winding. Every cell had the expected multiplicity; missing, excess and multiply covered projected areas are zero on every parent. The 876 pairs of output and parent IDs match the W2 construction-based assignment.

The independent keep predicate uses the original 26,244 design triangles. It computes an oriented crossing count from an exact open-cell witness, retrying other axes on an ambiguous ray. The cutter side belongs to the subtraction boundary where the witness is inside the design. The program rejects a nontransverse witness without a usable ray, a positive-area coplanar input intersection, an outward-wound saved patch, and an unexpected winding number. Binary64 AABBs only select candidates using inclusive comparisons; the incidence, intersection, winding and area decisions use exact coordinates. The archive, prepared request and compressed point-000 material are hash-checked before reading their geometry.

[Per-parent cells, areas and output IDs](evidence/m30-h8f-point-000-cutter-faces.json) complement [design-side coverage](M29_HEART_CUT_FACES.md). These data concern the relative interiors of the two operand boundaries of this saved historical subtraction. Separate evidence is needed for lower-dimensional boundary agreement, a global solid claim, attribution to a fresh native run, a smooth derivative or Newton admission. This replay performs zero native calls.

## Reproduce

Run from the repository root with NumPy and the frozen archive whose SHA-256 is `99a050f4604109e0323c0110a029aa5d7601d729d755e6956ad48ed8c77ccda9`. Choose a new output path.

```sh
P=research/NeuralMoments/performance
python "$P/m30_heart_cutter_faces.py" "$H8F_ZIP" --out /tmp/m30-cutter.json
cmp /tmp/m30-cutter.json "$P/evidence/m30-h8f-point-000-cutter-faces.json"
M30_H8F_ZIP="$H8F_ZIP" python -m unittest discover -s "$P" -p 'test_m30_heart_cutter_faces.py' -v
```

The eight new tests include a real mutation check. On parent 141, removing one of the 28 saved triangles creates a missing region, duplicating it creates positive overlap, and flipping the vertex order is rejected. The same pinned input gave byte-identical reports on two complete runs.
