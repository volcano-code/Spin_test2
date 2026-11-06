# M34: design input transport into the native evaluator

M34's rational design solid D′ has 46,679 distinct exact vertices and 94,150 faces. Converting its model coordinates to the doubles consumed by the locked Cherchi adapter changes 65,395 coordinates on 23,207 vertices. The greatest absolute change is approximately 1.11×10⁻¹⁶ model units. In the represented input 938 independent points coalesce; 1,877 positive-area faces become exactly degenerate (1,876 have coincident corners, one has three distinct collinear corners). The [receipt](M34_DPRIME_BINARY64_TRANSPORT.json) identifies the exact rational and rounded hex coordinates of face #25819, using zero-based indices. The pinned W1 GMP checker rejects the represented surface with `DEGENERATE_EXACT_TRIANGLE`.

At W2 `2a79008d`, [InProcessCherchiEvaluator](../../../python/make_it_stand/verification/cherchi_inprocess.py) fixes its physical design operand as a float array (constructor), passes that array to the Boolean binding in `_boolean_with_valid_source`, and assembles the moving-cut differential using per-vertex source construction records. The [binding](../../../bindings/cherchi/InProcessBoolean.cpp) accepts `std::array<double,3>` for both input meshes. Exact rational vertices for the resulting Boolean surface and exact ten-moment integration describe this represented input. The previously verified W4 P23/P24 token transport normalizes rational output text for retained mesh snapshots.

The [CGAL reference driver](../../../experiments/faster_core_repair_20261012/exact_input_boolean.cpp) accepts the original rational D′ together with a second rational operand and exports rational vertices, faces and per-face parent identifiers (`birth_faces`). Thus it can supply geometry and an independent moment value for a trial. Reusing the existing Cherchi differential also requires point/edge construction and source-motion provenance. A different exact backend would need a separately verified derivative, for example a triangulated boundary-flux derivative using moving source face velocities, plus construction/event checks, multiple-step-size directional comparisons and a fresh evaluation identity. Dropping tiny faces or welding nearby vertices changes the registered entity and would require an exact region-equivalence proof. This transport receipt alone does not certify any transformed or retriangulated input.

To replay, regenerate the M34 D′ operand and compile the W1 checker from the source pin used by [M33](M33_NOODLES_EPRIME.md). In the W2 worktree run:

```bash
python research/NeuralMoments/teacher/m34_binary64_design_transport.py \
  /path/to/Dprime.txt c3633c74bc74c2be488eb22de108454276f28941f0477cfee5837f57db99d622 \
  --gmp-checker /path/to/checker > /tmp/m34-binary64.json
cmp /tmp/m34-binary64.json research/NeuralMoments/teacher/M34_DPRIME_BINARY64_TRANSPORT.json
```
