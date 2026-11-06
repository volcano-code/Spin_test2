# M22-P3 · Heart point-000 operand and construction replay

The H8f prepared request and the M21 pilot request have the same canonical
SHA-256, `7084fecb6bbd4d17bca842bfe68299f9711ecead09eb01593364424a3e0c4a8a`.
The archived point-000 material identifies a subtraction, the frozen 10
coefficients and an indexed source mesh. All 8,455 historical boundary vertices
were reconstructed exactly in physical coordinates and in the M21 target frame.

| Archived construction | Count |
|---|---:|
| Explicit design vertex | 7,625 |
| Explicit quadric-source vertex | 46 |
| Design edge intersected by a quadric-source triangle | 696 |
| Quadric-source edge intersected by a design triangle | 88 |

For each line-plane construction, the replay checks the two original edge
indices, the three opposite triangle indices, each input coordinate, the exact
intersection, the closed segment parameter and containment in the closed triangle.
The result is computed from the inputs before comparing the two archived vertex
coordinates. See [machine-readable hashes and counts](M22_P3_POINT000_OPERAND_REPLAY.json).

W1 can use these operand identities for this specific H8f point when checking
retained patch coverage or detecting a missing patch. The geometry-side
observation concerns a single real input. W2 has not established complete
Boolean region equality, neighborhood validity or an admissible Jacobian.
The H8f and M21 evaluation-map and full material digests still differ.
The M21 producer's original full attachment remains unavailable. Training
remains excluded and this replay launches zero new native calls.

## Reproduce this result

From this directory, use the original H8f ZIP and the extracted M21 pilot
directory containing the original M21_PILOT_SPEC.json and M21_PILOT_LABEL.json.
The program authenticates both against the existing receipts.

```sh
python m22_point0_operand_replay.py --pilot-dir "$M21_PILOT_DIR" --h8f "$H8F_ZIP" > observed.json
cmp M22_P3_POINT000_OPERAND_REPLAY.json observed.json
python -m unittest -q test_m22_point0_operand_replay test_run_native_smoke test_m6_evidence test_m22_recover_h8f_point0 test_m22_audit_heart_label
```

## Capture a fresh native attachment

On a host with the pinned native modules and validated v3 spec, use new paths
under a new run ID. The material-output option reads the actual evaluator cache
from that invocation. After checking the sealed label identity it verifies the
canonical attachment digest and evaluation-map identity, then publishes and
reads back the complete JSON before publishing the label. The two outputs
are exclusive-create. For a failed invocation use the recorded failure label.
This option was exercised with a synthetic producer during testing. The real
H8f replay above does not supply the missing M21 process attachment.

```sh
python run_native_smoke.py --spec "$FROZEN_SPEC" --output "$NEW_RUN/label.json" --material-output "$NEW_RUN/material.json"
```
