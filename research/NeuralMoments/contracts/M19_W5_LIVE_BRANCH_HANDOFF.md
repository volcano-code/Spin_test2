# W0 M19 — live cross-branch W5 adoption detection (no external branch writes)

**Frozen owner window:** W0 `research/neuralmoments-w0-contract-20261019`.
**Dependencies:** W5 `research/neuralmoments-w5-deeponet-20261019`, W2 `research/neuralmoments-w2-teacher-20261019`.
**Scientific status:** real teacher labels / W3 Newton remain blocked. Historical H8f job `39123384402` conclusion was failure.

M18 delivered a COMPLETE v2-aware excluded-only W5 parser proposal at
`migration_proposals/w5_contract_gate_v2_m18.py`, exact Git Blob SHA
`4c97aacc1d50366c7cc97876064f4183469d9910`. W5 still owns the
actual adoption: GitHub [issue #2](https://github.com/volcano-code/Spin_test/issues/2).
Original W5 `contract_gate.py` blob `14b2ab225a032188fe16570414ef321af9a81e00`
is v1-only and uses a `1e-9` native-c check, while W0 requires `1e-12`.

## Actual M19 change

`w5_live_handoff_m19.py` accepts **actual commit SHA + content blob SHA**
from git `rev-parse` and source bytes from `git show` for the W5 and W2
branch refs. It independently recomputes the blob hashes and rejects
missing/mismatched/symlinked data, duplicate JSON keys and unreviewed W2
training claims. A new unreviewed W5 blob is a hard CI failure rather than
implicit acceptance.

* `PENDING_W5_OWNER_ADOPTION`: the live W5 file is the unchanged old Git
  blob. This is a valid **negative** report, never proof of migration.
* `REVIEWED_W5_M18_SOURCE_PRESENT_ON_W5_BRANCH`: exact W5 file bytes
  match the approved M18 proposal. In that case execute source-bound
  *structural* runtime tests: W0 nonunit sample rejection, valid W0 record,
  W2 v2 excluded-only read, and re-signed fake train split denial. This
  does not certify full W5 branch tests or real Cherchi material.
* Other W5 code is `W5_UNREVIEWED_REMOTE_SOURCE_DRIFT`; refuse and
  require explicit W0 re-review and updated test/pins.

The W0 GitHub Actions workflow now fetches the CURRENT W5 and W2 heads
**read-only** at runtime and saves an M19 JSON report with the precise
branch HEADs and source blob identities. Its push trigger is W0-only;
changes exclusively in W5/W2 do **NOT** automatically trigger W0's CI.
Re-run manually or on a new W0 push when W5 reports adoption.

All outputs preserve `training_admitted=false`, no release of native
`r` or `J_D`, no accepted Newton or independently sealed full terminal.

## Reproduction

```bash
# Use the owner/repository of the origin from which both refs are fetched.
source_repo=volcano-code/Spin_test2
git fetch --no-tags --depth=1 origin refs/heads/research/neuralmoments-w5-deeponet-20261019
w5_head="$(git rev-parse FETCH_HEAD)"
w5_blob="$(git rev-parse 'FETCH_HEAD:research/NeuralMoments/models/deeponet/contract_gate.py')"
git show 'FETCH_HEAD:research/NeuralMoments/models/deeponet/contract_gate.py' > /tmp/w5-live.py
git fetch --no-tags --depth=1 origin refs/heads/research/neuralmoments-w2-teacher-20261019
w2_head="$(git rev-parse FETCH_HEAD)"
w2_blob="$(git rev-parse 'FETCH_HEAD:research/NeuralMoments/teacher/teacher_manifest.json')"
git show 'FETCH_HEAD:research/NeuralMoments/teacher/teacher_manifest.json' > /tmp/w2-live.json
python -m research.NeuralMoments.contracts.w5_live_handoff_m19 \
  --source-repository "$source_repo" \
  --w5-source /tmp/w5-live.py --w5-head "$w5_head" --w5-blob "$w5_blob" \
  --w2-manifest /tmp/w2-live.json --w2-head "$w2_head" --w2-blob "$w2_blob" \
  --output /tmp/w0-m19-review.json
```
