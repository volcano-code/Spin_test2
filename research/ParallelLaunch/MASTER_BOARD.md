# Spin_test2 Neural Moments 开发入口

当前对话统一调度 W0–W6，开发与验收均在 `volcano-code/Spin_test2`。见[首月计划](SPIN_TEST2_30_DAY_PLAN.md)、[首轮 Heart 验证](SPIN_TEST2_HEART_POINT0_FIRST.md)和[完整差集验收](SPIN_TEST2_HEART_POINT0_SOLID.md)。

## 迁移前协调记录

**Established:** 2026-10-19. **This board records prepared branches, not running ChatGPT chat sessions or completed experiments.**

## Source identity

- Full source, current Heart baseline: `0bcc187b3005478baa586a6d6f450e73811ca342` (`research/heart-p0-p3-20261017`).
- Teapot P0 baseline: `24f89ed21c251a8678aceb997e4ca30e4ba24125` (`research/spin-it-teapot-20261018`).
- P-CRAFT source: `volcano-code/test` at `24eda3e9b3f87ba16251a9025761986d6cfe3079`.
- Preserve `volcano-code/Spin`, `Spin_test/main`, historical CI failures and materials.

## Windows and startup instructions

| Window | Role | Branch / exact instructions | Dependencies | Earliest evidence target | State |
|---|---|---|---|---|---|
| W0 | 数学合同与总控 | [research/neuralmoments-w0-contract-20261019](https://github.com/volcano-code/Spin_test/blob/research/neuralmoments-w0-contract-20261019/research/ParallelLaunch/W0_START_HERE.md) | 无 | MomentContractV1＋质量矩坐标测试 | `M19_LIVE_W5_W2_GIT_FETCH; 157_ACTUAL_CI_PASS; W5_PENDING_OWNER_ADOPTION; TRAINING_DENIED` |
| W1 | Cherchi/Faster 性能 | [research/neuralmoments-w1-native-20261019](https://github.com/volcano-code/Spin_test/blob/research/neuralmoments-w1-native-20261019/research/ParallelLaunch/W1_START_HERE.md) | 可独立 profiling；合并需 W0/W2 | 冻结同输入逐段耗时＋原生改造对照 | `RECENT_COMMITS_OBSERVED; SCIENTIFIC_STATUS_NOT_INDEPENDENTLY_CERTIFIED` |
| W2 | 精确教师与标签 | [research/neuralmoments-w2-teacher-20261019](https://github.com/volcano-code/Spin_test/blob/research/neuralmoments-w2-teacher-20261019/research/ParallelLaunch/W2_START_HERE.md) | 正式发布标签需 W0 | 真实 r,J＋SHA＋可微/失败状态 | `W2_M12_GIT_LEDGER_81_RAYS_17_UNIT_64_NONUNIT; ZERO_TRAINING_RELEASE_OBSERVED` |
| W3 | Heart H9 | [research/neuralmoments-w3-heart-h9-20261019](https://github.com/volcano-code/Spin_test/blob/research/neuralmoments-w3-heart-h9-20261019/research/ParallelLaunch/W3_START_HERE.md) | 无 | 81 点 point60 构造变化诊断 | `W3_STAGE12_FORENSIC_COMMIT; LOCAL_90_TESTS_REPORTED; NEW_NATIVE_SIDES_0_OF_4; NEWTON_DENIED` |
| W4 | Teapot P1 | [research/neuralmoments-w4-teapot-p1-20261019](https://github.com/volcano-code/Spin_test/blob/research/neuralmoments-w4-teapot-p1-20261019/research/ParallelLaunch/W4_START_HERE.md) | 已完成 P0；体积/求解需 W0/W2 | 合法茶壶输入、F/D 与初态 | `RECENT_COMMITS_OBSERVED; SCIENTIFIC_STATUS_NOT_INDEPENDENTLY_CERTIFIED` |
| W5 | DeepONet＋MLP | [research/neuralmoments-w5-deeponet-20261019](https://github.com/volcano-code/Spin_test/blob/research/neuralmoments-w5-deeponet-20261019/research/ParallelLaunch/W5_START_HERE.md) | 解析 smoke 立即；真实训练需 W0/W2 | r,J 模型＋导数监督/成本报告 | `RECENT_COMMITS_OBSERVED; SCIENTIFIC_STATUS_NOT_INDEPENDENTLY_CERTIFIED` |
| W6 | P-CRAFT UFNO | [research/neuralmoments-w6-pcraft-20261019](https://github.com/volcano-code/Spin_test/blob/research/neuralmoments-w6-pcraft-20261019/research/ParallelLaunch/W6_START_HERE.md) | 解析 smoke 立即；真实训练需 W0/W2 | U-FNO 几何编码缓存＋r,J head | `RECENT_COMMITS_OBSERVED; SCIENTIFIC_STATUS_NOT_INDEPENDENTLY_CERTIFIED` |

## Merge/review gates

1. W0 publishes `MomentContractV1`, including native coefficient order and cross-term factor-two mapping. No published teacher/NN experiment can silently change it.
2. W2 publishes immutable real Cherchi/GMP `r,J` labels with native/source identity and explicit validity categories. Training may use analytic smoke data before this.
3. W5/W6 benchmark on equal verified frozen labels and geometry-family splits, report geometry encoding separately from hot parameter-head inference and Jacobian.
4. W1 shares same-call baselines, never merges validation-skipping shortcuts, and reports end-to-end wall time and raw geometry/moment invariance.
5. W3 retains H8f failure at 1/80 branch-feature classification; no converged Heart model before real Newton terminal proof.
6. W4 retains public teapot original input. 19 indexed components and 160 geometric open edges require independently validated repair.
7. W0 alone coordinates integration/cherry-pick via reviewed commits. No other window edits this board; work progresses only when a connected new chat or agent actually executes tools.

## Resource and evidence policy

- Full native jobs serialized in same environment: 24GiB RAM, swap0, 60min/job, compile ≤4 jobs; coordinate GPU usage.
- Every milestone: commit SHA, changed files, tests/CI, measured results, failure ledger, model and data identities.
- Network prediction/physical constraints are not a geometry validity certificate. Final material and raw residual require true Cherchi/Faster audit.



## W0 milestone 1 record (2026-10-19)

- W0 math contract and reference implementation pushed exclusively to `research/neuralmoments-w0-contract-20261019` under `research/NeuralMoments/contracts/`.
- Native c order `a,bx,by,bz,Axx,Ayy,Azz,Ayz,Axz,Axy`; ordinary r order `M,LX,LY,LZ,QXX,QYY,QZZ,QYZ,QXZ,QXY`; map `q=B*c` with last three B diagonal entries equal to 2.
- Local analytic tests: 15 passed from independent, more extensive W0 working implementation. GitHub CI and run of exact committed files are not claimed; frozen Cherchi production geometry not run.
- W1–W6 completion and CI statuses not independently verified; preserve previous statuses until per-window evidence is reviewed.
- Required integration gates: independent exact material seals, true geometry and derivative identity, and terminal Newton validation, none replaced by P0 analytic smoke.


## W0 milestone 2 record (2026-10-19)

- Implementation `6eb8d16ccec11ef3de283f239facd6a9bc183a04`: strict source/shape/math consistency validator, W2 pre-release identity adapter, and adversarial tests; original native `moment_coordinates.py` mixed-boolean coercion fixed.
- Local test command `python research/NeuralMoments/contracts/run_contract_smoke.py` passes 57/57 (0.13 seconds); all published source files were checked byte-for-byte by Git blob SHA before pushing, runner blob verified separately. See [W0 M2 evidence](W0_M2_EVIDENCE_20261019.md).
- Actual Cherchi/Eigen runtime, exact Boolean material certification, derivative smooth-neighborhood proof, end-to-end Newton terminal and GitHub Actions result are **NOT VERIFIED**, so teacher and network training admission remain blocked.
- The current W2 manifest (read-only observation) reports adapter-smoke only and zero native teacher/training samples. Other W rows are unchanged pending independent audit.


## W0 milestone 3 source-linked native / exact P0 observation (2026-10-19)

- The unchanged frozen production `src/spin/QuadricCoefficients.cpp` and header are Git-blob verified as `ae831afe9191f67fb2659f112d1b85a8e2b61581` and `d315960a8d9616197df656a3c74b74b2da037798`, compiled together with W0 native probe and a linker stub **only** for unrelated slicer SHA hashing.
- 125 actual native-Eigen points passed (value, gradient, homogeneous matrix, pullback; worst absolute error `6.66e-16`). Signed exact-rational 24-face fixed shell F plus 12-face inner D was independently integrated against separable analytic box mass-moments and transformed meshes, with **exact zero** algebraic difference. New M3 tests: `9 passed in 11.85s`.
- Source/test files committed under W0-only scope; see [W0 M3 evidence](W0_M3_EVIDENCE_20261019.md) and `research/NeuralMoments/contracts/contract_smoke_m3.json`.
- Machine has only 5.8GiB RAM, no swap, and neither `_make_it_stand` nor `_cherchi_core_boolean` present. **Real Cherchi material, independent exact Boolean seal, teacher r/J, Newton terminal and training admission NOT VERIFIED.** GitHub CI is not assumed green.
- The following W1–W6 branch SHAs and commit subjects were **read-only observed**, not independently benchmarked or certified. The earlier state cells remain unchanged to avoid treating commit messages as tested evidence:
  - W1: `01a1d480ec4044ff678bc648051234bcb7e85aeb` — research(w1): M2 exact GMP PreparedMaterial parity and H8 replay profiler
  - W2: `14b1d2d5da4e258d125e664b5665ef84f7ebbb4e` — W2: add mock smoke for one-call Cherchi adapter and failure accounting
  - W3: `0f5229bb10cac098011f42ab9c996eec485ac034` — Heart H9 W3: certify archived edge-event sign bracket, add guarded native microprobe and tests
  - W4: `8fc1dd33e6cabb29a54849dce271ec63660b6af4` — W4 P2: reproducible CGAL teapot solid union with independent self-contact and moment gates
  - W5: `7054cf6624eeb85f8ba06093090127bdb8016dc4` — W5 M2: gate W0/W2 teacher provenance and audit multi-seed analytic baselines
  - W6: `694e5764e656de0c769f59d2838dc53eddde0b63` — W6 M2: W0-native analytic cut oracle, derivative smoke, W2 fail-closed gate and CPU evidence


## W0 M4 native replay gate and read-only branch heads (2026-10-19)

- Implemented real-source Cherchi single-call replay, independently exact rational mass moments, frozen 6-way input/native/source SHA pins, resource-guarded native worker and 43 adversarial pytest cases. See [M4 evidence](W0_M4_EVIDENCE_20261019.md).
- Local M4 tests 43/43 passed; on the actual 5.8085-GiB/swap0 host the native admission command returned exit 3, `BLOCKED_INSUFFICIENT_PHYSICAL_RAM`; no real Cherchi teacher evaluation, exact Nef replay, derivative-neighborhood certificate or training labels were produced.
- W2 manifest remains `ADAPTER_SMOKE_ONLY`, `native_teacher_runs=0`, `training_samples=0`. Its pre-W0 label envelope omits exact rational material vertices/faces, so a genuine live `evaluator.material(c)` attachment is required for W0 numeric replay.
- Read-only branch HEAD snapshots (commit presence **not** proof of runtime correctness):
  - W0: `0c3ad0a8f6a6f8088ecb617c980d979fafe68383`
  - W1: `bbee1eb3044bf41bb5328f0348e97334d8faf0a5`
  - W2: `56212f46b007fa255cc5796224192010aa67513c`
  - W3: `fc08c229ecf966f9a4453414dbb54095d22cc118`
  - W4: `dcae39d2113aeed91c26d7bacceba3e252306d27`
  - W5: `c535cbde306a6ec4447e4d60583fbbe997ac26f5`
  - W6: `b5d3985ec4a10c74316bda6579767b58c13b4eaa`
- Earlier W1–W6 status columns are retained pending formal per-window test evidence review. Native jobs must remain serialized (24 GiB/swap0; compile ≤4; ≤60min/job).


## W0 M5 cross-window independent reference and tangent diagnostic (2026-10-19)

- **Read-only W2 snapshot:** W2 HEAD observed `c2628ea75a8680910557855896e7317f92567756`; exact `independent_reference.py` Git blob `d734398292e1fdaf839db3413c537de2474df3e1` and SHA256 `3238fb14d4cbf85d7754551689fa40cbb08653bf3a9bf631faf16138f9c6bea9`. This snapshot is copied ONLY into W0's contract directory; W2 branch is not modified.
- **Measured:** 93/93 local tests passed (`python -m pytest -q research/NeuralMoments/contracts/test_*.py`, 0.20 sec): 43 M4 regression tests plus 50 M5 cases. New W0/W2 exact rational r10 equality includes nonzero XY/XZ/YZ and fixed-shell+design accounting; also detects shared-vertex pinched nonmanifold that the W2 edge-only reference admits. Analytic sphere-cap tangent Richardson check maximum normalized error `4.8248891104663016e-08` within predeclared P0 `2e-7` tolerance, and deliberately vetoes base-versus-probe branch changes.
- **Native:** Real M5 CLI preflight on this 5.8085 GiB / swap0 container exited 3 with `BLOCKED_INSUFFICIENT_PHYSICAL_RAM`, no Cherchi, no teacher data; manifest still `native_teacher_runs=0`, `training_samples=0`. Full exact-region Boolean partition, stable derivative neighborhood and Newton terminal remain **NOT VERIFIED**.
- **Evidence:** [W0 M5 record](W0_M5_EVIDENCE_20261019.md) and `research/NeuralMoments/contracts/contract_smoke_m5.json`. No 1% universal threshold and no green CI claim. Other windows were read-only.


## W0 M6 exact full partition contract gate (2026-10-19)

- Own branch only: M6 code and source identity in `research/NeuralMoments/contracts/`; see [W0 M6 native handoff](../NeuralMoments/contracts/M6_NATIVE_HANDOFF.md) and [executed M6 evidence](W0_M6_EVIDENCE_20261019.md).
- **Published but NOT native-executed:** one pinned actual Cherchi quadric source, retained/opposite-side Boolean, exact inverse of binary64 target frame, independent native `check_exact_material_partition` for original = F+D, D = retained+cavity, original = F+retained+cavity, exact ten-moment identity and fixed-density accounting. Output remains `training_admissible=false` even if future native partition succeeds.
- **Observed local test execution:** `142 passed in 0.27s`, local M4/M5/M6 mirror only; 49 new M6 tests plus 93 rerun M4/M5 tests, not full W0 M1–M3 checkout. Synthetic native test doubles are not production proof.
- **Observed native preflight:** actual RAM 5.808525085449219 GiB, swap 0; exit 3 `BLOCKED_INSUFFICIENT_PHYSICAL_RAM`; no Cherchi call. W2's last inspected manifest reports 0 native runs and 0 training labels.
- No Green GitHub Actions result claimed, no shared native serialization lock with W1/W2, and no automatic derivative neighborhood/Newton terminal/printability certification. Historic failures are preserved and M6 CLI never overwrites an output evidence path.


## W0 M7 authentic-terminal-readiness and independent KKT (2026-10-19)

- Source-linked M7 terminal reader, original Spin2014 KKT re-evaluator, M6 proof-chain binder and production-only CLI committed on the **W0 branch exclusively**. It reads existing output; neither launches Newton nor performs a Boolean. See [M7 evidence](W0_M7_EVIDENCE_20261019.md) and [M7 handoff](../NeuralMoments/contracts/M7_TERMINAL_HANDOFF.md).
- Measured new M4–M7 local mirror: **222 passed in 0.55 s** (142 prior M4–M6 rerun, 80 M7 tests including 5 multi-step journal tests). All deployed source/test Git blobs match the local tested bytes. M1–M3's source-linked binary tests were not rerun in this isolated mirror.
- M7 requires independent **post-run** SHA256 pins for every `MAP_SAVED` `evaluation-%06d.json`, every essential terminal/complete/material journal attachment, the last accepted Newton c, and the native source/geometry identity. An independently *pre-run* frozen model/build/source SHA remains a different class of pin. Do not allow local self-resealed evidence as native authority.
- Synthetic journal consistent numerical smoke: exit0 `NUMERICAL_ARCHIVE_CONSISTENT_NOT_NATIVE_CERTIFICATE`, never training true; actual installed production journal-verifier and original-objective KKT requests: **exit3 `BLOCKED_PRODUCTION_VERIFIER_UNAVAILABLE`**. No real Cherchi solver or independent M6 native partition was run on this 5.8-GiB/swap0 host.
- **W2 actual native labels and training admission remain BLOCKED**; an apparent CONVERGED JSON, archive hash equality, several stable sampled branches, or an ML prediction cannot substitute for independently authenticated exact geometry, derivative smooth-neighborhood, sealed final material and native original-equation terminal.


## W0 M8 real H8e archived Cherchi replay (2026-10-19)

- GitHub native workflow 39121571756: completed FAILURE (unmodified historical result); authentic ZIP artifact 12134772056 digest `14de425c18ac8b4704424976983d7130923432cd67a20f314e65c7f48f85ad38`.
- GitHub independent classification workflow 39122997405: completed SUCCESS (only finite-point classification); ZIP artifact 12134499573 digest `e17d90a8195637358d19e61fda3d8e1946cb33540f26190cead6723e7a36ef16`.
- W0 M8 downloaded both real ZIPs, verified artifact SHA-256, original frozen request SHA `1d2b06063c5b2c7010fcb2764518c05c9707f013d6914bc4e20298f8516e061c`, original report SHA `7937da1e76a73612d71bc0d4352c292c2549655c770d024a06548eb8e8096d0f`, classifier report source-binding and all 25 per-point gzip SHA-256.
- Real 25 archived Cherchi materials (8455 vertices and 16910 oriented triangles per point) independently re-integrated using rational signed tetrahedra, all 250 raw moment component comparisons pass with binary64 absolute maximum 0; target-frame fixed material max discrepancy 2.220446049250313e-16. This is an independent offline arithmetic recheck, not a new native Cherchi solve.
- Nine 5-point FD stencils recomputed (max normalized difference 2.6019151368622826e-10), three stencils contain original raw feature-hash drift, eight perturbed source feature hashes differ. Independent finite-point classification 24/24 `SYMBOLIC_PATCH_AND_TRIANGLE_IDENTITY_STABLE_NOT_CERTIFIED` does NOT grant smooth neighborhood.
- Scientific status: non-author reconstructed H8 CAD only; original native job failure preserved; Newton terminal, continuous derivative neighborhood, exact full original/F/D/cavity region partition, dynamics and printability NOT authenticated. No real W5/W6 training labels authorized.

- Reproduction: `research/NeuralMoments/contracts/real_h8e_archive_m8.py` plus `test_real_h8e_archive_m8.py`; 15 local fast regressions passed with both REAL ZIPs supplied, authentic 25-point exact reintegration wall 76.19s. See [M8 evidence](W0_M8_EVIDENCE_20261019.md) and `research/NeuralMoments/contracts/contract_smoke_m8.json`. W0 changed only its own two directories.


## W0 M9: real 81-point H8f Cherchi archive & point 60 audit (2026-10-19)

- Real historical GitHub Actions [39123384402](https://github.com/volcano-code/Spin_test/actions/runs/39123384402) **failed** at exact finite-point construction classification after 81/81 native material evaluations and 30/30 numerical FD stencils passed. Original failure is immutable; `complete_eighty_one_samples=false` and `formal_newton_converged=false` remain required.
- W0 independently reloaded and hash-verified original artifact `12135528290` (SHA256 `99a050f4604109e0323c0110a029aa5d7601d729d755e6956ad48ed8c77ccda9`), 81 same-call materials, oriented-closure checks, 81×10 Fraction signed tetrahedral raw moments and 30 finite-difference stencils. **All 81 exact moment binary64 comparisons had max abs error 0**; fixed-frame discrepancy 2.22e-16, maximum FD scaled error 9.45e-10. Full 4-worker exact rerun about 113.34 sec.
- Normalizing mathematically equivalent LPI operand order and triangle cyclic starts leaves **one genuine sampled combinatorial difference**: point 60 at c[7]=+3e-6 versus stable +1.5e-6, 5 modified vertex-position source features, 2 removed/2 new source features, 17 removed/17 new oriented signature triangles; source quadric triangulation unchanged. No continuous event root, smooth-neighborhood or exact Boolean-set topological theorem claimed.
- M8+M9 local real ZIP pytest **38 passed** in 28.37 sec. M9 source and tests committed W0 scope only; W1–W6 branches untouched. See [M9 evidence](W0_M9_EVIDENCE_20261019.md) and `research/NeuralMoments/contracts/contract_smoke_m9.json`. Real W5/W6 teacher label admission remains blocked.


## W0 M10: Real H8f endpoints and W3 event handoff (2026-10-20)

- W0 reads the original SHA-pinned, failed H8f run 39123384402 artifact 12135528290, computes exact-rational determinants for real source edge (24,90) against frozen design edge (5517,5518). Signs are +1.5630561011148573e-08 (point59, c7+1.5e-6) and −1.4405585247826644e-08 (point60, c7+3e-6). Both closest infinite-line points lie in segment interiors and squared distances at endpoints remain >0. Point60 structural anomaly persists (5 vertex positions, 2 feature substitutions, 17 removed /17 added symbolic oriented triangles).
- Archived c7 moment secant vs average Jc column7 deviation max 6.11608878342898e-10; no continuous native event theorem or differentiability follows. The W3 fixed-normal analytic surrogate root 2.28058622834443e-06 is a **conditional estimate**, not an observed Cherchi root. Original CI failure is unchanged; Newton/training labels remain BLOCKED.
- Local regression (M8/M9/M10 with original archives) **71/71 PASS in 27.28s**. Original immutable W3 certificate Git Blob `213d0d7f7bfec6ffa293cfd735556d8b10316601` is snapshotted under W0 for independent exact endpoint validation, never modified in W3. See [M10 evidence](W0_M10_EVIDENCE_20261020.md), [handoff](../NeuralMoments/contracts/M10_H9_W3_EVENT_HANDOFF.md), `contract_smoke_m10.json`.


## W0 M11 four-side readiness and independent W3 evidence gate (2026-10-20)

- Implementation under W0-only `research/NeuralMoments/contracts/`: `w3_four_side_replay_m11.py`, 66 new regression tests, stage9 read-only observation, and a fail-closed machine-readable blocker. No W3 branch/source or historical Actions artifacts changed.
- W3 stage9 evidence, frozen git blob `ec6995c2b60cb5158d94f79a783a60d9ad64f23d`, originally observed HEAD `533f2f78e4b9e62920a8e667edc3cce9924bff8e`: zero new native Cherchi calls, zero new four-side material packets, Newton not admitted. W3 branch HEAD at this W0 push: `533f2f78e4b9e62920a8e667edc3cce9924bff8e`; W0 has not inferred new source facts solely from that HEAD.
- Actual tests: 137/137 Python pytest cases passed in 26.14 seconds (M8–M11 local mirror, genuine pinned H8e/H8f historical ZIPs), separate real W3 Stage9 CLI exit 3 `BLOCKED_MISSING_FOUR_NEW_CHERCHI_SIDE_MATERIALS`. These are no substitute for **four new** Cherchi native results.
- Future W3 four-packet contract independently rechecks 13 postrun file SHA256 receipts, three independently precommitted binary SHA256 values, frozen four native c samples, true rational r10 and exact source/geometry identities, actual edge signs and 10×10 Jc column7, and left/right finite differences. Even a passing finite test does not certify open smooth stratum, full Boolean partition, Newton terminal or W5/W6 training admission.
- See [M11 evidence](W0_M11_EVIDENCE_20261020.md), [M11 handoff](../NeuralMoments/contracts/M11_W3_FOUR_SIDE_HANDOFF.md), and `research/NeuralMoments/contracts/contract_smoke_m11.json`. No GitHub CI green claimed.


## W0 M12 — Stage10 original identity and independent 13-file custody (2026-10-20)

- W0 reviewed the **actual W3 Stage10** HEAD `324256786bc189abbbb93452ed2627cfe0b28a13`. Pinned original Git Blob SHA1 for Stage10 execution `254e93076d6fbec0a60632cee4c4731945f579d4` and Stage10 frozen input `8a352efca5f7f88dda24c359b7d5225f10f9cf66`. Input gate has four planned proxy c[7] values, but actual output records **zero** new Boolean evaluations and **zero** four-side materials. W3's own reported local 86/86 tests are read-only metadata, **not** W0 retested W3 source or successful native execution.
- New `w3_custody_m12.py` enforces externally SHA-frozen pre-run source/binary identities and post-run 13 independent file digests, rejects symlinks/in-read modifications/corrupt gzip, and delegates genuine future geometry/Jacobian replay to M11. No automatic external runtime attestation or Newton/training approval. The frozen W3 head must be explicitly reviewed again after W3 source updates.
- **Actually executed:** M8–M12 joint `178 passed in 28.60s` using genuine prior H8e/H8f GitHub archive ZIPs; 41 new M12 cases include synthetic negative fixtures and original W3 Stage10 blob readings. Local Stage10 CLI returned **exit 3** `BLOCKED_MISSING_FOUR_NEW_CHERCHI_SIDE_MATERIALS`. Current cgroup 4 GiB, swap0; no new native solve performed.
- See [W0 M12 evidence](W0_M12_EVIDENCE_20261020.md), [M12 handoff](../NeuralMoments/contracts/M12_NATIVE_CUSTODY_HANDOFF.md), `contract_smoke_m12.json`. Original H8f run `39123384402` remains `failure`; W5/W6 real Cherchi label admission stays blocked. GitHub CI for new W0 commit must be separately verified.


## W0 M13 — immutable native c vs signed positive ray, W3 Stage11 drift (2026-10-20)

- Independently reopened the authentic H8f 81-point source archive (GitHub Actions 39123384402, original conclusion **failure**) and verified every saved coefficient/raw material binding. 17 original native c satisfy W0 V1 unit norm (±1e-12), 64 do not, maximum norm deviation 2.282620930493806e-6. The original signed binary64 c and native Jc are **never rewritten**; point60 c[7]=3e-6 and polynomial q[7]=6e-6 under B. A computed positive-ray display vector is nonrelease diagnostic only.
- Read-only W3 Stage11 HEAD 8322f381ee23f26ef6be0378bba7d4cececbb60e introduced an early unverified-* packet capture (source blob 22a62f46a458452186126ae571b50a44212197aa) and corrected full q metadata (probe blob 083ea0075d9df7d71f97e77591a18f5453691665). It still reported **zero** fresh Cherchi material and no Newton. The old W3 Stage10 source lock is not transferable without a fresh independently frozen build identity.
- Local M8–M13 tests completed in two batches: **210 passed, 2 deselected (21.54s)**, and **2 real original H8f coefficient tests passed (23.21s)**; **212 distinct tests** were executed successfully. An earlier monolithic timeout is excluded. New M13 tests 34/34 passed. No new production Cherchi job or W5/W6 training admission. Original MomentContractV1 remains unchanged.
- See [M13 evidence](W0_M13_EVIDENCE_20261020.md), [M13 handoff](../NeuralMoments/contracts/M13_RAW_COEFFICIENT_AND_W3_STAGE11_HANDOFF.md), and [M13 smoke](../NeuralMoments/contracts/contract_smoke_m13.json). The full 81-row archival raw-c ledger is reproducibly generated by archived_raw_c_m13.py and retained in the downloadable local research bundle rather than checked into Git. No green CI assumed.


## W0 M14 — audited W2 exact signed rays without re-labeling native coefficients (2026-10-20)

- Compared actual W2 M12 full Git blob `2d4cf4aae1368e22e760803d9bfe49479125ac4b` (source SHA256 `255769c4401a89449ef864bf0ef72ec99403d0085b5eb04891512dfe2daab473`) with W0 M13 saved genuine native H8f original c and all 81 material gzip digests. GitHub connector independent exact BigInt-rational check: **81/81 matching** archived c, signed positive rational tokens, material gzips, q=Bc and unit flags. No new native samples. Official full W2 JSON not cloned into local Python, but the full 81-row crosswalk compact Git blob `051a9bb7149ff63af92d7018019165face426f52` independently reproduces in Python from the authentic H8f ZIP.
- Exactly **17 unit-compatible, 64 nonunit** original c. 81 distinct signed exact positive rays; float64 display's exact rational ray agrees for 17/17 originally unit, **0/64** originally nonunit. Point 60 retains original c[7]=3e-6 and monomial q[7]=6e-6; no silent normalization or Jacobian reparameterization. W0 MomentContractV1 and its source/schema unchanged.
- New **diagnostic-only** schema/code/CLI, never a new native label. Local actual tests completed in three separate successful runs: M8–M12 `178 passed in 31.92s`, M13 `34 passed in 28.83s`, M14 `27 passed in 14.56s`; **239/239 distinct tests**. Earlier combined timeout excluded. CLI separately re-opened the real pinned 81-point H8f ZIP and produced 81-row `contract_smoke_m14.json` with `training_admissible=false`.
- Read-only W3 Stage12 at `d69a9c91bf3a9359cf2a480fc22fec8bc8393ca3` reports no new native side samples or Newton; its source-lock must be regenerated for every future changed W3 checkout. Original historical H8f Actions `39123384402` remains failure, no claim of W0 GitHub CI green. Review [M14 evidence](W0_M14_EVIDENCE_20261020.md), [signed-ray contract](../NeuralMoments/contracts/M14_W2_SIGNED_RAY_HANDOFF.md) and [M14 JSON](../NeuralMoments/contracts/contract_smoke_m14.json).


## W0 M15 integration (2026-10-20)

- Seven branch SHA observations frozen in research/NeuralMoments/contracts/m15_observation.json. A commit subject alone does not certify tests or scientific outputs; W1/W4/W5/W6 moved beyond their old, stale status cells.
- cross_window_release_m15.py is a fail-closed source-bound training gate: historical H8f 81 sample failure and 17/64 coefficient distribution stay recorded, W2 has no certified released J_D, W3 has zero new four-side samples and no accepted Newton terminal.
- M15_CI_WORKFLOW_TEMPLATE.yml is NOT ACTIVE: the .github/workflows/ directory is outside W0 write scope, requiring separately approved workflow activation.
- Local 18/18 tests validate the precursor implementation; the submitted refactor still requires an exact committed-source checkout/CI test. No GitHub CI green or native certificate is claimed.


## W0 M16 — actual GitHub Actions success and frozen interfaces (2026-10-20)

- User explicitly approved a narrow W0-branch-only CI activation outside the original exclusive directories: new `.github/workflows/w0-contract-ci.yml` only. No main or W1–W6 edits. Initial code/workflow commit `d472b3fcc295e10dbba86d83169dbc149f846bbb` activated a real push workflow.
- **Authenticated actual GitHub Actions:** [run 39450767913](https://github.com/volcano-code/Spin_test/actions/runs/39450767913), success, W0 commit d472b3..., [upload artifact 12278446140](https://github.com/volcano-code/Spin_test/actions/runs/39450767913/artifacts/12278446140), SHA-256 `02b3443483e19dfb84a86428a2351a4237cf73299b73ce2e41930299073ee03d`. Actual CI log: **89 passed in 0.30s**, no compiled Cherchi, no native teacher/newton or full cross-window training. The exact committed-source CI proved that the M15 submitted code and new M16 code/test operate together.
- W2 manifest v2 (source Git blob `41e7c4dc65a029a3cc7ce746c694bc8a03cdad70`) has zero native teacher runs, zero training records and 81 excluded H8f historical observations. W5 `contract_gate.py` (blob `14b2ab225a032188fe16570414ef321af9a81e00`) still expects v1 W2 envelope and tolerates `1e-9` in W0 native-c norm, weaker than W0 `1e-12`. W6 `teacher_gate.py` (blob `24dafd66dce83d3076d4133cae8fdeff34671dae`) hard-denies unauthenticated native import. Frozen source snapshots are observations, **not live external-branch head certification**.
- W5 should migrate to an explicitly versioned v2 adapter under W5 ownership with independent source, physical and derivative admission; W6 must keep real-teacher import blocked until a separately audited W0/W2 release. W3 remains without true four-side Cherchi or converged Newton acceptance. No real W5/W6 teacher labels published or authorized.
- The workflow is truly **active on W0 branch push** and may be manually dispatched if authorized. It is only a structural/mathematical regression: no H8f original failed CI is rewritten. See [M16 evidence](W0_M16_EVIDENCE_20261020.md) and [M16 handoff](../NeuralMoments/contracts/M16_CI_AND_CROSSWINDOW_HANDOFF.md).


## W0 M17 — real W5 consumer version/normalization runtime review (2026-10-21)

- **Actual W0 GitHub Actions run [39459023309](https://github.com/volcano-code/Spin_test/actions/runs/39459023309), success at commit `e20de2d5b3450d68e7908bbcb809c9b6f9b92971`**: 119/119 non-native math/consumer tests passed in 0.41 seconds, no skips. Both existing frozen-source audit and new pinned W5-vs-W0 runtime counterexample steps succeeded. CI artifact 12282041315 ZIP SHA256 `747b437f64c7a31ff9904f7dae975c2bf64cc6699772f19cf45a75b403e9c588`, downloaded and independently inspected.
- Actual W5 old `check_w0_contract` at source Git blob `14b2ab225a032188fe16570414ef321af9a81e00` accepts a **synthetic** W0 c norm drift `4e-10` while W0's `validate_record` at `fe5f71e7cb2b6c6ad45f48c49ab4042f797a113e` rejects. This is a structural compatibility bug, **not evidence of actual W5 prohibited training**. W5 still expects W2 v1 while W2 manifest is v2 with zero real teacher labels and all H8f 81 excluded.
- New `consumer_v2_preview_m17.py` is a W0-only, external-identity-pinned **excluded-v2 diagnostic reference**; no v2 W5 branch code was changed. It refuses split promotion, non-null `J_D`, raw coefficient rewrites, wrong `q=Bc`, source/geometry SHA spoofing and invalid moments. Always `training_admitted=false`.
- Local isolated M17 tests were 29 passed/1 skipped (local W5 source unavailable); the skipped actual W5 import test ran successfully in the complete CI checkout. The underlying H8f original run 39123384402 remains failure. No new W3 native four-side materials, certified smooth neighborhood or Newton terminal; real-teacher release stays blocked. See [M17 evidence](W0_M17_EVIDENCE_20261021.md), [handoff](../NeuralMoments/contracts/M17_W5_V2_MIGRATION_HANDOFF.md), [M17 smoke](../NeuralMoments/contracts/contract_smoke_m17.json).


## W0 M18 — W5 v2 candidate, real CI, W5 adoption still pending (2026-10-21)

- Frozen W5 v1 parser has 1e-9 tolerance; W0 contract is 1e-12. W2 manifest is v2, has 81 excluded historical candidates and no qualified training J_D.
- W0-owned complete proposed W5 replacement `research/NeuralMoments/contracts/migration_proposals/w5_contract_gate_v2_m18.py` corrects both norm gates and adds explicit excluded-only W2 v2 inspection; no release of native r/J. Git Blob `4c97aacc1d50366c7cc97876064f4183469d9910`.
- First CI `39548751329` failed (126 passed / 14 test-fixture exception-class mismatches); repaired test fixture and added class-identity regression. Follow-up actual GitHub Actions [39548862852](https://github.com/volcano-code/Spin_test/actions/runs/39548862852) **141/141 PASS**, no skips, at tested commit `654ba8166e2046f28138aef159de4369dab17f82`.
- Real W5 branch SHA still `219cc548fd5f37a1f88b666242cdea92d8e408cd`, NOT modified by W0. Formal W5 migration handoff [issue #2](https://github.com/volcano-code/Spin_test/issues/2); W5 must apply and run its own tests before W0 can repin and independently approve. See [M18 evidence](W0_M18_EVIDENCE_20261021.md).
- Historical H8f failure, W3 four native side absence, formal W2 teacher release absence and Newton denial unchanged. No claim of scientific proof from Python CI.


## W0 M19 — live W5/W2 adoption monitor and real CI (2026-10-21)

- GitHub Actions on the W0 branch now uses read-only **live `git fetch`** of W5 `research/neuralmoments-w5-deeponet-20261019` and W2 `research/neuralmoments-w2-teacher-20261019`; the verifier binds the remote commit SHA, source Git blob SHA and local `git show` bytes, with strict JSON parsing.
- Current actual W5 HEAD `219cc548fd5f37a1f88b666242cdea92d8e408cd` still has the original v1-only `contract_gate.py` blob `14b2ab225a032188fe16570414ef321af9a81e00`. The live W0 result was **PENDING_W5_OWNER_ADOPTION**, not adopted. The M18 candidate Git blob to adopt in W5 is `4c97aacc1d50366c7cc97876064f4183469d9910`.
- Current W2 live HEAD `ff3abf2b88518045aedb3f2e95fb5b34dc0d586f` has v2 manifest blob `fd2b2d8c386b5f179653cd71d155dfbfa0cbfeac` and no released real training or Jacobian labels; all 81 H8f archived candidates remain excluded.
- First actual M19 CI [run #39577087692](https://github.com/volcano-code/Spin_test/actions/runs/39577087692) succeeded, **157/157 tests** and all source/live audit/artifact upload steps passed; no skipped tests. Local W0 M19 isolated unit tests separately recorded `14 passed, 2 skipped` because original W5/W2 frozen source files only existed on full checkout, and were executed in CI.
- The live auditor does not execute unknown W5 source: old W5 is an explicitly reported expected dependency, the exact byte-reviewed M18 candidate receives only a source-bound structural smoke, and any third unreviewed W5 Git blob (or W2 claimed release) causes a hard verification failure. W0 CI on `push` does NOT automatically trigger on W5/W2 branch changes; re-run after W5 adoption and inspect artifacts before updating status.
- Handed back to W5 owner in [Issue #2](https://github.com/volcano-code/Spin_test/issues/2#issuecomment-6312667068). Original H8f failed CI, W3 four-native-side and Newton truth, and W2/W5/W6 real teacher releases remain denied. See [M19 handoff](../NeuralMoments/contracts/M19_W5_LIVE_BRANCH_HANDOFF.md).
