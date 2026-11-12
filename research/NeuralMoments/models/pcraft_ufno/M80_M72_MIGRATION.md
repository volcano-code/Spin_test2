# M80 M72 migration and implementation record

Date: 2026-11-12  
Repository: `volcano-code/Spin_test2`  
Branch: `research/m80-pnmo-large-train-20261112`

## Repository policy

Spin_test2 is the only development repository. No separate local clone is required. M72 is the source of the validated native-interval/query/data contracts; its code is selectively ported and adapted here. The separate Spin_test repository is not the project target.

## M72 components retained

- Native RLE columns `(x,y,lo,hi)`, origin, voxel width, side, density, fixed moments and original moments.
- One exact interval descriptor per native RLE interval: physical center, mass, cell extent and raw ten-moment basis.
- M72 ten raw moments and coefficient order, including the factor-of-two convention in quadric cross terms.
- Fixed material moments remain separate from variable editable-domain moments.
- The bounded odd response head:
  `r(q)=r_F + 0.5 r_D + 0.5 a_D tanh(f(q))`,
  with `f` antisymmetrized between `q` and `-q`.
- Positive coefficient-scale invariance, full/empty endpoint certificates, and AD Jacobian from the same response function.
- Geometry/query cache is valid only for the same model instance and parameter versions.
- M72 manifest/audit binding, source-group splits, file SHA-256 verification and `derivative_valid` masking.
- The M72 derivative audit thresholds are not changed by M80.

## P-CRAFT integration in this branch

The new `spin_pino/native_pcraft.py` implements a P-CRAFT-dominant hybrid:

1. The M72 native RLE intervals are kept intact and converted to exact physical interval descriptors.
2. An exact conservative volume aggregation creates a coarse context grid. For the 512 native lattice, the configured 32/64/128 grid sizes divide 512 exactly. This grid is context only, not a replacement for the native geometry.
3. The P-CRAFT UFNO3d encoder processes the four-channel context grid.
4. The final P-CRAFT decoder feature map is sampled at every native interval center. Those spatial features are fused with each interval's exact M72 descriptors.
5. Chunked, checkpointed interval processing performs a mass-weighted geometry reduction.
6. The M72 bounded odd moment head consumes the resulting geometry latent and the native quadric coefficients.
7. The Jacobian is obtained by differentiating that same moment response.

`pino_backbone.py` gains `forward_with_map()`. It adds no parameters; the original `forward()` still returns the original pooled latent and old checkpoint parameter keys are unchanged.

## Training entry point

`spin_pino/native_train.py` reads the M72 formal manifest and a fresh `corpus69.py audit` receipt. It verifies:
- formal/frozen manifest and its fingerprint;
- audit schema, validity, `ready_for_training`, blockers and fingerprint binding;
- train/validation/test source-group isolation;
- geometry and label file hashes;
- query/moment/Jacobian shapes and finiteness on derivative-valid labels;
- minimum query coverage for every qualified record before long training.

The `preflight` command selects the largest qualified training geometry, performs a full moment/Jacobian backward pass, AdamW update, checkpoint save/reload and output-parity check, and reports peak CUDA allocated/reserved memory.

The `train` command defaults to at least 256 queries per qualified record and at least 50/7/10 train/validation/test source groups. It refuses a 32-query corpus for a formal long run unless the caller explicitly lowers the query minimum for an engineering pilot.

## Current presets (candidates, not GPU-certified)

| Preset | P-CRAFT context | P-CRAFT width/depth/modes | Interval width/depth | Purpose |
|---|---:|---:|---:|---|
| large | 32^3 | 16 / 4 / 8 | 96 / 2 | first GPU smoke |
| xl | 64^3 | 32 / 6 / 12 | 192 / 4 | main candidate |
| xxl | 128^3 | 48 / 8 / 16 | 256 / 6 | capacity candidate |

All presets preserve the full native interval branch. These are configuration proposals; actual memory and speed must be measured on the RTX 5090D.

## Explicit non-claims / remaining gates

- The M72 256-query formal corpus is not assumed complete. The existing 67 geometry groups and 32-query banks do not become a 256-query dataset automatically.
- The previously discussed extra finite-difference step for the known M72 derivative failure is not accepted as a fix until reproduced against the exact local M72 source and data. No audit tolerance is relaxed.
- P-CRAFT/M72 query training is not yet a Spin-It Faster Newton acceleration result.
- The general constrained solver and the paper's equality-Newton core remain distinct. The solver bridge, event/active-set handling, adaptive exact correction and final independent Spin design audit are separate M80 gates.
- DensiCrafter and TRELLIS are not silently injected into this training set. TRELLIS assets require real generation receipts and provenance; DensiCrafter candidates require a separate rotation-task teacher and quadric-fit audit.
- No 5090D preflight or multi-day training is claimed by the source/CPU CI alone.

## Current status

- M80 development branch created in Spin_test2.
- P-CRAFT spatial-map port committed.
- M72-native RLE geometry/query model and M72 corpus trainer committed.
- CPU contract CI is running; its result must be checked before local GPU preflight.
- Large-training readiness remains **pending** until CI, local 5090D preflight, formal 256-query corpus audit and checkpoint/resume checks pass.
