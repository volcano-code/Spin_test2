# M80 Dataset V2 Specification

## 1. Existing baseline

The current \`combined70-v2\` baseline has 104 registered records, 67 qualified records and a 50/7/10 train/validation/test split by source group. The previous local summary reports a 32-query bank for the baseline; re-open every NPZ and re-run the manifest/audit before treating this count as final. The current M80 formal trainer requires at least 256 queries per qualified record by default.

The 67 objects are the first controlled geometry cohort, not sufficient evidence for broad open-world generalization by themselves.

## 2. Query bank

Minimum first formal bank: 256 unique, non-equivalent queries per qualified geometry.

The bank should stratify:
- coefficient directions and quadratic-form families;
- spatial cut locations and offsets;
- positive coefficient rescalings (deduplicated as equivalent, not counted as new coverage);
- regular smooth neighborhoods;
- near-event but still derivative-valid neighborhoods;
- tangent, repeated-root, topology-change and other invalid/unknown event cases;
- queries that are relevant to the declared Spin rotation-axis and physical constraints.

Do not inflate query count by duplicating q, positive scalar multiples of q, or numerically equivalent states.

## 3. Labels

Required arrays:
- \`queries[N,10]\`
- \`moments[N,10]\`
- \`jacobian[N,10,10]\`
- \`derivative_valid[N]\`

Optional V2 arrays now accepted by the reader:
- \`event_type[N]\` string;
- \`derivative_valid_reason[N]\` string;
- \`event_margin[N]\` finite numeric;
- \`finite_difference_audit[N]\` bool;
- \`moment_error_bound[N]\` nonnegative finite numeric;
- \`jacobian_error_bound[N]\` nonnegative finite numeric.

Optional event/audit fields are metadata and are not yet used as training targets by the current loss. Their semantics and producer must be frozen before enabling event/UQ losses.

## 4. Manifest provenance

Each qualified record should bind:
- geometry file SHA-256 and immutable geometry identity;
- label file SHA-256;
- source geometry and parent-object group;
- material density and fixed/editable partition policy;
- coordinate frame, native resolution and voxel spacing;
- coefficient basis/order version and moment order/version;
- teacher executable/source hash, teacher precision and integration settings;
- query-bank generator/version and random seed;
- derivative audit procedure and summary;
- train/validation/test split group.

No parent geometry, generated variant, orientation or near-duplicate may leak across splits.

## 5. Admission policy

- Moment labels require finite values and matching source hashes.
- Jacobian labels are admitted only where the teacher certifies a smooth derivative neighborhood and the independent derivative audit passes.
- Invalid/event samples remain in the dataset for event classification and risk analysis but are masked from Jacobian regression.
- Failed or unknown derivatives are never rewritten as zero Jacobians.
- Cherchi remains the authoritative Spin teacher. Algoim, NIFE, Ray-reps and learned quadrature are independent comparisons, not label replacements.

## 6. Expansion plan

- R0: re-audit all 67 current qualified objects and complete 256-query coverage.
- R1: expand to at least 150 independent parent geometry groups.
- R2: target 300+ independent groups for final generalization studies.

R1/R2 are targets, not currently available data. Geometry expansion from TRELLIS.2 or other sources must pass mesh/solid validation, canonicalization, native RLE conversion and teacher admission before entering the formal corpus.

DensiCrafter candidates are separate design proposals; they do not become geometry or successful Spin labels without exact physical evaluation.

## 7. Current implementation boundary

The M72 reader now loads and validates the optional event/derivative audit arrays if present. The current training objective still consumes the required moment/Jacobian arrays only. A later dataset release must version the manifest/audit schema and update the training contract before any optional metadata affects optimization.


## 8. Active query selection

A deterministic cost-aware selector is now available in spin_pino/active_query_selection.py. It consumes (but does not generate) calibrated uncertainty, event margin, solver sensitivity, coverage novelty and estimated teacher cost. It supports mixed, derivative-only and event-label purposes, positive-scale query deduplication, per-source-group caps and a hard teacher-cost budget.

It is not yet connected to a calibrated UQNO/ensemble or the local Cherchi query runner. Until those are connected, use it only with explicitly calibrated input scores or synthetic tests. Every selected query still requires exact teacher evaluation and dataset admission.
