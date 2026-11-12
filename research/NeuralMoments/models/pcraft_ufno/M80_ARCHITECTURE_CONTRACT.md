# M80 Architecture Contract

## 1. Frozen task contract

The learned system must preserve the Spin task and native solver semantics. Geometry, material density, fixed parts, rotation axis, contact conditions, coefficient convention and terminal constraints are not silently changed by the network.

### Geometry inputs
- Immutable geometry identity and provenance.
- M72 native RLE columns \`(x,y,z_lo,z_hi)\`.
- Native interval centers, extents, masses and ten-component moment basis.
- Four-channel P-CRAFT context: density, occupancy, normalized signed-distance feature and internal boundary.
- Fixed material moments are computed separately from editable-domain moments.

### Moment and coefficient order
- Moment order: \`mass, x, y, z, xx, yy, zz, yz, xz, xy\`.
- Native quadric coefficient order: \`1, x, y, z, xx, yy, zz, 2yz, 2xz, 2xy\`.
- Fixed and editable moments remain separate:
  \`r_total = r_fixed + r_editable\`, \`J_total = J_editable\`.

## 2. Encoder family

### Permanent baseline: M80.1
Single-scale P-CRAFT U-FNO global context + every M72 native interval + bounded odd moment response head. Old single-scale presets and checkpoints remain supported.

### M80.4 optional multi-resolution P-CRAFT
- Shared P-CRAFT weights are evaluated on exact conservative context grids at 32, 64 and 128.
- Each grid is independently aggregated from the original RLE data; it does not replace native intervals.
- A geometry-conditioned global gate fuses pooled scale contexts.
- A descriptor/context-conditioned local gate fuses sampled spatial maps at every native interval.
- All scales must divide the native lattice exactly; unsupported resolutions fail closed rather than silently resample.
- This is an opt-in ablation, not a replacement for the single-scale baseline.

### Optional GNO
- The original Euclidean kNN path remains available for exact reproduction.
- The new topology-filtered path builds 6-connected components from half-open RLE intervals.
- Intervals connect across x/y faces only when z ranges overlap; touching z intervals in the same x/y column connect vertically.
- Topology-filtered neighborhoods never cross disconnected components.
- This is a candidate improvement for thin-wall and near-layer cases; it must be compared against the original graph and exact geometry.

### Optional Transolver++-style physical slicing
- Deterministic multi-head soft slicing.
- Native interval mass-weighted pooling.
- Q/K/V attention over the compact slice axis, followed by deslicing to all native intervals.
- It is a Spin-specific adaptation, not an upstream checkpoint/weight transfer.

## 3. Physical response contract

Current production head:
- M72 bounded odd response head.
- Positive coefficient scaling invariance.
- Complement relation for q and -q.
- Exact full/empty endpoint handling where certified by bounds.
- Jacobian is derived by AD from the same response function.

Not yet implemented:
- Convex moment potential head.
- QuadrANN-inspired positive quadrature readout.
- Calibrated UQ.
- Explicit event classifier and event margin in the production predictor.
- Public HVP service API.

These must be independent candidates. Do not silently replace the current head.

## 4. Cache contract

Geometry caches are valid only for an immutable geometry hash, preprocessing contract and unchanged model parameter versions. Any optimizer update invalidates the encoded cache. Neighbor graph caches are geometry-only, bounded and device-keyed; they must include graph policy in the key if multiple graph policies coexist in one model instance.

## 5. Solver service interface (target)

A future solver-facing response must return:
- editable-domain raw ten moments;
- Jacobian and optional HVP/directional products;
- geometry/model/source hashes;
- derivative-valid status and event reason;
- uncertainty/risk estimate;
- latency and cache provenance.

The native exact evaluator remains authoritative. An invalid, unknown, event-adjacent or uncertified derivative is withheld and triggers exact evaluation.

## 6. Compatibility

- M80.1 single-scale presets remain unchanged.
- M80.2 GNO and M80.3 Transolver remain independently switchable.
- New M80.4 multi-resolution presets are opt-in.
- Checkpoint schema v4 accepts the recorded v1/v2/v3 legacy schemas with new modules disabled by default.
- A checkpoint must not silently enable a module absent from its original config.
