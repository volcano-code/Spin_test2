# M80 Spin-It Faster Solver Bridge Specification

## 1. Purpose

Connect the learned geometry/moment service to the native Spin-It Faster solver without changing the task or the solver's correctness authority. This document is a target interface, not a claim that the bridge is already implemented.

## 2. Required source audit before coding

Pin and inspect the exact native Spin-It Faster source used for comparison:
- coefficient order and cross-term factors;
- moment order and units;
- equality-Newton optimality equations;
- objective and constraints;
- initial guess and stopping criteria;
- line search/trust region and singular-system handling;
- event/active-set behavior;
- exact integration entry points and return statuses.

Do not invent an adapter around an assumed API. Preserve the native equality-Newton path separately from any general inequality/active-set extension.

## 3. Proposed service contract

Input:
- immutable geometry/material identity;
- native coefficient vector in the frozen order;
- fixed/variable material partition;
- solver iteration and event context.

Output:
- editable-domain ten moments;
- Jacobian from the same differentiable response;
- optional HVP/directional products only after validated implementation;
- derivative-valid status, event type/reason and event margin;
- calibrated uncertainty/risk estimate;
- model/data/source hashes and latency.

Fixed material moments remain separate and are composed under the native task contract.

## 4. Solver modes to compare

A. Native exact baseline: original initial guess + exact native evaluator + original Newton.

B. Learned initialization: neural proposal for q0, then original native exact evaluator and original Newton.

C. Learned response with safe correction: use predicted moments/Jacobian only when all admission gates pass; otherwise call the exact evaluator.

D. Adaptive exact correction: periodically or conditionally compare learned and exact residuals, correct the model state, and fall back immediately on disagreement or event uncertainty.

These are separate ablations. Do not combine their effects into one speedup number.

## 5. Fail-closed admission

A learned derivative may be used only when:
- geometry/task/model hashes match;
- the point is in a certified smooth neighborhood;
- event classifier and event margin permit it;
- uncertainty is within calibrated threshold;
- residual and derivative consistency checks pass;
- trust-region/step acceptance passes.

Unknown, singular, tangent, repeated-root, topology-changing, stale-cache or out-of-distribution states withhold learned derivatives and invoke the exact native path.

## 6. Accounting

Online wall time includes:
- model loading or amortized initialization (report both cold and warm);
- geometry encoding and cache construction;
- Moment/Jacobian/HVP inference;
- native Newton linear solves;
- exact teacher evaluations;
- correction and fallback;
- terminal independent audit.

Report exact-call count, fallback rate, Newton iterations, residual history, terminal success, final material mass and full elapsed time.

## 7. Correctness gate

The final design must pass the independent native physical audit. A neural prediction or an internal model confidence score is never itself a terminal acceptance certificate.


## 8. Existing native bridge discovered and M80 adapter status

The repository already contains a native strict-Newton external-evaluator path:
- Python wrapper: python/make_it_stand/spin.py, function solve_spin_it_faster_with_external_moment_evaluator.
- Pybind callback contract: bindings/pybind_module.cpp.
- Native solver implementation: src/spin/QuadricHollowingSolver.cpp and the related solver headers.
- Required equation profile: DUAL_MONOMIAL_RAW_V1.
- Required callback fields: raw_moments[10], moment_jacobian[10,10], derivative_defined, geometry_hash, topology_hash, evaluation_map_identity, boolean_backend_identity and derivative_identity. Optional variable_raw_moments and timing/diagnostic fields are supported.

The wrapper documentation explicitly states that this path produces numerical Newton evidence only; it does not seal or independently validate the final material geometry.

Existing Cherchi/strict integration experiments are not a successful end-to-end neural result. The 2026-09-25 integration note records a non-converged synthetic boundary diagnostic; the 2026-09-27 locked-shell note records a seed-only in-process pilot with an unsupported terminal state. These must remain historical diagnostics, not be described as a completed solver bridge.

M80.8 now has an initial callback adapter at:
- spin_pino/neural_solver_bridge.py
- tests/test_neural_solver_bridge.py

The adapter:
- requires explicit coefficient-to-model and moment-to-target 10x10 transforms;
- distinguishes the original solver geometry hash from the M72/RLE cache hash;
- validates model/cache ownership and parameter versions;
- requires an exact evaluator fallback;
- fails closed when no admission callback is supplied;
- supports event/UQ/physical admission through a caller-supplied gate;
- checks finite values and the existing absolute-moment envelope;
- additionally checks symmetry/PSD before admitting the convex-potential Jacobian;
- uses one declared hybrid map identity across neural and exact-fallback calls while recording the per-call path in diagnostics.

## 9. Remaining integration gates

The callback adapter is not yet a complete production bridge. Before calling the native wrapper:
1. Bind each M72 record to the original source mesh and the exact SpinItFasterRequest geometry/topology identity.
2. Derive and independently test coefficient_to_model and moment_to_target from the request target frame; do not assume identity transforms.
3. Bind the exact Cherchi evaluator for the same source geometry and topology.
4. Construct a calibrated admission callback. Until UQ/event certification is ready, leave admission=None so every call uses the exact fallback.
5. Run native callback parity tests with identity transforms first, then nontrivial rigid/affine frame transforms.
6. Run a synthetic native Newton replay, then a real geometry replay with neural path disabled (exact fallback) to prove callback parity.
7. Enable neural calls only after model/teacher parity, derivative validity and risk gates pass.
8. Independently audit the final material geometry and terminal constraints using the exact native path.

The exact evaluator and neural surrogate must use the same source geometry and target frame. A model trained on an RLE approximation may be used only if its training labels are tied to the same source geometry and the approximation/provenance relationship is explicitly recorded.


### Frame-transform helper

The adapter now provides target_frame_moment_transforms(target_origin, target_axes). It constructs the 10x10 raw-moment transform T for the basis (1,x,y,z,x²,y²,z²,yz,xz,xy), with:

- moment_to_target = T;
- coefficient_to_model = T.T;
- q_model = T.T @ q_target;
- r_target = T @ r_model;
- J_target = T @ J_model @ T.T.

The helper requires a finite origin and a right-handed orthonormal target frame. It is tested pointwise against polynomial basis evaluation. It does not infer the relationship between an arbitrary M72 RLE preprocessing frame and the original mesh frame; that source-to-cache transform/provenance must still be supplied and audited.

Native solver API reference:
- python/make_it_stand/spin.py: solve_spin_it_faster_with_external_moment_evaluator.
- bindings/pybind_module.cpp: callback field parsing and strict equation-profile binding.


Frame-transform consistency is enforced in code: coefficient_to_model must equal moment_to_target.T. This is required to preserve the dual pairing q^T m under coordinate changes. The target-frame helper is tested pointwise; the actual request/source-mesh transform still requires parity against the native C++ transform.
