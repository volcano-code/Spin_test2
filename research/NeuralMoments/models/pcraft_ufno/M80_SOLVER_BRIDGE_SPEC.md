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
