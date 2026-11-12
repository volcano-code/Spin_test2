"""Deterministic, cost-aware active query selection for expensive Spin teachers.

This is a selection utility, not a UQ model. It consumes calibrated uncertainty,
event-margin, solver-sensitivity and coverage-novelty scores produced by a
separately validated predictor/auditor. It never creates labels or certifies
derivative validity.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class AcquisitionWeights:
    uncertainty: float = 1.0
    solver_sensitivity: float = 0.75
    coverage_novelty: float = 0.5
    event_proximity: float = 0.75

    def __post_init__(self):
        values=(self.uncertainty,self.solver_sensitivity,self.coverage_novelty,self.event_proximity)
        if any(not np.isfinite(v) or v<0 for v in values) or sum(values)<=0:
            raise ValueError("acquisition weights must be finite, nonnegative and not all zero")


@dataclass
class AcquisitionResult:
    selected: list[dict[str, Any]]
    spent_cost: float
    budget: float
    skipped: dict[str, int]
    ranked_count: int

    @property
    def remaining_budget(self) -> float:
        return max(0.0,self.budget-self.spent_cost)

    def summary(self):
        return dict(selected=len(self.selected),spent_cost=self.spent_cost,
                    budget=self.budget,remaining_budget=self.remaining_budget,
                    skipped=dict(self.skipped),ranked_count=self.ranked_count)


def _canonical_positive_scale(q: Any,decimals: int=10) -> tuple[float,...]:
    vector=np.asarray(q,dtype=np.float64)
    if vector.shape!=(10,) or not np.isfinite(vector).all():
        raise ValueError("candidate q must be a finite 10-vector")
    norm=float(np.linalg.norm(vector))
    if norm<=1e-15:
        raise ValueError("zero quadric coefficient vector is not a valid query")
    # Divide by a positive norm only. q and alpha*q (alpha>0) deduplicate;
    # q and -q remain distinct because their material domains are complementary.
    return tuple(np.round(vector/norm,decimals=decimals).tolist())


def select_teacher_queries(
    candidates: Iterable[Mapping[str,Any]],
    *,
    budget: float,
    purpose: str="mixed",
    weights: AcquisitionWeights=AcquisitionWeights(),
    event_tau: float=0.05,
    min_event_margin: float=0.0,
    max_per_group: int|None=None,
    dedup_decimals: int=10,
) -> AcquisitionResult:
    """Select candidates under an estimated teacher-call cost budget.

    Required candidate fields:
      candidate_id, q, uncertainty, solver_sensitivity, coverage_novelty,
      event_margin, predicted_derivative_valid, estimated_teacher_cost.

    purpose='derivative' excludes predicted-invalid and event-near queries;
    purpose='event' favors near-event queries for event-label generation;
    purpose='mixed' ranks all candidates. A selected event query is not a
    valid Jacobian label unless the exact teacher independently certifies it.
    """
    if not np.isfinite(budget) or budget<=0:
        raise ValueError("budget must be finite and positive")
    if purpose not in {"mixed","derivative","event"}:
        raise ValueError("purpose must be mixed, derivative or event")
    if not np.isfinite(event_tau) or event_tau<=0:
        raise ValueError("event_tau must be finite and positive")
    if min_event_margin<0 or not np.isfinite(min_event_margin):
        raise ValueError("min_event_margin must be finite and nonnegative")
    if max_per_group is not None and max_per_group<1:
        raise ValueError("max_per_group must be positive")

    rows=[]; skipped={"invalid":0,"duplicate":0,"purpose_filter":0}
    seen_ids=set(); seen_q=set()
    for raw in candidates:
        row=dict(raw)
        cid=str(row.get("candidate_id",""))
        if not cid or cid in seen_ids:
            skipped["invalid"]+=1
            continue
        seen_ids.add(cid)
        try:
            key=_canonical_positive_scale(row.get("q"),dedup_decimals)
            u=float(row["uncertainty"]); s=float(row["solver_sensitivity"])
            n=float(row["coverage_novelty"]); margin=float(row["event_margin"])
            cost=float(row["estimated_teacher_cost"])
            derivative_valid=row["predicted_derivative_valid"]
            if not isinstance(derivative_valid,(bool,np.bool_)):
                raise ValueError("predicted_derivative_valid must be bool")
            if any(not np.isfinite(v) or v<0 or v>1 for v in (u,s,n)):
                raise ValueError("uncertainty/sensitivity/novelty must lie in [0,1]")
            if not np.isfinite(margin) or not np.isfinite(cost) or cost<=0:
                raise ValueError("event_margin must be finite and teacher cost positive")
        except (KeyError,TypeError,ValueError):
            skipped["invalid"]+=1
            continue
        if key in seen_q:
            skipped["duplicate"]+=1
            continue
        seen_q.add(key)
        if purpose=="derivative" and (not derivative_valid or abs(margin)<min_event_margin):
            skipped["purpose_filter"]+=1
            continue
        event_score=exp(-abs(margin)/event_tau)
        if purpose=="event":
            numerator=(weights.uncertainty*u+
                       weights.event_proximity*event_score+
                       weights.coverage_novelty*n)
        else:
            numerator=(weights.uncertainty*u+
                       weights.solver_sensitivity*s+
                       weights.coverage_novelty*n+
                       weights.event_proximity*event_score)
        row["event_proximity_score"]=event_score
        row["acquisition_score"]=numerator/cost
        row["canonical_query_key"]=key
        rows.append(row)

    rows.sort(key=lambda r:(-r["acquisition_score"],str(r["candidate_id"])))
    selected=[]; spent=0.0; group_counts={}
    for row in rows:
        cost=float(row["estimated_teacher_cost"])
        if spent+cost>budget+1e-12:
            continue
        group=str(row.get("source_group_id","__ungrouped__"))
        if max_per_group is not None and group_counts.get(group,0)>=max_per_group:
            skipped["purpose_filter"]+=1
            continue
        selected.append(row)
        spent+=cost
        group_counts[group]=group_counts.get(group,0)+1
    return AcquisitionResult(selected=selected,spent_cost=spent,budget=float(budget),
                             skipped=skipped,ranked_count=len(rows))
