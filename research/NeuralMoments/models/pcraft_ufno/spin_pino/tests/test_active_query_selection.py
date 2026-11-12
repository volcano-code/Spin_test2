import numpy as np
import pytest

from research.NeuralMoments.models.pcraft_ufno.spin_pino.active_query_selection import (
    AcquisitionWeights,select_teacher_queries
)


def _candidate(cid,q,**overrides):
    row=dict(candidate_id=cid,q=np.asarray(q,dtype=np.float64),
        uncertainty=.8,solver_sensitivity=.7,coverage_novelty=.6,
        event_margin=.2,predicted_derivative_valid=True,
        estimated_teacher_cost=1.,source_group_id="g0")
    row.update(overrides)
    return row


def test_active_selection_is_deterministic_and_cost_bounded():
    candidates=[
        _candidate("a",[1,0,0,0,0,0,0,0,0,0],estimated_teacher_cost=2.),
        _candidate("b",[0,1,0,0,0,0,0,0,0,0],estimated_teacher_cost=1.),
        _candidate("c",[0,0,1,0,0,0,0,0,0,0],estimated_teacher_cost=1.),
    ]
    a=select_teacher_queries(candidates,budget=2.0)
    b=select_teacher_queries(candidates,budget=2.0)
    assert [x["candidate_id"] for x in a.selected]==[x["candidate_id"] for x in b.selected]
    assert a.spent_cost<=2.0
    assert a.summary()["remaining_budget"]>=0


def test_positive_scale_equivalent_queries_deduplicate_but_negative_does_not():
    q=np.asarray([1.,.2,0,0,0,0,0,0,0,0])
    candidates=[
        _candidate("q",q),
        _candidate("scaled",7*q),
        _candidate("complement",-q),
    ]
    result=select_teacher_queries(candidates,budget=10)
    ids={x["candidate_id"] for x in result.selected}
    assert "q" in ids
    assert "scaled" not in ids
    assert "complement" in ids
    assert result.skipped["duplicate"]==1


def test_derivative_purpose_excludes_invalid_and_event_near_queries():
    candidates=[
        _candidate("safe",[1,0,0,0,0,0,0,0,0,0],event_margin=.2),
        _candidate("near",[0,1,0,0,0,0,0,0,0,0],event_margin=.001),
        _candidate("invalid",[0,0,1,0,0,0,0,0,0,0],event_margin=.2,
                   predicted_derivative_valid=False),
    ]
    result=select_teacher_queries(candidates,budget=10,purpose="derivative",min_event_margin=.01)
    assert [x["candidate_id"] for x in result.selected]==["safe"]
    assert result.skipped["purpose_filter"]==2


def test_event_purpose_prioritizes_near_event_but_does_not_certify_jacobian():
    near=_candidate("near",[1,0,0,0,0,0,0,0,0,0],event_margin=.001,
                    predicted_derivative_valid=False)
    far=_candidate("far",[0,1,0,0,0,0,0,0,0,0],event_margin=.5)
    result=select_teacher_queries([far,near],budget=1.,purpose="event")
    assert result.selected[0]["candidate_id"]=="near"
    assert result.selected[0]["predicted_derivative_valid"] is False


def test_group_cap_and_invalid_candidates_are_handled():
    candidates=[
        _candidate("a",[1,0,0,0,0,0,0,0,0,0],source_group_id="g1"),
        _candidate("b",[0,1,0,0,0,0,0,0,0,0],source_group_id="g1"),
        _candidate("c",[0,0,1,0,0,0,0,0,0,0],source_group_id="g2"),
        _candidate("bad",[0,0,0,0,0,0,0,0,0,0]),
    ]
    result=select_teacher_queries(candidates,budget=5.,max_per_group=1)
    groups=[x["source_group_id"] for x in result.selected]
    assert len(groups)==len(set(groups))
    assert result.skipped["invalid"]==1


def test_acquisition_weight_and_budget_validation():
    with pytest.raises(ValueError):
        AcquisitionWeights(0,0,0,0)
    with pytest.raises(ValueError):
        select_teacher_queries([],budget=0)
