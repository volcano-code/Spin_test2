from collections import Counter
from fractions import Fraction as F

import pytest

from . import heart_c0_column as common
from . import heart_linear_columns as linear
from .real_h8e_archive_m8 import canonical_digest as digest


def linear_fixture(monkeypatch, col):
    q = [0.375, -0.125, -0.0] + [0.]*7
    slopes = [[F(10*i+j) for j in range(10)] for i in range(10)]
    moments = [F(4)]*10
    cap = dict(source_checkout='tree', evaluation_map_identity='eval', module_sha256={'core':'same'})
    mesh = ({'v'}, Counter({'f':1}), 'same source')
    base = (cap, q, moments, slopes, mesh, {})
    planned, recorded = [], []
    first = 1 if col == 1 else 5
    for i, off in enumerate(common.OFFSETS, first):
        coeff = q.copy()
        coeff[col] += off
        x = F(coeff[col]) - F(q[col])
        run_id = f'group-{i:03d}'
        planned.append(dict(index=i, run_id=run_id, nominal_offset=off, native_c_sha256=digest(coeff),
                            native_c=coeff, column=col, actual_offset_exact=str(x)))
        recorded.append(dict(index=i, run_id=run_id, column=col, same_call_material_sha256=run_id, capture_sha256=run_id))

    def read(directory, run_id, capture_sha, expected_c, *, sample_schema):
        ii = int(run_id[-3:]) - first
        coeff = planned[ii]['native_c']
        assert capture_sha == run_id and expected_c == coeff and sample_schema == linear.SAMPLE_SCHEMA
        assert all(digest(coeff[j]) == digest(q[j]) for j in range(len(q)) if j!=col)
        delta = F(coeff[col]) - F(q[col])
        # Independent oracle: x contributes the second column; y contributes the third.
        values = [m + row[1]*(F(coeff[1])-F(q[1])) + row[2]*(F(coeff[2])-F(q[2])) + delta**2
                  for m,row in zip(moments,slopes)]
        return cap,coeff,values,None,mesh,dict(same_call_material_sha256=run_id)
    monkeypatch.setattr(common, 'load_capture', read)
    return base,planned,recorded,first


@pytest.mark.parametrize('column', [1,2])
def test_linear_column_selects_single_coefficient_and_correct_ten_native_derivatives(monkeypatch, tmp_path, column):
    base, plan, result, first = linear_fixture(monkeypatch, column)
    out = common.audit_column(base, tmp_path, 'group', 'tree', column, plan, result, first, linear.SAMPLE_SCHEMA)
    assert out['sampled_column_passed']
    assert out['native_column'] == [10*i+column for i in range(10)]
    assert all(max(stencil['absolute_errors']) == 0 for stencil in out['stencils'].values())


def test_wrong_axis_or_reordered_capture_cannot_satisfy_linear_plan(monkeypatch, tmp_path):
    base, plan, result, first = linear_fixture(monkeypatch, 2)
    result[1]['column'] = 1
    with pytest.raises(ValueError, match='wrong column index'):
        common.audit_column(base, tmp_path, 'group', 'tree', 2, plan, result, first, linear.SAMPLE_SCHEMA)
    result[1]['column'] = 2
    result.reverse()
    with pytest.raises(ValueError, match='wrong stencil sample layout'):
        common.audit_column(base, tmp_path, 'group', 'tree', 2, plan, result, first, linear.SAMPLE_SCHEMA)


def test_public_receipt_rejected_before_loading_packets(tmp_path):
    (tmp_path/'stencil-result.json').write_bytes(b'{}')
    with pytest.raises(ValueError, match='published stencil result mismatch'):
        linear.audit(tmp_path, tmp_path, 'group', 'wrong')
