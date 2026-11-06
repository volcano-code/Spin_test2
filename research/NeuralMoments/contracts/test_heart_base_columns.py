from collections import Counter
from fractions import Fraction as F
import json

import pytest

from . import heart_base_columns as batch
from . import heart_c0_column as common
from .real_h8e_archive_m8 import canonical_digest, digest_bytes


def encode(obj):
    return json.dumps(obj).encode()


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source_sha = 'a'*40
    q = [0.25, -0.125, -0.0, -0.375, 0.125, 0.125, 0.125, 0.0, 0.0, 0.0]
    # Each column has a distinct derivative in each of the ten moment rows.
    jac = [[F(row*10+col) for col in range(10)] for row in range(10)]
    mesh = ({'v'}, Counter({'tri': 1}), 'operand-faces')
    cap = dict(source_checkout=source_sha, module_sha256={'core': 'test'}, evaluation_map_identity='test')
    base = (cap, q, [F(1)]*10, jac, mesh, dict(exact_moments_sha256=batch.POINT0_EXACT_TEN, capture_sha256=batch.BASE_CAPTURE))
    receipts = {}
    samples = {}
    for col, half in [(k, False) for k in batch.COLUMNS] + [(7, True)]:
        group_id = 'heart-base-c7-half-1235-1' if half else f'heart-base-c{col}-1234-1'
        schema = 'w2-m28-base-c7-half' if half else batch.SCHEMA
        group = tmp_path / group_id
        group.mkdir()
        plan = dict(schema=schema+'-plan-v1', group_id=group_id, source_checkout=source_sha,
                    base_run_id=batch.BASE_ID, base_capture_sha256=batch.BASE_CAPTURE,
                    module_sha256=cap['module_sha256'], request_sha256=batch.POINT0_REQUEST,
                    columns=[col], h=1.5e-6 if half else 3e-6, samples=[])
        result = dict(schema=schema+'-result-v1', group_id=group_id, fresh_native_boolean_calls=4,
                      J_D=None, training_admitted=False, samples=[])
        for index, off in enumerate(common.OFFSETS, 1):
            if half:
                off /= 2
            name = f'{group_id}-{index:03d}'
            coords = q.copy()
            coords[col] += off
            dx = F(coords[col]) - F(q[col])
            plan['samples'].append(dict(index=index, run_id=name, column=col, nominal_offset=off,
                                        actual_offset_exact=str(dx), native_c=coords,
                                        native_c_sha256=canonical_digest(coords)))
            result['samples'].append(dict(index=index, run_id=name, column=col,
                                          same_call_material_sha256=name, capture_sha256=name))
            # c6 fails a numeric test; c7 fails a geometry test despite good numbers.
            values = [1 + (row[col] + int(col == 6 and j == 4))*dx for j,row in enumerate(jac)]
            sample_mesh = (mesh[0], mesh[1], 'changed connectivity') if col == 7 and index == 4 and not half else mesh
            samples[name] = cap, coords, values, None, sample_mesh, dict(same_call_material_sha256=name)
        (group/'stencil.json').write_bytes(encode(plan))
        result['stencil_sha256'] = digest_bytes(encode(plan))
        (group/'stencil-result.json').write_bytes(encode(result))
        if not half:
            receipts[str(col)] = digest_bytes(encode(result))

    def load_sample(path, run_id, capture_sha, expected_c, *, sample_schema):
        assert run_id == capture_sha == path.name
        assert sample_schema == ('w2-m28-base-c7-half-sample-v1' if '-half-' in run_id else batch.SAMPLE_SCHEMA)
        assert canonical_digest(expected_c) == canonical_digest(samples[run_id][1])
        return samples[run_id]

    def load_base(path, run_id, capture_sha):
        assert run_id == batch.BASE_ID and capture_sha == batch.BASE_CAPTURE
        return base

    monkeypatch.setattr(batch, 'load_capture', load_base)
    monkeypatch.setattr(common, 'load_capture', load_sample)
    return (tmp_path, tmp_path, 1234, 1, source_sha, receipts)


def test_all_seven_scored_despite_numeric_and_geometry_failures(campaign):
    out = batch.audit(*campaign)
    assert out['accepted_sampled_columns'] == [3,4,5,8,9]
    assert out['rejected_sampled_columns'] == [6,7]
    assert not out['all_sampled_columns_passed'] and out['new_native_boolean_calls_in_w2'] == 28
    assert out['columns']['6']['sampled_geometry_equal'] and not out['columns']['6']['numeric_column_passed']
    assert out['columns']['7']['numeric_column_passed'] and not out['columns']['7']['sampled_geometry_equal']
    assert out['J_D'] is None and not out['newton_admitted'] and not out['training_admitted']


def test_receipts_cannot_be_substituted_across_columns(campaign):
    campaign[-1]['7'] = campaign[-1]['8']
    with pytest.raises(ValueError, match=r'c\[7\] published result mismatch'):
        batch.audit(*campaign)


def test_missing_receipt_does_not_become_column_rejection(campaign):
    campaign[-1].pop('6')
    with pytest.raises(ValueError, match='expected public result digests'):
        batch.audit(*campaign)


def test_rejected_scientific_result_does_not_mask_later_integrity_failure(campaign):
    path = campaign[1] / 'heart-base-c8-1234-1' / 'stencil.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match=r'c\[8\] plan digest mismatch'):
        batch.audit(*campaign)


def test_wrong_source_commit_rejected_with_valid_receipts(campaign):
    with pytest.raises(ValueError, match='unexpected stencil'):
        batch.audit(*campaign[:4], 'f'*40, campaign[-1])


def test_m28_smaller_radius_is_scored_separately_and_requires_m27_outer_rejection(campaign):
    m27_path = campaign[0] / 'm27.json'
    m27 = batch.audit(*campaign)
    m27_path.write_bytes(encode(m27))
    raw = (campaign[1]/'heart-base-c7-half-1235-1'/'stencil-result.json').read_bytes()
    args = (*campaign[:2], 1235, 1, campaign[4], {'7': digest_bytes(raw)})
    out = batch.audit(*args, prior_m27=m27_path)
    assert out['schema'] == 'w0-heart-m28-base-c7-half-v1' and out['accepted_sampled_columns'] == [7]
    assert out['prior_m27_report_sha256'] == digest_bytes(encode(m27)) and out['new_native_boolean_calls_in_w2'] == 4
    assert out['nominal_inner_radius'] == 0.75e-6 and out['nominal_outer_radius'] == 1.5e-6
    assert out['J_D'] is None and not out['newton_admitted'] and not out['training_admitted']
    # An inner-only rejection is outside the registered M28 trigger.
    for i in (0,3):
        m27['columns']['7']['perturbations'][i]['sampled_geometry_equal'] = True
    m27_path.write_bytes(encode(m27))
    with pytest.raises(ValueError, match='requires an M27 outer geometry rejection'):
        batch.audit(*args, prior_m27=m27_path)
