"""Check the two predeclared linear-coefficient columns on the eight M25 native captures."""
import argparse
import json
from pathlib import Path
import sys

from .heart_c0_column import (BASE_CAPTURE, BASE_ID, NAMES, TOL, audit_column, get_json,
                              load_capture)
from .heart_point0_capture_moments import POINT0_EXACT_TEN, POINT0_REQUEST
from .real_h8e_archive_m8 import digest_bytes, read_json, require

COLUMNS = (1, 2)
SAMPLE_SCHEMA = 'w2-m25-linear-sample-v1'


def audit(base_directory, group, group_id, public_result_sha):
    raw = (group/'stencil-result.json').read_bytes()
    require(digest_bytes(raw) == public_result_sha, 'published stencil result mismatch')
    result = read_json(raw)
    plan = get_json(group/'stencil.json')
    require(plan['schema'] == 'w2-m25-linear-plan-v1' and result['schema'] == 'w2-m25-linear-result-v1'
            and plan['group_id'] == result['group_id'] == group_id
            and plan['base_run_id'] == BASE_ID and plan['base_capture_sha256'] == BASE_CAPTURE
            and plan['columns'] == list(COLUMNS) and plan['h'] == 3e-6
            and result['stencil_sha256'] == digest_bytes((group/'stencil.json').read_bytes())
            and len(plan['samples']) == len(result['samples']) == result['fresh_native_boolean_calls'] == 8
            and result['J_D'] is None and result['training_admitted'] is False, 'unexpected linear stencil')
    base_data = load_capture(base_directory, BASE_ID, BASE_CAPTURE)
    base, q, _, _, _, info = base_data
    require(info['exact_moments_sha256'] == POINT0_EXACT_TEN and
            plan['module_sha256'] == base['module_sha256'] and plan['request_sha256'] == POINT0_REQUEST,
            'different base or native modules')
    columns = {}
    for i, column in enumerate(COLUMNS):
        start = 4*i
        rows = slice(start, start+4)
        columns[str(column)] = audit_column(base_data, group, group_id, plan['source_checkout'], column,
                                            plan['samples'][rows], result['samples'][rows], start+1, SAMPLE_SCHEMA)
    return dict(schema='w0-heart-linear-column-stencils-v1', group_id=group_id,
                public_result_sha256=public_result_sha, source_checkout=plan['source_checkout'],
                new_native_boolean_calls_in_w2=8, native_calls_in_w0_replay=0,
                module_sha256=base['module_sha256'], evaluation_map_identity=base['evaluation_map_identity'],
                base=info, base_native_coefficients=q, moment_order=list(NAMES), scaled_error_limit=float(TOL),
                columns=columns, all_sampled_columns_passed=all(c['sampled_column_passed'] for c in columns.values()),
                continuous_neighborhood_certified=False, J_D=None, newton_admitted=False, training_admitted=False)


if __name__ == '__main__':
    sys.set_int_max_str_digits(100000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--expected-group-id', required=True)
    parser.add_argument('--published-result-sha256', required=True)
    args = parser.parse_args()
    report = audit(args.base, args.group, args.expected_group_id, args.published_result_sha256)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report['all_sampled_columns_passed'] else 2)
