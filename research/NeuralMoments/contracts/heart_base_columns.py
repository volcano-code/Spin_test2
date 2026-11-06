"""Audit all seven M27 four-point Heart stencils with independent moment reintegration."""
import argparse
import json
from pathlib import Path
import re
import sys

from .heart_c0_column import (BASE_CAPTURE, BASE_ID, NAMES, OFFSETS, TOL, audit_column,
                              load_capture)
from .heart_point0_capture_moments import POINT0_EXACT_TEN, POINT0_REQUEST
from .real_h8e_archive_m8 import digest_bytes, read_json, require

COLUMNS = tuple(range(3, 10))
SCHEMA = 'w2-m27-base-column'
SAMPLE_SCHEMA = SCHEMA + '-sample-v1'


def audit(base_directory, groups, run_id, attempt, source_checkout, published_receipts):
    require(type(run_id) is int and run_id > 0 and type(attempt) is int and attempt > 0
            and re.fullmatch(r'[0-9a-f]{40}', source_checkout), 'invalid capture run identity')
    require(type(published_receipts) is dict and set(published_receipts) == set(map(str, COLUMNS))
            and all(isinstance(sha, str) and re.fullmatch(r'[0-9a-f]{64}', sha)
                    for sha in published_receipts.values()), 'expected seven public result digests')
    base_data = load_capture(base_directory, BASE_ID, BASE_CAPTURE)
    base, q, _, _, _, info = base_data
    require(info['exact_moments_sha256'] == POINT0_EXACT_TEN, 'different base moments')
    columns = {}
    for k in COLUMNS:
        group_id = f'heart-base-c{k}-{run_id}-{attempt}'
        group = groups / group_id
        raw = (group / 'stencil-result.json').read_bytes()
        receipt_sha = published_receipts[str(k)]
        require(digest_bytes(raw) == receipt_sha, f'c[{k}] published result mismatch')
        result = read_json(raw)
        plan_raw = (group / 'stencil.json').read_bytes()
        require(digest_bytes(plan_raw) == result['stencil_sha256'], f'c[{k}] plan digest mismatch')
        plan = read_json(plan_raw)
        require(plan['schema'] == SCHEMA + '-plan-v1' and result['schema'] == SCHEMA + '-result-v1'
                and plan['group_id'] == result['group_id'] == group_id
                and plan['source_checkout'] == source_checkout
                and plan['base_run_id'] == BASE_ID and plan['base_capture_sha256'] == BASE_CAPTURE
                and plan['module_sha256'] == base['module_sha256'] and plan['request_sha256'] == POINT0_REQUEST
                and plan['columns'] == [k] and plan['h'] == 3e-6
                and len(plan['samples']) == len(result['samples']) == result['fresh_native_boolean_calls'] == len(OFFSETS)
                and result['training_admitted'] is False and result['J_D'] is None,
                f'c[{k}] unexpected stencil')
        report = audit_column(base_data, group, group_id, source_checkout, k,
                              plan['samples'], result['samples'], 1, SAMPLE_SCHEMA)
        report.update(group_id=group_id, public_result_sha256=receipt_sha)
        columns[str(k)] = report

    passed = [k for k in COLUMNS if columns[str(k)]['sampled_column_passed']]
    return dict(schema='w0-heart-m27-base-columns-v1', source_checkout=source_checkout,
                run_id=run_id, run_attempt=attempt, base=info, base_native_coefficients=q,
                evaluation_map_identity=base['evaluation_map_identity'], module_sha256=base['module_sha256'],
                moment_order=list(NAMES), scaled_error_limit=float(TOL), columns=columns,
                new_native_boolean_calls_in_w2=len(COLUMNS)*len(OFFSETS), native_calls_in_w0_replay=0,
                accepted_sampled_columns=passed, rejected_sampled_columns=[k for k in COLUMNS if k not in passed],
                all_sampled_columns_passed=len(passed) == len(COLUMNS), continuous_neighborhood_certified=False,
                J_D=None, newton_admitted=False, training_admitted=False)


if __name__ == '__main__':
    sys.set_int_max_str_digits(100000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--groups', type=Path, required=True)
    parser.add_argument('--run-id', type=int, required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--source-checkout', required=True)
    parser.add_argument('--published-receipts', type=Path, required=True,
                        help='Public stencil-result.json SHA-256 values keyed by column "3" through "9".')
    args = parser.parse_args()
    report = audit(args.base, args.groups, args.run_id, args.attempt, args.source_checkout,
                   read_json(args.published_receipts.read_bytes()))
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report['all_sampled_columns_passed'] else 2)
