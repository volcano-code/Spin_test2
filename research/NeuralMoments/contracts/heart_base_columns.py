"""Audit the registered M27 Heart base stencils and conditional M28 smaller c[7] stencil."""
import argparse
import json
from pathlib import Path
import re
import sys

from .heart_c0_column import (BASE_CAPTURE, BASE_ID, NAMES, OFFSETS, TOL, audit_column,
                              load_capture)
from .heart_point0_capture_moments import POINT0_EXACT_TEN, POINT0_REQUEST
from .real_h8e_archive_m8 import digest_bytes, read_json, require
from .w5_live_handoff_m19 import no_duplicate_json

COLUMNS = tuple(range(3, 10))
SCHEMA = 'w2-m27-base-column'
SAMPLE_SCHEMA = SCHEMA + '-sample-v1'


def audit(base_directory, groups, run_id, attempt, source_checkout, published_receipts, *, prior_m27=None,
          attempts_by_column=None):
    half = prior_m27 is not None
    requested = (7,) if half else COLUMNS
    offsets = tuple(off/2 for off in OFFSETS) if half else OFFSETS
    schema = 'w2-m28-base-c7-half' if half else SCHEMA
    if attempts_by_column is None:
        require(type(attempt) is int and attempt > 0, 'invalid capture run attempt')
        attempts_by_column = {str(k): attempt for k in requested}
    else:
        require(not half and attempt is None and type(attempts_by_column) is dict
                and set(attempts_by_column) == set(map(str, requested))
                and all(type(a) is int and a > 0 for a in attempts_by_column.values()),
                'expected one positive attempt per M27 column')
    require(type(run_id) is int and run_id > 0 and re.fullmatch(r'[0-9a-f]{40}', source_checkout), 'invalid capture run identity')
    require(type(published_receipts) is dict and set(published_receipts) == set(map(str, requested))
            and all(isinstance(sha, str) and re.fullmatch(r'[0-9a-f]{64}', sha)
                    for sha in published_receipts.values()), 'expected public result digests for all requested columns')
    base_data = load_capture(base_directory, BASE_ID, BASE_CAPTURE)
    base, q, _, _, _, info = base_data
    require(info['exact_moments_sha256'] == POINT0_EXACT_TEN, 'different base moments')
    prior_digest = None
    if half:
        raw = prior_m27.read_bytes()
        previous = read_json(raw)
        c7 = previous['columns']['7']
        require(previous['schema'] == 'w0-heart-m27-base-columns-v1' and previous['run_id'] != run_id
                and previous['base']['capture_sha256'] == BASE_CAPTURE
                and previous['module_sha256'] == base['module_sha256'] and 7 in previous['rejected_sampled_columns']
                and c7['sampled_column_passed'] is False and len(c7['perturbations']) == 4
                and any(c7['perturbations'][i]['sampled_geometry_equal'] is False for i in (0, 3)),
                'M28 requires an M27 outer geometry rejection')
        prior_digest = digest_bytes(raw)
    columns = {}
    for k in requested:
        column_attempt = attempts_by_column[str(k)]
        group_id = f'heart-base-c{k}{"-half" if half else ""}-{run_id}-{column_attempt}'
        group = groups / group_id
        raw = (group / 'stencil-result.json').read_bytes()
        receipt_sha = published_receipts[str(k)]
        require(digest_bytes(raw) == receipt_sha, f'c[{k}] published result mismatch')
        result = read_json(raw)
        plan_raw = (group / 'stencil.json').read_bytes()
        require(digest_bytes(plan_raw) == result['stencil_sha256'], f'c[{k}] plan digest mismatch')
        plan = read_json(plan_raw)
        require(plan['schema'] == schema + '-plan-v1' and result['schema'] == schema + '-result-v1'
                and plan['group_id'] == result['group_id'] == group_id
                and plan['source_checkout'] == source_checkout
                and plan['base_run_id'] == BASE_ID and plan['base_capture_sha256'] == BASE_CAPTURE
                and plan['module_sha256'] == base['module_sha256'] and plan['request_sha256'] == POINT0_REQUEST
                and plan['columns'] == [k] and plan['h'] == offsets[-1]
                and len(plan['samples']) == len(result['samples']) == result['fresh_native_boolean_calls'] == len(OFFSETS)
                and result['training_admitted'] is False and result['J_D'] is None,
                f'c[{k}] unexpected stencil')
        report = audit_column(base_data, group, group_id, source_checkout, k,
                              plan['samples'], result['samples'], 1, schema + '-sample-v1', offsets=offsets)
        report.update(group_id=group_id, run_attempt=column_attempt, public_result_sha256=receipt_sha)
        columns[str(k)] = report

    passed = [k for k in requested if columns[str(k)]['sampled_column_passed']]
    return dict(schema='w0-heart-m28-base-c7-half-v1' if half else 'w0-heart-m27-base-columns-v1',
                source_checkout=source_checkout, prior_m27_report_sha256=prior_digest,
                nominal_inner_radius=offsets[-2], nominal_outer_radius=offsets[-1],
                run_id=run_id, run_attempt=attempt, attempts_by_column=attempts_by_column,
                base=info, base_native_coefficients=q,
                evaluation_map_identity=base['evaluation_map_identity'], module_sha256=base['module_sha256'],
                moment_order=list(NAMES), scaled_error_limit=float(TOL), columns=columns,
                new_native_boolean_calls_in_w2=len(requested)*len(offsets), native_calls_in_w0_replay=0,
                accepted_sampled_columns=passed, rejected_sampled_columns=[k for k in requested if k not in passed],
                all_sampled_columns_passed=len(passed) == len(requested), continuous_neighborhood_certified=False,
                J_D=None, newton_admitted=False, training_admitted=False)


if __name__ == '__main__':
    sys.set_int_max_str_digits(100000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--groups', type=Path, required=True)
    parser.add_argument('--run-id', type=int, required=True)
    attempts = parser.add_mutually_exclusive_group(required=True)
    attempts.add_argument('--attempt', type=int, help='Single official attempt for every column.')
    attempts.add_argument('--attempts-by-column', type=Path, metavar='ATTEMPTS_JSON',
                          help='M27 official attempt number per column in the same run and checkout.')
    parser.add_argument('--source-checkout', required=True)
    parser.add_argument('--c7-half', dest='prior_m27', type=Path, metavar='M27_REPORT',
                        help='Score the M28 half-radius c[7] against a recorded M27 outer-geometry rejection.')
    parser.add_argument('--published-receipts', type=Path, required=True,
                        help='Public stencil-result.json SHA-256 values keyed by the requested column indices.')
    args = parser.parse_args()
    report = audit(args.base, args.groups, args.run_id, args.attempt, args.source_checkout,
                   no_duplicate_json(args.published_receipts.read_bytes()), prior_m27=args.prior_m27,
                   attempts_by_column=no_duplicate_json(args.attempts_by_column.read_bytes())
                   if args.attempts_by_column is not None else None)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report['all_sampled_columns_passed'] else 2)
