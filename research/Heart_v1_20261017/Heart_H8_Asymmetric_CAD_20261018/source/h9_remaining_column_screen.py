"""Reintegrate and screen the historical H8f stencils of native c[3] through c[9]."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import json
from pathlib import Path

from audit_h9_branch_decision import ZIP_SHA256, det4, exact_point
from certify_h9_interior_edge_crossing import archival_data
from h9_c7_guarded_stencils import TOL, difference, patch
from h9_m20_precision_replay import NAMES, exact_surface_moments, need
from locate_h9_transition import DESIGN_EDGE, SOURCE_EDGE

COLUMNS = range(3, 10)


def indices(column):
    return [0, *range(1 + 8 * column, 9 + 8 * column)]


def changed_coordinate(base, sample, column, index):
    need(len(base) == len(sample) == 10, 'INVALID_COEFFICIENT_COUNT')
    changed = [j for j in range(10) if sample[j] != base[j]]
    need(changed == ([] if index == 0 else [column]), 'WRONG_PERTURBATION_COORDINATE')


def same_runtime(base, material):
    return all(material[k] == base[k] for k in (
        'evaluation_map_identity', 'boolean_backend_identity', 'rational_arithmetic_identity',
    )) and material['boolean_operation'] == 'subtraction' and (
        material['derivative_identity'] == 'cherchi-constructed-vertex-motion-v1'
        and material['derivative_defined'] is True
    )


def screen_column(archive, column):
    request, report, materials = archival_data(archive, indices(column))
    base = materials[0][1]
    base_q = base['quadric_coefficients']
    need(request['request']['material_density'] == 1 and base_q == report['target_q'],
         'UNEXPECTED_BASE')
    native = [Fraction(row[column]) for row in base['moment_jacobian']]
    need(native == [Fraction(row[column]) for row in report['analytic_10_by_10_jacobian']],
         'BASE_JACOBIAN_MISMATCH')
    base_keys, base_triangles, base_source = patch(base)
    d, e = [exact_point(request['request']['design_vertices'][j]) for j in DESIGN_EDGE]
    points, evidence = {}, []
    for i, (record, material) in materials.items():
        q = material['quadric_coefficients']
        changed_coordinate(base_q, q, column, i)
        need(same_runtime(base, material), 'CHANGED_EVALUATION_MAP')
        keys, triangles, source = patch(material)
        same_patch = keys == base_keys and triangles == base_triangles and source == base_source
        a, b = [exact_point(material['quadric_source_vertices'][j]) for j in SOURCE_EDGE]
        det = det4(a, b, d, e)
        need(det != 0, 'SAMPLED_KNOWN_EDGE_CONTACT')
        moments = exact_surface_moments(material)
        need(list(map(float, moments)) == record['independent_variable_raw_moments'],
             'ARCHIVED_MOMENT_MISMATCH')
        points[i] = dict(index=i, x=Fraction(q[column]), moments=moments, native=native,
                         base_patch=same_patch, edge_sign=1 if det > 0 else -1)
        evidence.append(dict(index=i, coefficient_hex=q[column].hex(),
                             archive_member_sha256=record['exact_same_call_material_archive_sha256'],
                             same_patch=same_patch, edge_sign=points[i]['edge_sign'],
                             removed_keys=len(base_keys - keys), added_keys=len(keys - base_keys),
                             removed_triangles=sum((base_triangles - triangles).values()),
                             added_triangles=sum((triangles - base_triangles).values())))
    stencils = []
    for check in report['numeric_checks']:
        if check['column'] != column:
            continue
        sample_ids = check['sample_indices']
        samples = [points[i] for i in sample_ids]
        need(len(samples) == 5 and samples[2]['index'] == 0 and
             all(a['x'] < b['x'] for a, b in zip(samples, samples[1:])), 'BAD_STENCIL')
        result = difference(points[0], samples)
        stencils.append(dict(outer_offset=check['step'], **result))
    need(len(stencils) == 3, 'MISSING_STENCIL')
    return dict(column=column, evaluation_map=base['evaluation_map_identity'],
                native_base_column=list(map(float, native)), samples=evidence, stencils=stencils)


def audit(archive, jobs):
    need(1 <= jobs <= len(COLUMNS), 'INVALID_WORKER_COUNT')
    if jobs == 1:
        columns = [screen_column(archive, col) for col in COLUMNS]
    else:
        with ProcessPoolExecutor(jobs) as pool:
            columns = list(pool.map(screen_column, [archive] * len(COLUMNS), COLUMNS))
    return dict(schema='h9-h8f-remaining-column-screen-v1', archive_sha256=ZIP_SHA256,
                moment_order=NAMES, scaled_error_limit=float(TOL), unique_historical_points=57,
                columns=columns, new_native_calls=0, continuous_neighborhood_certified=False,
                J_D=None, training_admitted=False, newton_admitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.jobs), indent=2, allow_nan=False))
