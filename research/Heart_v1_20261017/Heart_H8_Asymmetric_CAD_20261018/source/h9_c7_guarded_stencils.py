"""H8f c[7] replay: report numerical and sampled-branch checks independently."""
import argparse
from collections import Counter
from fractions import Fraction as F
import json
from math import prod
from pathlib import Path

from audit_h9_branch_decision import ZIP_SHA256, det4, exact_point
from certify_h9_interior_edge_crossing import archival_data
from h9_m20_precision_replay import exact_surface_moments, NAMES, need
from h9_m21_motion_math import key_for
from locate_h9_transition import DESIGN_EDGE, SOURCE_EDGE

# Index 60 is the only post-event sample in the original H8f stencil.
# Use stored binary64 abscissae, independently of the step labels.
STENCILS = {
    'base_backward_coarse': (0, 61, 58),
    'base_backward_fine': (0, 63, 61),
    'base_forward_coarse': (0, 62, 59),
    'base_forward_fine': (0, 64, 62),
    'base_five_point_outer': (57, 58, 0, 59, 60),
    'base_five_point_middle': (58, 61, 0, 62, 59),
    'base_five_point_inner': (61, 63, 0, 64, 62),
}
TOL = F(1, 1_000_000)


def patch(material):
    rows = material['material_proof']['resolved_construction_records']
    need([r['vertex_index'] for r in rows] == list(range(len(material['variable_rational_vertices']))),
         'UNBOUND_CONSTRUCTION_SLOTS')
    keys = [key_for(r['point_type'], r['operands']) for r in rows]
    identities = frozenset(keys)
    need(len(identities) == len(keys), 'NONUNIQUE_CONSTRUCTIONS')
    triangles = Counter()
    for face in material['variable_faces']:
        a, b, c = (keys[i] for i in face)
        triangles[min((a,b,c), (b,c,a), (c,a,b))] += 1
    return identities, triangles, material['quadric_source_faces']


def difference(anchor, samples):
    xs = [s['x'] - anchor['x'] for s in samples]
    need(F(0) in xs and len(set(xs)) == len(xs) and len(xs) >= 2, 'INVALID_STENCIL')
    weights = []
    for i, x in enumerate(xs):
        rest = xs[:i] + xs[i+1:]
        numerator = sum(prod(-v for j,v in enumerate(rest) if j != k) for k in range(len(rest)))
        weights.append(F(numerator, prod(x-v for v in rest)))
    fd = [sum(w*s['moments'][k] for w,s in zip(weights, samples)) for k in range(10)]
    native = anchor['native']
    err = [abs(v-j)/max(F(1),abs(j)) for v,j in zip(fd,native)]
    same = all(s['base_patch'] == anchor['base_patch'] and s['edge_sign'] == anchor['edge_sign']
               for s in samples)
    return dict(indices=[s['index'] for s in samples], derivative=list(map(float,fd)),
                scaled_errors=list(map(float,err)), max_scaled_error=float(max(err)),
                numeric_within_tolerance=max(err) <= TOL, observed_same_branch=same,
                sampled_stencil_passed=same and max(err) <= TOL)


def audit(archive):
    frozen, report, materials = archival_data(archive)
    base = materials[0][1]
    c = base['quadric_coefficients']
    need(c[7] == 0 and frozen['request']['material_density'] == 1, 'UNEXPECTED_BASE_OR_DENSITY')
    d, e = [exact_point(frozen['request']['design_vertices'][j]) for j in DESIGN_EDGE]
    base_keys, base_faces, base_source = patch(base)
    samples, evidence = {}, []
    for i, (record, m) in materials.items():
        q = m['quadric_coefficients']
        need(len(q) == len(c) == 10 and q[:7]+q[8:] == c[:7]+c[8:]
             and m['evaluation_map_identity'] == base['evaluation_map_identity']
             and m['boolean_backend_identity'] == base['boolean_backend_identity']
             and m['rational_arithmetic_identity'] == base['rational_arithmetic_identity']
             and m['boolean_operation'] == 'subtraction'
             and m['derivative_identity'] == 'cherchi-constructed-vertex-motion-v1'
             and m['derivative_defined'] is True, 'CHANGED_INPUT_OR_RUNTIME')
        sk, sf, src = patch(m)
        same = sk == base_keys and sf == base_faces and src == base_source
        a,b = [exact_point(m['quadric_source_vertices'][j]) for j in SOURCE_EDGE]
        det = det4(a,b,d,e)
        need(det != 0, 'EXACT_SAMPLED_CONTACT')
        sgn = 1 if det > 0 else -1
        moments = exact_surface_moments(m)
        need(list(map(float,moments)) == record['independent_variable_raw_moments']
             and len(m['variable_raw_moments']) == 10 and
             all(abs(v-F(w)) <= F(1e-10) for v,w in zip(moments, m['variable_raw_moments'])),
             'ARCHIVED_MOMENT_MISMATCH')
        samples[i] = dict(index=i, x=F(q[7]), edge_sign=sgn, base_patch=same, moments=moments,
                          native=[F(row[7]) for row in m['moment_jacobian']])
        evidence.append(dict(index=i,c7=q[7],material_archive_sha256=record['exact_same_call_material_archive_sha256'],
                             same_as_base=same, edge_sign=sgn,
                             removed_keys=len(base_keys-sk), added_keys=len(sk-base_keys),
                             removed_triangles=sum((base_faces-sf).values()), added_triangles=sum((sf-base_faces).values())))
    # Anchor all stencils at the base (native matrix coordinate c7, off-diagonal yz).
    need(samples[0]['native'] == [F(row[7]) for row in report['analytic_10_by_10_jacobian']],
         'BASE_J_COLUMN_MISMATCH')
    stencils = {name:difference(samples[0], [samples[i] for i in ids]) for name,ids in STENCILS.items()}
    need([i for i,s in samples.items() if not s['base_patch']] == [60] and
         [i for i,s in samples.items() if s['edge_sign']<0] == [60],
         'UNEXPECTED_EVENT_SAMPLE_PATTERN')
    return dict(schema='h9-h8f-c7-branch-guarded-stencils-v1', archive_sha256=ZIP_SHA256,
                historical_native_evaluation_map=base['evaluation_map_identity'], native_column=7,
                moment_order=NAMES, scaled_error_limit=float(TOL),
                base_native_column=list(map(float,samples[0]['native'])),
                samples=sorted(evidence,key=lambda s:s['c7']), stencils=stencils,
                native_sample_bracket=[dict(index=i,c7=float(samples[i]['x'])) for i in (59,60)],
                post_event_samples=[60], post_event_one_sided_estimate=None,
                new_native_calls=0, continuous_neighborhood_certified=False, newton_admitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('h8f_archive', type=Path)
    print(json.dumps(audit(parser.parse_args().h8f_archive), indent=2,allow_nan=False))
