"""Exact finite-stencil check of the native Heart constant-coefficient column."""
import argparse
from collections import Counter
from fractions import Fraction as F
import json
from math import prod
from pathlib import Path
import sys

from .heart_point0_capture_moments import POINT0_EXACT_TEN, POINT0_REQUEST
from .native_teacher_replay import affine_target_from_physical, frac
from .real_h8e_archive_m8 import (canonical_digest as digest, check_closed_indexed_surface,
                                   digest_bytes, exact_rational_ten_moments, read_json, require)
from .real_h8f_81_m9 import normalized_features, oriented_triangle_multiset

BASE_ID = "heart-point000-42177937306-1"
BASE_CAPTURE = "816fc58eda9b20a65efccff00b3b803b2870315d9422957d1d141dbe6f68da04"
OFFSETS = (-3e-6, -1.5e-6, 1.5e-6, 3e-6)
POINT0_SETTINGS = "3e1dab1b74ab0c4c60bd5d54845df590e908f7b2c11f4003e89cc0284b31b8da"
POINT0_FIXED_TEN = "8152490745a60b34bdfaaeb1f2586644ed5f97114e97b90a66fa6eb7dbfb8d16"
NAMES = ("m", "x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy")
TOL = F(1, 1000000)
FILES = {"frozen_normalized_outer.obj", "label.json", "material.json", "module-manifest.json",
         "native-result.json", "resources-before.json", "resources-after.json", "spec.json"}


def get_json(path):
    return read_json(path.read_bytes())


def load_capture(directory, run_id, capture_sha, expected_c=None, *, sample_schema="w2-m24-c0-sample-v1"):
    raw = (directory / 'capture.json').read_bytes()
    require(digest_bytes(raw) == capture_sha, 'capture digest mismatch')
    cap = read_json(raw)
    schema = 'w2-m23-fresh-heart-capture-v1' if expected_c is None else sample_schema
    require(cap['schema'] == schema and cap['run_id'] == run_id and cap['fresh_evaluation_verified'] is True
            and type(cap['native_cherchi_calls']) is int and cap['native_cherchi_calls'] == 1
            and cap['training_admitted'] is False and cap['J_D'] is None, 'invalid native capture')
    require(set(cap['files']) == FILES, 'unexpected capture files')
    items = {}
    for name, sha in cap['files'].items():
        path = directory / name
        require(path.is_file() and not path.is_symlink(), 'missing or linked file: '+name)
        data = path.read_bytes()
        require(digest_bytes(data) == sha, 'file digest mismatch: '+name)
        if name.endswith('.json'):
            items[name] = read_json(data)
    spec, label, m = (items[n+'.json'] for n in ('spec', 'label', 'material'))
    q = spec['native_c']
    if expected_c is not None:
        require(digest(q) == digest(expected_c) == cap['native_c_sha256'], 'wrong perturbed coefficients')
    req = spec['request']
    body = label['payload']
    proof = body['exact_proof']
    geo = dict(body['geometry'])
    seal = geo.pop('_w2_frozen_bindings')
    require(req['material_density'] == body['density'] == 1 and
            spec['reference_absolute_tolerances'] == [1e-10]*10 and
            digest(req) == seal['request_sha256'] == cap['request_sha256'] == POINT0_REQUEST and
            spec['rational_arithmetic_backend'] == 'NATIVE_GMP_SURFACE_V1', 'different frozen request')
    require(digest(body) == label['payload_sha256'] and geo == spec['geometry'] and geo['capture_run_id'] == run_id
            and digest(dict(spec, _verified_model_binding=seal['model_to_request_geometry_proof'])) == seal['spec_sha256']
            and cap['files']['frozen_normalized_outer.obj'] == spec['expected_model_sha256'] == body['model_sha256']
            and digest(spec['evaluator_settings']) == seal['evaluator_settings_sha256'] == POINT0_SETTINGS
            and body['frame'] == spec['frame'] == dict(kind='INPUT_NORMALIZED_TARGET_V1',
                native_target_origin=req['target_origin'], native_target_axes=req['target_axes']),
            'spec/label/frame mismatch')
    binding = proof['same_call_attachment']
    require(list(body['order']) == list(NAMES) and (expected_c is not None or m['derivative_defined'] is True) and
            m['derivative_identity'] == 'cherchi-constructed-vertex-motion-v1'
            and body['coefficient_convention'] == 'NATIVE_MATRIX_ENTRIES_C10_POLYNOMIAL_Q_EQUALS_BC_V1',
            'unexpected native derivative representation')
    require(digest(q) == digest(m['quadric_coefficients']) == digest(body['native_c']) == seal['native_c_sha256']
            == spec['expected_native_c_sha256'] == binding['native_c_sha256'], 'label coefficient mismatch')
    require(digest(m) == binding['serialized_material_sha256'] == cap['same_call_material_sha256'] and
            proof['native_cherchi'] is True and proof['native_stage_call_counts'] == cap['native_stage_call_counts'] and
            cap['native_stage_call_counts']['boolean'] == cap['native_stage_call_counts']['source'] == 1 and
            body['split'] == 'excluded' and body['J_D'] is None, 'native attachment mismatch')
    modules = {k:v['sha256'] for k,v in items['module-manifest.json']['extensions'].items()}
    require(modules == cap['module_sha256'] and modules['make_it_stand._make_it_stand'] == body['native_module_sha256']
            == spec['expected_native_module_sha256'] and modules['_cherchi_rational_moments'] ==
            m['rational_arithmetic_identity']['module_sha256'] == geo['gmp_backend_sha256'] and
            modules['_cherchi_core_boolean'] == geo['cherchi_bridge_sha256'] and
            m['boolean_backend_identity'] == 'cherchi-reference-7bd6c2697695f555f9a15d7e154e084d841316f0:'
            + modules['_cherchi_core_boolean'] and m['evaluation_map_identity'] == cap['evaluation_map_identity']
            and all(m[k] == proof[k] for k in ('geometry_hash', 'topology_hash', 'evaluation_map_identity',
                    'boolean_backend_identity', 'derivative_identity')), 'runtime/context mismatch')
    require(digest(body['r_F']) == POINT0_FIXED_TEN and len(body['r_D']) == 10
            and m['raw_moments'] == [r+f for r,f in zip(body['r_D'],body['r_F'])],
            'fixed-domain contribution mismatch')
    faces = m['variable_faces']
    physical = m['material_proof']['exact_vertices']
    target = m['variable_rational_vertices']
    reference = proof['independent_ten_moment_audit']
    require(digest(dict(vertices=physical, faces=faces)) == m['geometry_hash'] and
            digest(dict(vertices=target,faces=faces)) == cap['exact_boundary_sha256'] == reference['exact_boundary_sha256']
            and m['boolean_operation'] == 'subtraction' and m['quadric_source_identity'] ==
            digest(dict(vertices=m['quadric_source_vertices'],faces=m['quadric_source_faces'],coefficients=q)),
            'boundary/operand identity mismatch')
    require('normalization' not in spec and 'design_vertices_physical' not in spec, 'unexpected normalization')
    transformed = affine_target_from_physical((tuple(map(frac,p)) for p in physical),
            origin=(0,0,0), length=1, target_origin=req['target_origin'], target_axes=req['target_axes'])
    require(transformed == tuple(tuple(map(F,p)) for p in target), 'target transform mismatch')
    check_closed_indexed_surface(faces,len(target))
    moments = exact_rational_ten_moments(target, faces, body['density'])
    require(moments == [F(x) for x in reference['reference_raw_moments_exact']] and
            body['r_D'] == m['variable_raw_moments'] and len(body['r_D']) == 10 and
            all(abs(a-F(b)) <= F(1e-10) for a,b in zip(moments,body['r_D'])), 'value reintegration mismatch')
    features = normalized_features(m)
    mesh = (set(features), oriented_triangle_multiset(m,features), digest(m['quadric_source_faces']))
    info = dict(run_id=run_id, capture_sha256=capture_sha, same_call_material_sha256=cap['same_call_material_sha256'],
                exact_moments_sha256=digest([str(x) for x in moments]), vertices=len(features), faces=len(faces),
                construction_kinds=dict(Counter(key[0] for key in features)), native_c0=q[0],
                native_derivative_defined=m['derivative_defined'])
    return cap, q, moments, m['moment_jacobian'], mesh, info


def geometry_comparison(base, other):
    a, t, source = base
    b, u, next_source = other
    return dict(removed_features=len(a-b), added_features=len(b-a),
                removed_oriented_triangles=sum((t-u).values()), added_oriented_triangles=sum((u-t).values()),
                same_source_connectivity=(source==next_source),
                sampled_geometry_equal=(a==b and t==u and source==next_source))


def weights_at_zero(xs):
    """Derivative of the Lagrange interpolant at the base, using exact relative abscissae."""
    require(F(0) in xs and len(set(xs)) == len(xs), 'missing center or duplicate coefficient')
    weights = []
    for i, x in enumerate(xs):
        others = [u for j,u in enumerate(xs) if j != i]
        numerator = sum(prod(-v for v in others if v != u) for u in others)
        weights.append(F(numerator,prod(x-u for u in others)))
    return weights


def check_stencil(points, native):
    xs = [x for x,_ in points]
    weights = weights_at_zero(xs)
    exact = [sum(w*r[i] for w,(_,r) in zip(weights,points)) for i in range(10)]
    diffs = [abs(fd-j) for fd,j in zip(exact,native)]
    scaled = [err/max(1,abs(j)) for err,j in zip(diffs,native)]
    return dict(actual_offsets_exact=list(map(str,xs)), derivative_weights_exact=list(map(str,weights)),
                finite_difference=list(map(float,exact)), absolute_errors=list(map(float,diffs)),
                scaled_errors=list(map(float,scaled)), max_scaled_error=float(max(scaled)),
                passed=max(scaled)<=TOL)


def audit_column(base_data, group, group_id, source_checkout, column, planned, recorded, first_index, sample_schema,
                 *, offsets=OFFSETS):
    base, q0, m0, jac, mesh, _ = base_data
    require(type(column) is int and 0 <= column < 10, "invalid column index")
    require(len(planned) == len(recorded) == len(offsets) == 4, "incomplete column stencil")
    require(len(jac)==10 and all(len(row)==10 for row in jac), 'incomplete native diagnostic matrix')
    native=[F(row[column]) for row in jac]
    samples=[]
    points=[(F(0),m0)]
    for i,(offset,pr,rr) in enumerate(zip(offsets,planned,recorded),first_index):
        run_id=f'{group_id}-{i:03d}'
        require(pr['index']==rr['index']==i and pr['run_id']==rr['run_id']==run_id and
                pr['nominal_offset']==offset,'wrong stencil sample layout')
        if column:
            require(pr['column']==rr['column']==column, 'wrong column index')
        expected_c=q0.copy()
        expected_c[column] += offset
        cap,q,m,_,other,info=load_capture(group/run_id,run_id,rr['capture_sha256'],expected_c, sample_schema=sample_schema)
        x=F(q[column])-F(q0[column])
        require(F(pr['actual_offset_exact'])==x and digest(q)==pr['native_c_sha256']==digest(pr['native_c']) and
                cap['source_checkout']==source_checkout and
                cap['module_sha256']==base['module_sha256'] and
                cap['evaluation_map_identity']==base['evaluation_map_identity'] and
                info['same_call_material_sha256']==rr['same_call_material_sha256'], 'changed sample identity')
        if column:
            info['native_coefficient_value'] = q[column]
        info.update(geometry_comparison(mesh,other))
        points.append((x,m))
        samples.append(info)
    stencils={key:check_stencil([points[i] for i in subset],native) for key,subset in (
        ('three_point_h',(1,0,4)),('three_point_half_h',(2,0,3)),('five_point',(1,2,0,3,4)))}
    geometric=all(row['sampled_geometry_equal'] for row in samples)
    numeric=all(row['passed'] for row in stencils.values())
    return dict(perturbations=samples, native_column=list(map(float,native)), stencils=stencils,
                sampled_geometry_equal=geometric, numeric_column_passed=numeric,
                sampled_column_passed=geometric and numeric)


def audit(base_directory, group, expected_group_id):
    plan = get_json(group/'stencil.json')
    result = get_json(group/'stencil-result.json')
    require(plan['schema']=='w2-m24-c0-plan-v1' and result['schema']=='w2-m24-c0-result-v1'
            and plan['group_id']==result['group_id']==expected_group_id and
            plan['base_run_id']==BASE_ID and plan['base_capture_sha256']==BASE_CAPTURE and plan['column']==0
            and plan['h']==3e-6 and len(plan['samples'])==len(result['samples'])==4 and
            result['stencil_sha256']==digest_bytes((group/'stencil.json').read_bytes()) and
            result['fresh_native_boolean_calls']==4 and result['J_D'] is None and result['training_admitted'] is False,
            'unexpected stencil capture')
    base_data = load_capture(base_directory, BASE_ID, BASE_CAPTURE)
    base, *_, base_info = base_data
    require(base_info['exact_moments_sha256']==POINT0_EXACT_TEN and
            plan['module_sha256']==base['module_sha256'] and plan['request_sha256']==POINT0_REQUEST,
            'different base or native modules')
    column_result = audit_column(base_data, group, expected_group_id, plan['source_checkout'], 0, plan['samples'],
                                result['samples'], 1, 'w2-m24-c0-sample-v1')
    return dict(schema='w0-c0-literal-coefficient-stencil-v1', group_id=expected_group_id,
                new_native_boolean_calls_in_w2=4, native_calls_in_w0_replay=0,
                source_checkout=plan['source_checkout'], evaluation_map_identity=base['evaluation_map_identity'],
                module_sha256=base['module_sha256'], base=base_info,
                perturbations=column_result['perturbations'], moment_order=list(NAMES),
                native_column=column_result['native_column'], stencils=column_result['stencils'], scaled_error_limit=float(TOL),
                sampled_geometry_equal=column_result['sampled_geometry_equal'], numeric_column_passed=column_result['numeric_column_passed'],
                sampled_column_passed=column_result['sampled_column_passed'], continuous_neighborhood_certified=False,
                J_D=None, newton_admitted=False, training_admitted=False)


if __name__ == '__main__':
    sys.set_int_max_str_digits(100000)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--group',type=Path,required=True)
    p.add_argument('--expected-group-id',required=True)
    a=p.parse_args()
    record=audit(a.base,a.group,a.expected_group_id)
    print(json.dumps(record,indent=2,allow_nan=False))
    raise SystemExit(0 if record['sampled_column_passed'] else 2)
