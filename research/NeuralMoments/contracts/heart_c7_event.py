"""Replay six Heart native cross-term samples; fit two disjoint endpoint stencils."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import sys

from .heart_c0_column import (BASE_ID, BASE_CAPTURE, POINT0_REQUEST, NAMES, TOL,
    check_stencil, digest, digest_bytes, geometry_comparison, get_json, load_capture, require)
from .heart_point0_seams import cross, dot, sub

COORDINATES = (
    ('left', 'coarse', '0x1.24ac5f4eb5eaap-19'),
    ('left', 'mid',    '0x1.2ab68f7e53921p-19'),
    ('left', 'near',   '0x1.30c0bfadf1398p-19'),
    ('right','near',   '0x1.336ff18a374b0p-19'),
    ('right','mid',    '0x1.397a21b9d4f27p-19'),
    ('right','coarse', '0x1.3f8451e97299ep-19'),
)


def edge(mesh, faces, ids, neighbors):
    a,b=ids
    adjacent = [sorted(face) for face in faces if a in face and b in face]
    require(sorted(adjacent)==sorted(neighbors), 'event edge neighbors changed')
    return tuple(tuple(map(F,mesh[i])) for i in ids)


def signed_six_volume(a,b,c,d):
    return dot(cross(sub(b,a),sub(c,a)),sub(d,a))


def side_check(points, anchor):
    j = points[anchor]['native']
    x = points[anchor]['x']
    fd = check_stencil([(p['x']-x, p['moments']) for p in points],j)
    geometry = [geometry_comparison(points[anchor]['mesh'],p['mesh']) for p in points]
    return dict(anchor_run_id=points[anchor]['info']['run_id'],
                sampled_run_ids=[p['info']['run_id'] for p in points],
                native_column=list(map(float,j)), difference=fd,
                relative_to_anchor=geometry, numerically_passed=fd['passed'],
                same_sampled_branch=all(g['sampled_geometry_equal'] for g in geometry))


def audit(base_dir, group, group_id, public_result_sha256):
    plan=get_json(group/'stencil.json')
    raw=(group/'stencil-result.json').read_bytes()
    require(digest_bytes(raw)==public_result_sha256,'public W2 result digest mismatch')
    result=json.loads(raw)
    require(plan['schema']=='w2-m26-c7-plan-v1' and result['schema']=='w2-m26-c7-result-v1'
            and plan['group_id']==result['group_id']==group_id
            and plan['base_run_id']==BASE_ID and plan['base_capture_sha256']==BASE_CAPTURE
            and plan['column']==7 and len(plan['samples'])==len(result['samples'])==6
            and result['stencil_sha256']==digest_bytes((group/'stencil.json').read_bytes())
            and result['fresh_native_boolean_calls']==6 and result['J_D'] is None
            and result['training_admitted'] is False, 'unexpected c7 evidence group')
    base,q0,_,_,base_mesh,_ = load_capture(base_dir,BASE_ID,BASE_CAPTURE)
    require(q0[7]==0 and plan['module_sha256']==base['module_sha256'] and
            plan['request_sha256']==POINT0_REQUEST,'baseline or modules changed')
    req = get_json(base_dir/'spec.json')['request']
    d,e = edge(req['design_vertices'],req['design_faces'],(5517,5518),
               [[5516,5517,5518],[5517,5518,5523]])
    samples=[]
    for i,((side,location,hex_c7),pr,rr) in enumerate(zip(COORDINATES,plan['samples'],result['samples']),1):
        coord=float.fromhex(hex_c7)
        expected=q0.copy()
        expected[7]=coord
        run_id=f'{group_id}-{i:03d}'
        require(pr['index']==rr['index']==i and pr['run_id']==rr['run_id']==run_id and
                pr['side']==rr['side']==side and pr['location']==rr['location']==location
                and pr['native_c7_hex']==hex_c7 and pr['column']==rr['column']==7
                and digest(pr['native_c'])==pr['native_c_sha256']==digest(expected)
                and F(pr['actual_offset_exact'])==F(coord),'c7 layout or abscissa mismatch')
        directory=group/run_id
        cap,q,mom,jac,mesh,info=load_capture(directory,run_id,rr['capture_sha256'],expected,
                                              sample_schema='w2-m26-c7-sample-v1')
        require(cap['source_checkout']==plan['source_checkout'] and cap['module_sha256']==base['module_sha256']
                and cap['evaluation_map_identity']==base['evaluation_map_identity'] and
                cap['same_call_material_sha256']==rr['same_call_material_sha256']
                and len(jac)==10 and all(len(row)==10 for row in jac) and info['native_derivative_defined'],
                'c7 source, runtime, or derivative unavailable')
        # load_capture has bound both source triangle arrays and c to this same-call material.
        m=get_json(directory/'material.json')
        body=get_json(directory/'label.json')['payload']
        require(body['polynomial_q']==q[:7]+[2*v for v in q[7:]], 'cross-term factor mismatch')
        a,b=edge(m['quadric_source_vertices'],m['quadric_source_faces'],(24,90),
                 [[24,89,90],[24,90,151]])
        v=signed_six_volume(a,b,d,e)
        sign=(v>0)-(v<0)
        info.update(c7_hex=hex_c7,native_c7=q[7],edge_six_volume_exact=str(v),edge_sign=sign,
                    relative_to_M23_base=geometry_comparison(base_mesh,mesh))
        samples.append(dict(info=info,x=F(q[7]),moments=mom,mesh=mesh,native=[F(row[7]) for row in jac]))
    before=side_check(samples[:3],2)
    after=side_check(samples[3:],0)
    across=geometry_comparison(samples[2]['mesh'],samples[3]['mesh'])
    signs=[s['info']['edge_sign'] for s in samples]
    bracketed=signs==[1,1,1,-1,-1,-1] and not across['sampled_geometry_equal']
    passed=(bracketed and before['same_sampled_branch'] and after['same_sampled_branch']
            and before['numerically_passed'] and after['numerically_passed'])
    return dict(schema='w0-heart-c7-fresh-one-sided-v1',group_id=group_id,
                source_checkout=plan['source_checkout'],public_result_sha256=public_result_sha256,
                evaluation_map_identity=base['evaluation_map_identity'],column=7, moment_order=list(NAMES),
                native_boolean_calls_in_w2=6,native_calls_in_w0_replay=0,
                sampled_branch_change_bracket=([float(samples[2]['x']), float(samples[3]['x'])] if bracketed else None),
                samples=[s['info'] for s in samples], between_near_points=across,
                left_endpoint=before,right_endpoint=after, scaled_error_limit=float(TOL),
                sampled_event_and_endpoint_checks_passed=passed,
                event_one_sided_limits=None, continuous_neighborhood_certified=False,
                J_D=None,newton_admitted=False,training_admitted=False)


if __name__=='__main__':
    sys.set_int_max_str_digits(100000)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--group',type=Path,required=True)
    p.add_argument('--expected-group-id',required=True)
    p.add_argument('--result-sha256',required=True,help='SHA-256 of the runner public stencil-result.json')
    a=p.parse_args()
    report=audit(a.base,a.group,a.expected_group_id,a.result_sha256)
    print(json.dumps(report,indent=2,allow_nan=False))
    raise SystemExit(0 if report['sampled_event_and_endpoint_checks_passed'] else 2)
