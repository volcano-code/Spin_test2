"""Conditional homothetic source motion for the saved isotropic Heart c[0] base."""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path

from h9_m21_motion_math import (add, compare_vectors, dot, material_state, moment_tangent,
                                mul, need, sub, vec)

BASE_CAPTURE = '816fc58eda9b20a65efccff00b3b803b2870315d9422957d1d141dbe6f68da04'


def checked_json(path, expected):
    data = path.read_bytes()
    need(sha256(data).hexdigest() == expected, 'EVIDENCE_DIGEST_MISMATCH:'+path.name)
    return json.loads(data)


def sphere(q, request):
    # Target axes are the identity for this experiment. Physical coordinates
    # have the additional translation by target_origin with unit mesh scale.
    need(request['target_axes'] == [[1,0,0],[0,1,0],[0,0,1]], 'IDENTITY_TARGET_AXES_REQUIRED')
    c = list(map(F,q))
    need(len(c)==10,'TEN_COEFFICIENTS_REQUIRED')
    a = c[4]
    need(a>0 and c[4:7]==[a]*3 and c[7:]==[0]*3, 'ISOTROPIC_SPHERE_REQUIRED')
    center_target = mul(-1/(2*a), c[1:4])
    radius2 = dot(center_target,center_target)-c[0]/a
    need(radius2>0,'EMPTY_SPHERE')
    return add(vec(request['target_origin']),center_target), radius2, a


def source_rays(material, center):
    return [sub(vec(p),center) for p in material['quadric_source_vertices']]


def perturbed_mesh_checks(root, material, capture, request, center, radius2, base_rays):
    """Compare the ideal scaling against every published fresh source vertex."""
    result=json.loads((root/'stencil-result.json').read_bytes())
    plan=checked_json(root/'stencil.json',result['stencil_sha256'])
    need(result['group_id']==plan['group_id'] and plan['base_capture_sha256']==BASE_CAPTURE
         and plan['column']==0 and plan['h']==3e-6 and result['fresh_native_boolean_calls']==4 and
         [p['nominal_offset'] for p in plan['samples']]==[-3e-6,-1.5e-6,1.5e-6,3e-6] and
         [p['index'] for p in plan['samples']]==[1,2,3,4] and len(result['samples'])==4,'WRONG_STENCIL')
    out=[]
    for point,row in zip(plan['samples'],result['samples']):
        need(row['run_id']==point['run_id'] and row['index']==point['index']
             and Path(row['run_id']).name==row['run_id'],'WRONG_SAMPLE_ID')
        d=root/point['run_id']; record=checked_json(d/'capture.json',row['capture_sha256'])
        need(record['run_id']==row['run_id'] and record['module_sha256']==capture['module_sha256']
             and record['request_sha256']==capture['request_sha256'], 'SAMPLE_BASE_MISMATCH')
        m=checked_json(d/'material.json',record['files']['material.json'])
        q=m['quadric_coefficients']
        need(q==point['native_c'] and q[1:]==material['quadric_coefficients'][1:]
             and F(q[0])-F(material['quadric_coefficients'][0])==F(point['actual_offset_exact'])
             and m['quadric_source_faces']==material['quadric_source_faces'], 'C0_ONLY_SAME_SOURCE_FACES_REQUIRED')
        ctr,r2,_=sphere(q,request);need(ctr==center,'CENTER_CHANGED')
        rays=source_rays(m,center);need(len(rays)==len(base_rays),'SOURCE_VERTEX_COUNT_CHANGED')
        def dec(x):return Decimal(x.numerator)/Decimal(x.denominator)
        with localcontext() as ctx:
            ctx.prec=60
            scale=dec(r2/radius2).sqrt()
            residual=max(abs(dec(v)-scale*dec(u)) for rb,rs in zip(base_rays,rays) for u,v in zip(rb,rs))
        out.append(dict(run_id=row['run_id'], source_vertices=len(rays),
                        max_source_coordinate_residual_against_ideal_scale=float(residual)))
    return out


def replay(root, stencil=None):
    capture=checked_json(root/'capture.json',BASE_CAPTURE)
    spec=checked_json(root/'spec.json',capture['files']['spec.json'])
    material=checked_json(root/'material.json',capture['files']['material.json'])
    req=spec['request'];q=material['quadric_coefficients']
    need(spec['native_c']==q and spec.get('normalization') is None and spec.get('design_vertices_physical') is None
         and material['boolean_operation']=='subtraction' and req['material_density']==1
         and spec['evaluator_settings']['retain_fixed_normal_provenance'] is True,'BASE_GEOMETRY_MISMATCH')
    center,radius2,a=sphere(q,req)
    rays=source_rays(material,center)
    # Continuation through the rounded stored vertices, fixing each ray, with
    # analytic scale r(c0)/r(base). No finite differences or native J in velocities.
    factor=-1/(2*a*radius2)
    state=material_state(material,req,[mul(factor,r) for r in rays])
    tangent,active=moment_tangent(state,req['material_density'])
    native=[row[0] for row in material['moment_jacobian']]
    comparison=compare_vectors(tangent,native)
    result=dict(schema='w3-heart-c0-conditional-radial-v1',base_capture_sha256=BASE_CAPTURE,
                center_model=list(map(float,center)),radius_squared=float(radius2),source_vertices=len(rays),
                maximum_radius_squared_residual=float(max(abs(dot(r,r)-radius2) for r in rays)),
                constructions=state['kinds'],nonzero_motion_faces=active,native_column=0,
                comparison=comparison,
                max_abs_column_difference=max(c['absolute_difference'] for c in comparison['components'].values()),
                native_calls=0,continuous_neighborhood_certified=False,J_D=None,newton_admitted=False)
    if stencil is not None:
        result['fresh_source_vertex_checks']=perturbed_mesh_checks(stencil,material,capture,req,center,radius2,rays)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('--stencil-dir', type=Path)
    args=parser.parse_args()
    print(json.dumps(replay(args.baseline,args.stencil_dir),indent=2))
