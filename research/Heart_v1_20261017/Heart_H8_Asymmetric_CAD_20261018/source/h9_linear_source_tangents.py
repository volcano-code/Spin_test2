"""Conditional center translation and radial motion of the M23 sphere for b_x and b_y."""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
import json
from pathlib import Path

from h9_c0_radial_diagnostic import BASE_CAPTURE, checked_json, sphere, source_rays
from h9_m21_motion_math import add, compare_vectors, material_state, moment_tangent, mul, need


def motion(q, column, a, r2):
    need(column in (1, 2), 'EXPECTED_LINEAR_COLUMN')
    # The moving origin is in model space; fixed target axes are the identity.
    center_velocity = tuple(-F(1, 2*a) if i == column-1 else F(0) for i in range(3))
    return center_velocity, F(q[column])/(4*a*a*r2)


def side_checks(root, receipt_sha, capture, material, request, base_rays, r2):
    result = checked_json(root/'stencil-result.json', receipt_sha)
    plan = checked_json(root/'stencil.json', result['stencil_sha256'])
    need(result['group_id']==root.name==plan['group_id'] and plan['columns']==[1,2] and
         plan['base_capture_sha256']==BASE_CAPTURE and len(plan['samples'])==len(result['samples'])==
         result['fresh_native_boolean_calls']==8 and plan['h']==3e-6, 'WRONG_STENCIL')
    rows = []
    for i, (point, saved) in enumerate(zip(plan['samples'],result['samples'])):
        col = (1, 2)[i//4]
        run = f'{root.name}-{i+1:03d}'
        need(i+1==point['index']==saved['index'] and run==point['run_id']==saved['run_id'] and
             col==point['column']==saved['column'], 'WRONG_SAMPLE')
        sample = checked_json(root/run/'capture.json', saved['capture_sha256'])
        need(sample['run_id']==run and sample['request_sha256']==capture['request_sha256'] and
             sample['module_sha256']==capture['module_sha256'], 'SAMPLE_CONTEXT_CHANGED')
        side = checked_json(root/run/'material.json', sample['files']['material.json'])
        expected = material['quadric_coefficients'].copy()
        expected[col] += (-3e-6,-1.5e-6,1.5e-6,3e-6)[i%4]
        need(side['quadric_coefficients']==point['native_c']==expected and
             side['quadric_source_faces']==material['quadric_source_faces'], 'WRONG_COEFFICIENT_OR_FACETS')
        ctr, side_r2, _ = sphere(expected, request)
        rays = source_rays(side, ctr)
        need(len(rays)==len(base_rays), 'SOURCE_VERTEX_COUNT_CHANGED')
        def dec(x):
            return Decimal(x.numerator)/Decimal(x.denominator)
        with localcontext() as ctx:
            ctx.prec=60
            scale=dec(side_r2/r2).sqrt()
            residual=max(abs(dec(v)-scale*dec(u)) for orig,new in zip(base_rays,rays) for u,v in zip(orig,new))
        rows.append(dict(run_id=run, column=col, source_vertices=len(rays), maximum_coordinate_residual=float(residual)))
    return rows


def replay(base, group, receipt_sha):
    cap=checked_json(base/'capture.json',BASE_CAPTURE)
    material=checked_json(base/'material.json',cap['files']['material.json'])
    spec=checked_json(base/'spec.json',cap['files']['spec.json'])
    q=material['quadric_coefficients']
    req=spec['request']
    need(q==spec['native_c'] and req['material_density']==1 and material['boolean_operation']=='subtraction'
         and spec.get('normalization') is None and spec.get('design_vertices_physical') is None
         and spec['evaluator_settings']['retain_fixed_normal_provenance'] is True,'WRONG_BASE_SETTINGS')
    center,r2,a=sphere(q,req)
    rays=source_rays(material,center)
    columns={}
    for col in (1,2):
        dcenter,rate=motion(q,col,a,r2)
        velocities=[add(dcenter,mul(rate,ray)) for ray in rays]
        state=material_state(material,req,velocities)
        values,active=moment_tangent(state,req['material_density'])
        comparison=compare_vectors(values,[row[col] for row in material['moment_jacobian']])
        columns[str(col)]=dict(center_velocity=list(map(float,dcenter)), relative_radius_velocity=float(rate),
                               nonzero_motion_faces=active, constructions=state['kinds'], comparison=comparison,
                               maximum_absolute_error=max(v['absolute_difference'] for v in comparison['components'].values()))
    return dict(schema='w3-heart-conditional-linear-motion-v1', base_capture_sha256=BASE_CAPTURE,
                source_vertex_count=len(rays), columns=columns,
                fresh_source_checks=side_checks(group,receipt_sha,cap,material,req,rays,r2),
                continuous_neighborhood_certified=False,J_D=None,newton_admitted=False,native_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('base', type=Path)
    p.add_argument('group', type=Path)
    p.add_argument('result_sha256')
    args=p.parse_args()
    print(json.dumps(replay(args.base,args.group,args.result_sha256),indent=2,allow_nan=False))
