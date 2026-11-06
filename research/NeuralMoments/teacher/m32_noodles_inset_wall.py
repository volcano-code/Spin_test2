"""Exact preflight for a new V7 protected-shell design mask at 9/10 linear scale."""
import argparse
from collections import Counter
from fractions import Fraction as Q
import json
from math import isqrt
from pathlib import Path

import numpy as np

from m16_native_acceptance import require
from m22_point0_operand_replay import cross, dot, sub
from m31_candidate_contact_screen import NOODLES, NOODLES_SHA, noodles_sources, read

SCALE=Q(9,10)
MM=Q(25)  # source declares model_unit_m=0.025 (25 mm per model unit)


def decimal_interval_sqrt(square, places=12):
    n=10**places
    lower=isqrt(square.numerator*n*n//square.denominator)
    return [str(Q(lower,n)),str(Q(lower+1,n))]


def supporting_wall(vertices,faces,center,scale):
    """For convex E, every point of the scaled E has each face's slack >= (1-s) times center slack."""
    require(0<scale<1, 'INVALID_INSET')
    pts=[tuple(Q(x) for x in v) for v in vertices];center=tuple(map(Q,center))
    factor=max(x.denominator for p in pts+[center] for x in p)
    require(all(factor%x.denominator==0 for p in pts+[center] for x in p),'NON_DYADIC_ENVELOPE_INPUT')
    scaled=[tuple(int(x*factor) for x in p) for p in pts]
    matrix=np.asarray(scaled,dtype=object)
    c=tuple(int(x*factor) for x in center)
    directed=Counter((a,b) for a,b,k in faces for a,b in ((a,b),(b,k),(k,a)))
    require(all(v==directed[b,a]==1 for (a,b),v in directed.items()),'OPEN_OR_INCONSISTENT_ENVELOPE')
    minimum=None; index=None
    for i,face in enumerate(faces):
        a,b,d=(scaled[j] for j in face)
        normal=cross(sub(b,a),sub(d,a))
        offset=dot(normal,a);slack=offset-dot(normal,c)
        require(any(normal) and slack>0 and all(matrix@np.asarray(normal,dtype=object)<=offset),
                'ENVELOPE_NOT_OUTWARD_CONVEX_OR_CENTER_NOT_INTERIOR')
        squared=Q(slack**2,factor**2*dot(normal,normal))*(1-scale)**2
        if minimum is None or squared<minimum:
            minimum=squared;index=i
    require(minimum is not None,'NO_ENVELOPE_FACES')
    return minimum,index


def preflight(main_tree):
    manifest=json.loads(read(main_tree,NOODLES/'frozen-parts-v7c/manifest.json',NOODLES_SHA['frozen-parts-v7c/manifest.json']))
    spec=manifest['spec']
    require(str(spec['model_unit_m'])=='0.025' and spec['target_axis']==[0,0,1] and 'pin-fixed' in manifest['construction']['fixed_parts'],
            'SOURCE_UNIT_OR_PIN_DRIFT')
    vertices,faces,bounds,sources=noodles_sources(main_tree,manifest)
    center=tuple(map(Q,spec['ellipsoid_center_model']))
    wall2,nearest=supporting_wall(vertices,faces,center,SCALE)
    # Probe inward from the known M31 envelope face; strict inequalities give an open design neighborhood.
    triangle=vertices[faces[74]]
    bary=tuple(sum(map(Q,col))/3 for col in zip(*triangle))
    point=tuple(x+Q(89,100)*(y-x) for x,y in zip(center,bary))
    box_gap=min(max(max(x-Q(y) for x,y in zip(point,hi)),max(Q(y)-x for x,y in zip(point,lo))) for lo,hi in bounds)
    require(box_gap>0,'INTERIOR_WITNESS_NOT_SEPARATED_FROM_LOCKED_MATERIAL')
    # Relative to the envelope this point has scale .89, while D' uses .90.
    design_ball2=min(box_gap**2,wall2*((SCALE-Q(89,100))/(1-SCALE))**2)
    mm2=wall2*MM**2
    return dict(schema='w2-m32-noodles-inset-wall-preflight-v1',candidate='Noodles V7 inset 9/10, new design registration',
                source_manifest_sha256=NOODLES_SHA['frozen-parts-v7c/manifest.json'],generator_sha256=sources,
                model_obj_sha256_matches=len(bounds)+1,locked_parts=len(bounds),envelope_vertices=len(vertices),
                envelope_support_planes_checked=len(faces),scale_exact=str(SCALE),center_exact_model=[str(x) for x in center],
                original_outer='A=E union F',new_variable='Dprime=(center+9/10*(E-center)) minus F',
                new_fixed='Fprime=A minus Dprime',locked_pin='pin-fixed',unchanged_registered_axis=spec['target_axis'],
                unchanged_registered_tip_model=spec['target_origin_model'],mm_per_model_unit=25,
                wall_floor_squared_exact_mm2=str(mm2),wall_floor_mm_interval=decimal_interval_sqrt(mm2),
                wall_floor_mm_approx=float(mm2)**.5,one_mm_lower_bound_proved=mm2>=1,limiting_envelope_face=nearest,
                open_design_ball_witness_exact_model=[str(x) for x in point],witness_fixed_box_gap_model=str(box_gap),
                design_ball_radius_lower_bound_squared_mm2=str(design_ball2*MM**2),
                positive_design_volume_proved=design_ball2>0,proof_scope='continuous registered polyhedral subsets before Boolean tessellation or solver',
                physical_feasibility_proved=False,exact_evaluator_registered=False,new_native_boolean_calls=0,newton_calls=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-tree',type=Path,required=True)
    print(json.dumps(preflight(parser.parse_args().main_tree),indent=2,sort_keys=True))
