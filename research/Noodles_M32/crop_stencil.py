"""Compare the full native Jacobian to an equivalent 41-query exact cropped-design stencil."""
import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path

from crossing_flux import SCALE
from crossing_stencil import read, enclose_ten


def audit(crop, w1, exact, primary):
    protocol=read(crop/'protocol.json')
    baseline=read(primary)
    J=baseline['base_sample']
    if (protocol['columns']!=list(range(10)) or protocol['exponents']!=[19,20] or
        protocol['design_sha256']!=J['design_operand_sha256']):
        raise ValueError('unregistered supplemental stencil')
    direct={}
    for row in baseline['finite_differences']:
        exp=Q(row['absolute_step']).denominator.bit_length()-1
        for sign,key in [('p','plus'),('m','minus')]:
            direct[f"c{row['column']}_{sign}{exp}"]=row['exact_moment_hash_'+key]
    base_construction=read(w1/'base.json')['construction']
    rows=[]
    direct['base']=None
    def load(name):
        root=crop/name
        cert=read(w1/(name+'.json'));r=read(root/'receipt.json')
        if (not cert['exact_ten_inside_all_bounds'] or
            cert['source_position']['zero_feature_predicates']!=0 or
            not cert['source_position']['locally_transverse_polyhedra'] or
            cert['construction']!=base_construction or not base_construction['unambiguous_vertex_stars'] or
            not r['original_parent_remapped'] or r['operation']!='intersection' or
            r['crop_operand_sha256']!=protocol['crop_sha256'] or
            r['design_sha256']!=cert['design_sha256'] or r['design_sha256']!=J['design_operand_sha256'] or
            r['csg_executable']!=protocol['executor_sha256'] or
            r['source_executable']!=protocol['source_sha256'] or
            cert['sample']!=name or bool(r['direct_oriented_face_and_parent_match'])!=(name in direct)):
            raise ValueError('crop equivalence, contact or construction evidence differs')
        for filename,key in [('source.json','source_sha256'),('result.json','result_sha256'),
                             ('crop-result.json','crop_raw_output_sha256'),('cutter.txt','cutter_sha256')]:
            if r[key]!=cert[key] or r[key]!=sha256((root/filename).read_bytes()).hexdigest():
                raise ValueError('mismatched original-parent map, cropped result or source')
        if name in direct and direct[name] is not None and direct[name]!=cert['exact_ten_sha256']:
            raise ValueError('independent full-design and equivalent crop moments differ')
        if name=='base' and (r['cutter_sha256']!=J['cutter_sha256'] or r['source_sha256']!=J['source_sha256']):
            raise ValueError('derivative belongs to a different base point')
        lo,hi=enclose_ten(cert,exact/(name+'.ten'))
        return list(map(Q,read(root/'source.json')['raw_q'])),cert,lo,hi
    center,_,_,_=load('base')
    for exp in protocol['exponents']:
        h=Q(1,2**exp)
        for j in protocol['columns']:
            l,lc,llo,lhi=load(f'c{j}_m{exp}')
            u,uc,ulo,uhi=load(f'c{j}_p{exp}')
            for k in range(10):
                step=h if k==j else 0
                if center[k]-l[k]!=step or u[k]-center[k]!=step:
                    raise ValueError('changed raw coefficient or unequal perturbation')
            errors=[]
            for i in range(10):
                dl=Q(ulo[i][0]-lhi[i][0],SCALE)/(2*h)
                dh=Q(uhi[i][0]-llo[i][0],SCALE)/(2*h)
                jl=Q(J['lower_tip'][i][j],SCALE);jh=Q(J['upper_tip'][i][j],SCALE)
                amin=0 if jl<=0<=jh else min(abs(jl),abs(jh))
                errors.append(max(abs(dl-jh),abs(dh-jl))/max(1,amin))
            rows.append(dict(column=j,step=str(h),max_scaled_error=float(max(errors)),
                             pass_1e_minus_6=max(errors)<=Q(1,10**6),
                             moment_hash_minus=lc['exact_ten_sha256'],moment_hash_plus=uc['exact_ten_sha256']))
    return dict(schema='w3-m35-all-native-columns-equivalent-crop-v1',
                primary_derivative_sha256=sha256(primary.read_bytes()).hexdigest(),crop_protocol=protocol,
                base_original_parent_construction=base_construction,direct_full_design_comparisons=len(direct),
                distinct_exact_queries=41,native_columns=10,steps=2,scalar_comparisons=200,
                comparisons_passing=sum(row['pass_1e_minus_6'] for row in rows),comparisons=rows,
                maximum_scaled_error=max(r['max_scaled_error'] for r in rows),
                scope='Dyadic tip moment/J intervals, strict features and matching construction at sampled points.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('crop','w1','exact','primary'):p.add_argument(key,type=Path)
    a=p.parse_args();print(json.dumps(audit(a.crop,a.w1,a.exact,a.primary),indent=2,allow_nan=False))
