"""Bind W2 real sampled calls, W1 exact moments and the W3 moving-source derivatives."""
import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path

from crossing_flux import SCALE
from crossing_report import DESIGN_SHA, TIP_Z, sample_flux, shifted_bounds
from crossing_vertex_motion import audit_vertices
from crossing_symmetry import tangent_skew


def read(path):
    return json.loads(path.read_bytes())


def enclose_ten(report,ten_path):
    raw=ten_path.read_bytes()
    if report['exact_ten_sha256']!=sha256(raw).hexdigest() or len(raw.splitlines())!=10:
        raise ValueError('W1 moment digest/count mismatch')
    lo=[];hi=[]
    for line in raw.splitlines():
        n,d=(int(token,0) for token in line.split(b'/'))
        x,r=divmod(n*SCALE,d)
        lo.append([x]);hi.append([x+bool(r)])
    return shifted_bounds(lo,hi)


def audit(stencil, design, w1):
    proto=read(stencil/'protocol.json')
    if (tuple(proto['columns'])!=(0,1,8) or len(set(proto['exponents']))!=len(proto['exponents']) or
        len(proto['exponents'])<2 or proto['design_sha256']!=DESIGN_SHA):
        raise ValueError('unregistered model or columns')
    def snapshot(name):
        root=stencil/name
        source=read(root/'source.json')
        receipt=read(root/'receipt.json')
        cert=read(w1/(name+'.w1.json'))
        if (cert['sample']!=name or cert['output_sha256']!=receipt['result_sha256'] or
            cert['design_sha256']!=receipt['design_sha256'] or cert['design_sha256']!=DESIGN_SHA or
            sha256((root/'result.json').read_bytes()).hexdigest()!=cert['output_sha256'] or
            sha256((root/'source.json').read_bytes()).hexdigest()!=receipt['source_sha256'] or
            sha256((root/'cutter.txt').read_bytes()).hexdigest()!=cert['cutter_sha256'] or
            cert['cutter_sha256']!=receipt['cutter_sha256']):
            raise ValueError('W2/W1 evidence mismatch')
        lo,hi=enclose_ten(cert,w1/(name+'.ten'))
        return list(map(Q,source['raw_q'])),cert,[r[0] for r in lo],[r[0] for r in hi]
    raw,base,_,_=snapshot('base')
    if not base['regularized_intersection_verified'] or not base['cutter_embedding_checked']:
        raise ValueError('W1 base set/embedding check missing')
    jac=sample_flux(stencil/'base')
    vertex=audit_vertices(stencil/'base',design,jac)
    rows=[]
    for e in proto['exponents']:
        for k in proto['columns']:
            q1,a,lo1,hi1=snapshot(f'c{k}_m{e}')
            q2,b,lo2,hi2=snapshot(f'c{k}_p{e}')
            h=Q(1,2**e)
            if any((raw[j]-q1[j],q2[j]-raw[j])!=((h,h) if j==k else (0,0)) for j in range(10)):
                raise ValueError('unequal or unregistered represented perturbation')
            err=[];scaled=[]
            for i in range(10):
                d_lo=Q(lo2[i]-hi1[i],SCALE)/(2*h)
                d_hi=Q(hi2[i]-lo1[i],SCALE)/(2*h)
                j_lo=Q(jac['lower_tip'][i][k],SCALE);j_hi=Q(jac['upper_tip'][i][k],SCALE)
                diff=max(abs(d_lo-j_hi),abs(d_hi-j_lo))
                err.append(float(diff))
                # Denominator conservatively lower-bounds max(1,|analytic|).
                amin=(0 if j_lo<=0<=j_hi else min(abs(j_lo),abs(j_hi)))
                scaled.append(float(diff/max(1,amin)))
            agreement={key: a['construction'][key]==base['construction'][key]==b['construction'][key]
                       for key in ('construction_sha256','seam_sha256','oriented_triangulation_sha256')}
            agreeids=all(r['construction']['unambiguous_vertex_stars'] for r in (a,b,base))
            rows.append(dict(column=k,absolute_step=str(h),all_vertices_unambiguous=agreeids,
                             signature_agreement=agreement,exact_moment_hash_minus=a['exact_ten_sha256'],
                             exact_moment_hash_plus=b['exact_ten_sha256'],absolute_errors=err,
                             maximum_absolute_error=max(err),maximum_scaled_error=max(scaled),
                             within_1e_minus_6=max(scaled)<=1e-6,
                             strict_sample_pass=agreeids and all(agreement.values()) and max(scaled)<=1e-6))
    return dict(schema='w3-m35-exact-rational-crossing-derivative-v1',retained='Dprime intersection cutter',
                moment_frame='tip centered in model units, native columns also in tip frame',
                base_sample=jac,vertex_motion_crosscheck=vertex,tangent_model_compatibility=tangent_skew(jac),
                finite_differences=rows,
                sampled_columns=proto['columns'],checked_components=10*len(rows),admission_tolerance=1e-6,
                strict_sampled_comparisons_passed=sum(row['strict_sample_pass'] for row in rows),
                native_newton_terminal=False,training_admitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stencil',type=Path);p.add_argument('design',type=Path);p.add_argument('w1',type=Path)
    a=p.parse_args()
    print(json.dumps(audit(a.stencil,a.design,a.w1),indent=2,allow_nan=False))
