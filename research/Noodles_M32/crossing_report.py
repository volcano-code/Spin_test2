"""Flux of W2's registered M32 discrete cutter, in model and solver-tip frames."""
import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

from crossing_flux import BITS, SCALE, integrate, monomials

TIP_Z=Q.from_float(-.16)
DESIGN_SHA='c3633c74bc74c2be488eb22de108454276f28941f0477cfee5837f57db99d622'
DESIGN_FACES=94150


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def packet(raw):
    rows=raw.decode('ascii').splitlines()
    nv,nf=map(int,rows[0].split())
    if nv<4 or nf<4 or len(rows)!=1+nv+nf:
        raise ValueError('malformed cutter packet')
    v=[tuple(map(Q,row.split())) for row in rows[1:1+nv]]
    f=[tuple(map(int,row.split())) for row in rows[1+nv:]]
    if any(len(p)!=3 for p in v) or any(len(t)!=3 or min(t)<0 or max(t)>=nv for t in f):
        raise ValueError('invalid triangular cutter')
    return v,f


def shifted_bounds(lo,hi):
    """The registered shift z' = z - t has positive (-t); all coefficients are nonnegative."""
    def apply(A,upper):
        out=[r.copy() for r in A]
        for j in range(len(A[0])):
            for row,value in ((3,A[3][j]-TIP_Z*A[0][j]),
                              (6,A[6][j]-2*TIP_Z*A[3][j]+TIP_Z**2*A[0][j]),
                              (7,A[7][j]-TIP_Z*A[2][j]),(8,A[8][j]-TIP_Z*A[1][j])):
                n,r=divmod(value.numerator,value.denominator)
                out[row][j]=n+int(upper and bool(r))
        return out
    return apply(lo,False),apply(hi,True)


def midpoint(lo,hi):
    return [[float(Q(a+b,2*SCALE)) for a,b in zip(x,y)] for x,y in zip(lo,hi)]


def check_source_motion(src):
    """Float64 independent bordered-system check at the represented native TIP-frame points.

    beta=0 and grad(beta)=alpha*n differentiate to four linear equations per native
    column. The scale-tangent projection changes the scalar alpha but not the point.
    """
    import numpy as np
    q=np.array(src['q'])
    x=np.array(src['vertices_tip']); n=np.array(src['normals'])
    A=np.array([[q[4],q[9],q[8]],[q[9],q[5],q[7]],[q[8],q[7],q[6]]])
    v=np.array(src['native_jacobian'])
    B=np.array([1]*7+[2]*3)
    basis_norm=np.array([np.array(monomials(p))*B for p in x])
    surface=float(abs(basis_norm@q).max())
    grad=q[1:4]+2*x@A
    normal=float(abs(np.cross(grad,n)).max())
    matrices=np.zeros((len(x),4,4));matrices[:,:3,:3]=2*A
    matrices[:,:3,3]=-n;matrices[:,3,:3]=grad
    rhs=np.zeros((len(x),4,10));rhs[:,3,:]=-basis_norm
    rhs[:,:3,1:4]=-np.eye(3)
    for k,(i,j) in enumerate(((0,0),(1,1),(2,2),(1,2),(0,2),(0,1)),start=4):
        rhs[:,i,k]-=B[k]*x[:,j]
        rhs[:,j,k]-=B[k]*x[:,i]
    recovered=np.linalg.solve(matrices,rhs)[:,:3,:]
    err=float(abs(v-recovered).max())
    rel=float((abs(v-recovered)/np.maximum(1.,abs(recovered))).max())
    if (not np.isfinite([surface,normal,err,rel]).all() or
        max(surface,normal)>1e-11 or rel>1e-8):
        raise ValueError('inverse Gauss point or native motion differs from independent bordered-system check')
    return dict(vertices=len(x),max_abs_quadric=surface,max_abs_normal_cross=normal,
                max_abs_recovered_motion_error=err,max_scaled_recovered_motion_error=rel)


def sample_flux(folder):
    """Replay this sample; provenance of actual calls/branch stability is separately checked by W2/W1."""
    root=Path(folder)
    cutter_raw=(root/'cutter.txt').read_bytes()
    src_raw=(root/'source.json').read_bytes()
    result_raw=(root/'result.json').read_bytes()
    receipt=json.loads((root/'receipt.json').read_bytes())
    if receipt['design_sha256']!=DESIGN_SHA or any(receipt[k]!=digest(raw) for k,raw in
            [('source_sha256',src_raw),('cutter_sha256',cutter_raw),('result_sha256',result_raw)]):
        raise ValueError('source or material hash differs from its W2 receipt')
    parents,triangles=packet(cutter_raw)
    src=json.loads(src_raw)
    if parents!=[tuple(map(Q,p)) for p in src['vertices']] or triangles!=list(map(tuple,src['faces'])):
        raise ValueError('cutter input and source coordinates or faces differ')
    m=[[[Q(x) for x in col] for col in v] for v in src['native_jacobian']]
    out=json.loads(result_raw)
    if out.get('succeeded') is not True or out['operation']!=receipt['operation'] or len(src['q'])!=10:
        raise ValueError('wrong Boolean or quadric source')
    if 'face_is_cap' in src:
        expected_op='intersection' if src['interior_is_positive'] else 'difference'
        cap=src['face_is_cap']
        if (len(cap)!=len(triangles) or receipt['cap_output_faces']!=0 or
            any(cap[b-DESIGN_FACES] for b in out['birth_faces'] if b>=DESIGN_FACES)):
            raise ValueError('truncation geometry used by retained boundary')
        source_check=dict(verification='general original source differential, represented velocities',
                          source_native_hash=src['source_native_hash'],triangulation_hash=src['triangulation_hash'],
                          surface_type=src['surface_type'],cap_output_faces=0)
    else:
        expected_op='intersection'
        source_check=check_source_motion(src)
    if out['operation']!=expected_op:
        raise ValueError('wrong retained side for source selector')
    vertices=[None]*len(out['vertices'])
    moving_ids={i for f,b in zip(out['faces'],out['birth_faces']) if b>=DESIGN_FACES for i in f}
    if any(i<0 or i>=len(vertices) for i in moving_ids):
        raise ValueError('invalid moving output point')
    for i in moving_ids:
        vertices[i]=tuple(map(Q,out['vertices'][i]))
    if len(m)!=len(parents) or any(len(col)!=10 for v in m for col in v):
        raise ValueError('incomplete native source Jacobian')
    lo,hi,nmoving=integrate(vertices,out['faces'],out['birth_faces'],parents,triangles,m,DESIGN_FACES,out['operation'])
    if not 0<nmoving<len(out['faces']):
        raise ValueError('registered test requires both design and cutter output')
    tlo,thi=shifted_bounds(lo,hi)
    if (nmoving!=receipt['cutter_parent_faces'] or len(out['faces'])-nmoving!=receipt['design_parent_faces']):
        raise ValueError('parent counters differ from native receipt')
    model=midpoint(lo,hi)
    tip=midpoint(tlo,thi)
    return dict(schema='w3-m35-crossing-flux-sample-v1',design_operand_sha256=DESIGN_SHA,
                source_sha256=digest(src_raw),cutter_sha256=digest(cutter_raw),result_sha256=digest(result_raw),
                q=src['q'],tip_z=str(TIP_Z),dyadic_bits=BITS,source_motion_readback=source_check,operation=out['operation'],
                moving_output_faces=nmoving,fixed_output_faces=len(out['faces'])-nmoving,
                lower_model=lo,upper_model=hi,lower_tip=tlo,upper_tip=thi,
                J_model=model,J_tip=tip,
                maximum_absolute_scaling_null=max(abs(sum(a*b for a,b in zip(row,src['q']))) for row in tip),
                scope='Polyhedral flux conditional on represented moving-parent velocities; immutable design Dprime first, cutter second.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    s=json.dumps(sample_flux(args.sample),indent=2,allow_nan=False)+'\n'
    if args.output: args.output.write_text(s)
    else: print(s,end='')
