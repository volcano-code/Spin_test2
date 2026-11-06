"""Continuous ten moments for a wholly interior ellipsoid cavity, in the model frame."""
import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

PAIRS = [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]
W4_CERT = 'af14f1d246a07e391e08a549be4c49d52d9eaf7d249aad0ec5a23a1b49fabdb6'
SEPARATORS = '4be661618a5ac1543ad9006c9d0c19717f8b5ed63732864367bbf7debf15041c'
D_HASH = 'c3633c74bc74c2be488eb22de108454276f28941f0477cfee5837f57db99d622'
A_HASH = '5912e9f1dc44378919f5c00690983d284189edadc857ea3b354f09c4b86a1986'
H = Q(1, 2**30)


def read_bytes(path, digest):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError('source hash mismatch: '+path.name)
    return raw


def parts(q):
    k = np.array([[q[4],q[9],q[8]],[q[9],q[5],q[7]],[q[8],q[7],q[6]]])
    return q[0], np.array(q[1:4]), k


def ellipsoid(q):
    a, b, k = parts(q)
    adj = np.column_stack([np.cross(k[(i+1)%3], k[(i+2)%3]) for i in range(3)])
    det = k[0] @ adj[:,0]
    if min(k[0,0], adj[2,2], det) <= 0:
        raise ValueError('positive definite quadratic required')
    inv = adj/det
    c = -(inv @ b)/2
    level = -a - (b@c)/2
    if level <= 0:
        raise ValueError('empty ellipsoid')
    return c, level*inv, level, inv, level**3/det


def monomials(x):
    return np.array([1, *x, *(x[i]*x[j] for i,j in PAIRS)])


def enclosure(q, h, dp, hints, inside):
    # For |dq_i| <= h, |beta_dq(x)| <= h*(3+4|x|²), by 2|xy| <= x²+y².
    change = np.array([3,0,0,0,4,4,4,0,0,0])*h
    c, shape, level, _, _ = ellipsoid(q-change)
    metric=parts(q-change)[2]/level
    weights = monomials(inside)
    weights[7:] *= 2
    if weights @ (q+change) >= 0:
        raise ValueError('W4 interior point must remain in every cavity')
    lines = dp.decode().splitlines()
    nv,nf = map(int,lines[0].split())
    vertices = [np.array(list(map(Q,line.split()))) for line in lines[1:1+nv]]
    faces = [tuple(map(int,line.split())) for line in lines[1+nv:]]
    if len(faces) != nf:
        raise ValueError('incomplete boundary')
    masks = []
    for v in vertices:
        d = v-c
        mask = 0
        for j in range(3):
            if d[j]*d[j] > shape[j,j]:
                mask |= (1 if d[j] < 0 else 2) << (2*j)
        masks.append(mask)
    count=0
    for i,(x,y,z) in enumerate(faces):
        if masks[x] & masks[y] & masks[z]:
            continue
        n = hints.get(i)
        if n is None:
            tri = [vertices[j]-c for j in (x,y,z)]
            candidates = list(tri)
            for v,w in zip(tri,tri[1:]+tri[:1]):
                edge=w-v
                t=max(0,min(1,-(edge@metric@v)/(edge@metric@edge)))
                candidates.append(v+t*edge)
            e1,e2=tri[1]-tri[0],tri[2]-tri[0]
            aa,ab,bb=e1@metric@e1,e1@metric@e2,e2@metric@e2
            rhs1,rhs2=-(e1@metric@tri[0]),-(e2@metric@tri[0])
            det=aa*bb-ab*ab
            u,v=(bb*rhs1-ab*rhs2)/det,(aa*rhs2-ab*rhs1)/det
            if u>=0 and v>=0 and u+v<=1:
                candidates.append(tri[0]+u*e1+v*e2)
            closest=min(candidates,key=lambda v:v@metric@v)
            n=metric@closest
        n = np.array(n,dtype=object)
        d = min(n @ vertices[j] for j in (x,y,z)) - n @ c
        if d <= 0 or d*d <= n @ shape @ n:
            raise ValueError(f'wrapper intersects supporting plane of face {i}')
        count += 1
    return dict(native_linf_radius=str(h), strict_AABB_faces=nf-count, strict_support_faces=count,
                inside_point_source='W4 authenticated nominal center, winding 1')


def values(q):
    c,g,level,inv,detg = ellipsoid(q)
    v = (4*math.pi/3)*math.sqrt(float(detg))
    means = [1,*c,*(c[i]*c[j]+g[i,j]/5 for i,j in PAIRS)]
    return v*np.array(means,dtype=float)


def jacobian(q):
    c,g,level,inv,detg=ellipsoid(q)
    v=(4*math.pi/3)*math.sqrt(float(detg))
    mu = np.array([1,*c,*(c[i]*c[j]+g[i,j]/5 for i,j in PAIRS)],dtype=object)
    cols=[]
    for e in np.eye(10,dtype=int):
        da,db,dk=parts(e)
        dc=-inv@(dk@c+db*Q(1,2))
        dl=-da-db@c-c@dk@c
        dg=dl*inv-level*inv@dk@inv
        dvlog=(3*dl/level-np.trace(inv@dk))/2
        dmu=np.array([0,*dc,*(dc[i]*c[j]+c[i]*dc[j]+dg[i,j]/5 for i,j in PAIRS)])
        cols.append(-v*np.array(dvlog*mu+dmu,dtype=float))
    return np.array(cols).T


def surface_j(q):
    c,g,k,_,detg=ellipsoid(q)
    L=np.linalg.cholesky(np.array(g,dtype=float))
    gram=np.zeros((10,10))
    for z,w in zip(*np.polynomial.legendre.leggauss(4)):
        radial=math.sqrt(1-z*z)
        for a in range(12):
            t=2*math.pi*a/12
            p=np.array(c,dtype=float)+L@np.array([radial*math.cos(t),radial*math.sin(t),z])
            m=monomials(p)
            gram+=np.outer(m,m)*(2*math.pi*w/12)
    gram[:,7:]*=2
    return gram*(math.sqrt(float(detg))/(2*float(k)))


def exact_balance(q, payload):
    # Enclose the 9-MB W1 rationals by truncation, without large GCD operations.
    s = 1 << 160
    def bound_a(raw):
        a,b = (int(v,0) for v in raw.split(b'/'))
        l,r = divmod(a*s,b)
        return Q(l,s),Q(l+bool(r),s)
    whole = list(map(bound_a,payload.splitlines()))
    def atan_range(x):
        n = 32
        lo = sum(Q((-1)**k,(2*k+1)*x**(2*k+1)) for k in range(n))
        return lo, lo + Q(1,(2*n+1)*x**(2*n+1))
    aa,bb = atan_range(5),atan_range(239)
    pi = 16*aa[0]-4*bb[1],16*aa[1]-4*bb[0]
    c,g,_,_,gdet = ellipsoid(q)
    k = math.isqrt(gdet.numerator*s*s//gdet.denominator)
    volume = Q(4,3)*pi[0]*Q(k,s),Q(4,3)*pi[1]*Q(k+1,s)
    means = [1,*c,*(c[i]*c[j]+g[i,j]/5 for i,j in PAIRS)]
    retained = [(a-max(v*m for v in volume),b-min(v*m for v in volume))
                for (a,b),m in zip(whole,means)]
    mass = retained[0][0]
    if mass <= 0:
        raise ValueError('nonpositive retained mass')
    v = [max(abs(a),abs(b)) for a,b in retained]
    upper = [v[1],v[2],v[8]+v[1]*v[3]/mass,v[7]+v[2]*v[3]/mass]
    digits = 10**20
    ceil = lambda v: str(Q(-((-v*digits)//1),digits))
    izfloor = (retained[4][0]+retained[5][0])/(whole[4][1]+whole[5][1])
    return dict(physical_constraint_abs_upper=[ceil(v) for v in upper],
                axis_inertia_ratio_floor=str(Q((izfloor*digits)//1,digits)),
                retained_mass_ratio_floor=str(Q(((mass/whole[0][1])*digits)//1,digits)),
                exact_stationarity_checked=False)

def historical_profile(q, r):
    # V7's pilot passes already-normalized model vertices and weight 1.
    # 0.025 m per model unit is metadata. The target coordinates subtract
    # the registered binary64 tip (identity axes).
    tip = Q(-0.16)
    o = np.array([Q(0),Q(0),tip])
    a,b,k = parts(q)
    coeff = list(q)
    coeff[:4] = [a+b@o+o@k@o, *(b+2*k@o)]
    dual = np.array(coeff,dtype=float)
    dual[7:] *= 2
    dual /= np.linalg.norm(dual)
    t = float(tip)
    target = r.copy()
    target[3] -= t*r[0]
    target[6] += -2*t*r[3]+t*t*r[0]
    target[7] -= t*r[2]
    target[8] -= t*r[1]
    def physics(m):
        v,x,y,z,xx,yy,zz,yz,xz,xy = m
        i1,i2,i3 = yy+zz-(y*y+z*z)/v, xx+zz-(x*x+z*z)/v, xx+yy-(x*x+y*y)/v
        i12 = -xy+x*y/v
        return np.array([(i1*i1+i2*i2+2*i12*i12)/(i3*i3)+z/v,
                         x,y,-xz+x*z/v,-yz+y*z/v])
    ds = np.column_stack([physics(target+e*1e-25j).imag/1e-25 for e in np.eye(10)])
    proj=np.column_stack((-ds[1:].T,dual))
    lm,_,rank,_=np.linalg.lstsq(proj,-ds[0],rcond=None)
    value=physics(target)
    residual=np.linalg.norm(np.r_[ds[0]+proj@lm,value[1:]])
    return dict(profile='V7 historical SPINNING_TOP InertiaAndLowering / DUAL_MONOMIAL_RAW_V1',
                units='model (0.025 m/unit), identity axes and z_tip=-0.16, unit density',
                center_of_mass_weight=1.0, objective=float(value[0]), multiplier_mu=float(lm[-1]),
                projection_rank=int(rank),raw_equation_residual=float(residual),
                continuum_positive_initial_multiplier=bool(lm[-1]>0), native_initializer_observed=False)

def audit(w4,design,full_ten):
    cert=json.loads(read_bytes(w4/'M32_SINGLE_ELLIPSOID.json',W4_CERT))
    hints=[list(map(int,l.split())) for l in read_bytes(w4/'M32_ELLIPSOID_SEPARATORS.txt',SEPARATORS).splitlines()]
    payload=read_bytes(full_ten,A_HASH)
    full=np.array([int(x,16)/int(y,16) for x,y in (l.split(b'/') for l in payload.splitlines())])
    if len(full)!=10 or cert['ray_winding']!=1:
        raise ValueError('invalid bound source')
    q=np.array(list(map(Q,cert['coefficient_hint_model_normalized_native_order'])),dtype=object)
    guard=enclosure(q,H,read_bytes(design,D_HASH),{i:row for i,*row in hints},np.array(list(map(Q,cert['nominal_center_model']))))
    J=jacobian(q)
    sj=surface_j(q)
    checks=[]
    for h in (H,H/2,H/4):
        fd=np.column_stack([(values(q-e*h)-values(q+e*h))/(2*float(h)) for e in np.eye(10,dtype=int)])
        checks.append(dict(h=str(h),columns=10,max_absolute_error=float(np.max(abs(fd-J)))))
    err=float(np.max(abs(sj-J)))
    if max(err,*(r['max_absolute_error'] for r in checks))>1e-6:
        raise ValueError('independent Jacobian tolerance exceeded')
    retained=full-values(q)
    return dict(schema='W3_M32_INTERIOR_ROUNDED_QUADRIC_CONTINUUM_V1',source_w4_certificate_sha256=W4_CERT,
                full_ten_source_sha256=A_HASH,exact_design_operand_sha256=D_HASH,containment=guard,
                coordinate_frame='original model xyz; outside beta>=0; ambient unnormalized coefficient perturbations',
                q=list(map(float,q)),retained_ten=retained.tolist(),retained_ambient_native_J=J.tolist(),
                four_constraints_model=(retained[[1,2,8,7]] * [1,1,-1,-1] +
                    [0,0,retained[1]*retained[3]/retained[0],retained[2]*retained[3]/retained[0]]).tolist(),
                rigorous_rounded_point=exact_balance(q,payload),
                historical_profile_continuum_diagnostic=historical_profile(q,retained),
                central_differences=checks,surface_J_max_absolute_error=err,
                scaling_null_max_absolute=float(np.max(abs(J@np.array(q,dtype=float)))),
                new_native_geometry_evaluations=0,strict_optimization_terminal=False,training_admitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('w4','design','full_ten'):
        parser.add_argument(key,type=Path)
    params=parser.parse_args()
    print(json.dumps(audit(params.w4,params.design,params.full_ten),indent=2))
