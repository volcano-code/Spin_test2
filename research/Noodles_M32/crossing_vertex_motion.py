"""Independent pointwise edge/face velocity reconstruction and signed-tetrahedron Jacobian."""
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path
import argparse
import hashlib
import json

from crossing_flux import SCALE, PAIRS, cross, dot, sub
from crossing_report import DESIGN_SHA, packet


def det2(a,b,k):
    i,j=(k+1)%3,(k+2)%3
    return a[i]*b[j]-a[j]*b[i]


def linear_shape(p, tri):
    # Project the triangle and compute edge areas explicitly. This solves for
    # point coordinates, independently of Cutter's computed Boolean vertices.
    a,b,c=tri
    d,e=sub(b,a),sub(c,a); n=cross(d,e)
    k=max(range(3),key=lambda t:abs(n[t]))
    if not n[k] or dot(n,sub(p,a)):
        raise ValueError('singular or noncoplanar incident parent')
    w=(det2(sub(b,p),sub(c,p),k)/n[k],det2(sub(c,p),sub(a,p),k)/n[k],
       det2(sub(a,p),sub(b,p),k)/n[k])
    if min(w)<0 or sum(w)!=1:
        raise ValueError('bad projected barycentric support')
    return w


def solve_velocity(p, parents, sources, facets, src_vel, fixed_count):
    feature = [None,None]
    for birth in parents:
        k=int(birth>=fixed_count); f=facets[k][birth-k*fixed_count]
        pts=[sources[k][i] for i in f]
        shape=linear_shape(p,pts)
        positive=tuple(sorted((i,x) for i,x in zip(f,shape) if x))
        if feature[k] is None or len(positive)<len(feature[k]):
            feature[k]=positive
        elif len(positive)==len(feature[k]) and positive!=feature[k]:
            raise ValueError('ambiguous parent feature')
    ncol=len(src_vel[0][0])
    zero=(Q(0),)*3
    def vel(s,j):
        if s==0:return [zero]*ncol
        return list(zip(*src_vel[j]))
    sizes=tuple(len(f) if f else 0 for f in feature)
    if sizes in ((1,0),(0,1)):
        side=sizes.index(1)
        return vel(side,feature[side][0][0]),('fixed_vertex' if not side else 'cutter_vertex')
    if sorted(sizes)!=[2,3]:
        raise ValueError(f'event or unsupported feature intersection: {sizes}')
    s=sizes.index(2);k=1-s
    edge=feature[s];face=feature[k]
    a,b=[sources[s][i] for i,_ in edge]
    x,y,z=[sources[k][i] for i,_ in face]
    n=cross(sub(y,x),sub(z,x)); d=sub(b,a); nd=dot(n,d)
    if not nd:raise ValueError('nontransverse edge-face intersection')
    motions=[]
    def interpolated(side, entity):
        return [(w,vel(side,i)) for i,w in entity]
    edge_v=interpolated(s,edge);face_v=interpolated(k,face)
    for col in range(ncol):
        w=tuple(sum(l*v[col][j] for l,v in edge_v) for j in range(3))
        u=tuple(sum(l*v[col][j] for l,v in face_v) for j in range(3))
        dt=dot(n,sub(u,w))/nd
        motions.append(tuple(w[j]+d[j]*dt for j in range(3)))
    return motions,('fixed_edge_moving_face' if s==0 else 'moving_edge_fixed_face')


def tetra_jacobian(points,faces,motion,ncol):
    lo=[[0]*ncol for _ in range(10)]; hi=[[0]*ncol for _ in range(10)]
    for face in faces:
        verts=[points[i] for i in face]; a,b,c=verts
        bxc,cxa,axb=cross(b,c),cross(c,a),cross(a,b)
        vol=dot(a,bxc)/6
        S=tuple(map(sum,zip(a,b,c)))
        base=[1,*(x/4 for x in S),*((S[k]*S[l]+sum(p[k]*p[l] for p in verts))/20 for k,l in PAIRS)]
        for j in range(ncol):
            w=[motion[i][j] for i in face]
            if not any(any(p) for p in w):continue
            dv=(dot(w[0],bxc)+dot(w[1],cxa)+dot(w[2],axb))/6
            ds=tuple(map(sum,zip(*w)))
            dbase=[0,*(v/4 for v in ds),*((ds[k]*S[l]+S[k]*ds[l]+sum(
                wp[k]*p[l]+p[k]*wp[l] for p,wp in zip(verts,w)))/20 for k,l in PAIRS)]
            for i,(r,dr) in enumerate(zip(base,dbase)):
                v=(dv*r+vol*dr)*SCALE
                f,rem=divmod(v.numerator,v.denominator)
                lo[i][j]+=f;hi[i][j]+=f+bool(rem)
    return lo,hi


def audit_vertices(folder,design,flux):
    raw=Path(design).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=DESIGN_SHA:
        raise ValueError('changed fixed operand')
    base=Path(folder)
    out_raw=(base/'result.json').read_bytes()
    if hashlib.sha256(out_raw).hexdigest()!=flux['result_sha256']:
        raise ValueError('changed output mesh')
    src_raw=(base/'source.json').read_bytes();cutter_raw=(base/'cutter.txt').read_bytes()
    if any(hashlib.sha256(raw).hexdigest()!=flux[k] for k,raw in
           [('source_sha256',src_raw),('cutter_sha256',cutter_raw)]):
        raise ValueError('changed moving operand or motion')
    out=json.loads(out_raw);src=json.loads(src_raw)
    vs=[tuple(map(Q,p)) for p in out['vertices']]
    dv,df=packet(raw);cv,cf=packet(cutter_raw)
    motion=[tuple(tuple(map(Q,col)) for col in v) for v in src['native_jacobian']]
    inc=[set() for _ in vs]
    for owner,face in zip(out['birth_faces'],out['faces']):
        for i in face:inc[i].add(owner)
    deriv=[];kinds=Counter()
    for p,parents in zip(vs,inc):
        v,k=solve_velocity(p,parents,(dv,cv),(df,cf),motion,len(df))
        deriv.append(v);kinds[k]+=1
    lo,hi=tetra_jacobian(vs,out['faces'],deriv,10)
    dmax=0
    for i in range(10):
        for j in range(10):
            a,b=flux['lower_model'][i][j],flux['upper_model'][i][j]
            if lo[i][j]>b or hi[i][j]<a:
                raise ValueError(f'independent tetra derivative misses oriented-face flux at {i},{j}')
            dmax=max(dmax,b-lo[i][j],hi[i][j]-a)
    return dict(point_types=dict(kinds),vertices=len(vs),model_components_with_overlapping_intervals=100,
                maximum_certified_absolute_difference_upper=dmax/SCALE)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('sample','design','flux'):parser.add_argument(key,type=Path)
    args=parser.parse_args()
    print(json.dumps(audit_vertices(args.sample,args.design,json.loads(args.flux.read_bytes())),indent=2))
