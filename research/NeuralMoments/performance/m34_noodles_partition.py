"""Independent exact moment and surface audits for the M32 Noodles material partition."""
import argparse
from collections import Counter
try:
    from gmpy2 import mpq as Q
except ImportError:
    from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/faster_core_repair_20261012'))
from exact_planar_patches import cross, dot, oriented_planar_chain_signature, subtract

ORDER = ('mass','x','y','z','xx','yy','zz','yz','xz','xy')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def surface_data(path):
    raw = path.read_bytes()
    obj = json.loads(raw)
    vertices, faces = obj['vertices'], obj['faces']
    if not vertices or not faces or not all(isinstance(s,str) for p in vertices for s in p):
        raise ValueError('missing rational material')
    return sha(raw), vertices, faces


def mesh_stream(vertices, faces):
    return ('\n'.join([f'{len(vertices)} {len(faces)}', *(' '.join(p) for p in vertices),
                      *(' '.join(map(str,face)) for face in faces)])+'\n').encode('ascii')


def balanced_sum(values):
    """Keep large common rational denominators local until the last reduction."""
    values = list(values)
    while len(values)>1:
        values = [sum(values[i:i+2]) for i in range(0,len(values),2)]
    return values[0] if values else Q(0)


def token(v):
    a,b = v.split('/')
    return Q(int(a,0),int(b,0))


def exact_ten(binary, vertices, faces):
    blocks = []
    # M25 integrates oriented signed tetrahedra, also valid for face subsets by additivity.
    for start in range(0, len(faces), 256):
        group = faces[start:start+256]
        vi = sorted({v for f in group for v in f})
        indices = {v:i for i,v in enumerate(vi)}
        fs = [[indices[j] for j in f] for f in group]
        run = subprocess.run([str(binary)],input=mesh_stream([vertices[i] for i in vi],fs),
                             check=True,capture_output=True)
        vals = [token(s) for s in run.stdout.decode().splitlines()]
        if len(vals)!=10: raise ValueError('incomplete M25 result')
        blocks.append(vals)
    return tuple(map(balanced_sum, zip(*blocks)))


def point_tip(vertices):
    """Exactly compare minimum z contact to the binary64 registration in the V7 specification."""
    tip = (Q(0),Q(0),Q(-.16))
    low = min(p[2] for p in vertices)
    tips = {p for p in vertices if p[2] == low}
    return dict(minimum_z=float(low), lowest_vertex_positions=len(tips), exact_fixed_tip=tips=={tip})


def point_winding(vertices, faces, point, lower, upper, skip_face=None, outward_normal=None):
    """Exact signed winding, or its outward limit when starting on one embedded face."""
    p=np.array(point,dtype=float)
    for k in range(3):
        if outward_normal is not None and not outward_normal[k]:continue
        sign=1 if outward_normal is None or outward_normal[k]>0 else -1
        u,v=(k+1)%3,(k+2)%3
        ray_bounds=(upper[:,k]>=p[k]) if sign==1 else (lower[:,k]<=p[k])
        bounds=ray_bounds&(lower[:,u]<=p[u])&(upper[:,u]>=p[u])&(lower[:,v]<=p[v])&(upper[:,v]>=p[v])
        winding=0
        for index in np.flatnonzero(bounds):
            if index==skip_face:continue
            a,b,c=(vertices[i] for i in faces[index]);tri=(a,b,c)
            n=cross(subtract(b,a),subtract(c,a));height=dot(n,subtract(a,point))
            if not n[k]:
                if not height:break
                continue
            if height*n[k]*sign<0:continue
            turns=[((b[u]-a[u])*(point[v]-a[v])-(b[v]-a[v])*(point[u]-a[u]))*n[k]
                   for a,b in zip(tri,tri[1:]+tri[:1])]
            if min(turns)<0:continue
            if not min(turns) or not height:break
            winding+=sign if n[k]>0 else -sign
        else:return winding
    raise ValueError('NO_GENERIC_AXIAL_RAY')


def regular_orientation(vertices, faces):
    """Each embedded connected boundary has winding zero just outside its oriented normal."""
    vv=[tuple(map(Q,p)) for p in vertices]
    tri=np.array(vv,dtype=float)[np.array(faces)]
    lower=np.nextafter(tri.min(axis=1),-np.inf);upper=np.nextafter(tri.max(axis=1),np.inf)
    parent=list(range(len(vv)))
    def find(i):
        while i!=parent[i]:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for a,b,c in faces:
        i,j,k=find(a),find(b),find(c)
        parent[j]=parent[k]=i
    groups={}
    for i,(a,b,c) in enumerate(faces):
        groups.setdefault(find(a),[]).append(i)
    counts=[]
    for ids in groups.values():
        for i in ids:
            a,b,c=map(vv.__getitem__,faces[i])
            n=cross(subtract(b,a),subtract(c,a));p=tuple(sum(t)/3 for t in zip(a,b,c))
            try: w=point_winding(vv,faces,p,lower,upper,skip_face=i,outward_normal=n)
            except ValueError:continue
            if w:raise ValueError('BOUNDARY_WINDING_NOT_ZERO_ON_OUTWARD_SIDE')
            counts.append(len(ids));break
        else:raise ValueError('NO_EXACT_COMPONENT_WINDING_TEST')
    return counts


def run_surface(packet, ten, embedding, max_seconds):
    digest,v,f = surface_data(packet)
    nums = exact_ten(ten,v,f)
    proc = subprocess.run([str(embedding),'--max-seconds',str(max_seconds)],
                          input=mesh_stream(v,f),capture_output=True,timeout=max_seconds+60)
    fields = dict(line.split('=',1) for line in proc.stdout.decode().splitlines() if '=' in line)
    if proc.returncode:fields['exit_code']=str(proc.returncode)
    return surface_record(digest,v,f,nums,fields),nums


def surface_record(digest, v, f, nums, fields):
    geometry_ok = (fields.get('status')=='EXACT_POST_BOOLEAN_SELF_INTERSECTION_FREE'
                   and int(fields['vertices'])==len(v) and int(fields['faces'])==len(f)
                   and fields['candidate_pairs']==fields['exact_pairs_checked'])
    surface=dict(source_sha256=digest,vertices=len(v),faces=len(f),signed_ten=list(map(float,nums)),
                 axial_inertia_model=float(nums[4]+nums[5]), embedding_passed=geometry_ok,
                 embedding={k:v for k,v in fields.items() if k!='elapsed_seconds'},
                 **point_tip([tuple(map(Q,x)) for x in v]))
    return surface


def half_flux(polygon, axis):
    """Integrate n_i*x_i²/2 on clipped faces. The closing axis-plane contributes zero."""
    value = Q(0)
    for i in range(1,len(polygon)-1):
        a,b,c=polygon[0],polygon[i],polygon[i+1]
        j,k=(axis+1)%3,(axis+2)%3
        twice_area=(b[j]-a[j])*(c[k]-a[k])-(b[k]-a[k])*(c[j]-a[j])
        x,y,z=a[axis],b[axis],c[axis]
        value+=twice_area*(x*x+y*y+z*z+x*y+y*z+z*x)/24
    return value


def clip(triangle, axis, sign):
    out=[]
    for a,b in zip(triangle, triangle[1:]+triangle[:1]):
        u,v=a[axis],b[axis]
        if sign*u>=0: out.append(a)
        if u*v<0:
            factor=u/(u-v)
            out.append(tuple(a[j]+factor*(b[j]-a[j]) for j in range(3)))
    return out


def interval(raw,faces,axis):
    buckets=[[],[]]
    for face in faces:
        tri=tuple(raw[i] for i in face)
        values=[v[axis] for v in tri]
        if max(values)<=0:
            buckets[0].append(half_flux(tri,axis))
        elif min(values)>=0:
            buckets[1].append(half_flux(tri,axis))
        else:
            for out,sgn in zip(buckets,(-1,1)):
                out.append(half_flux(clip(tri,axis,sgn),axis))
    lo,hi=map(balanced_sum,buckets)
    if not lo<=0<=hi: raise ValueError('incorrect oriented material integral')
    return lo,hi


def partition_chain(meshes):
    """Cancel identical oriented triangles before the exact coplanar interval check."""
    coords={}
    chain=Counter()
    for vs,fs,sign in meshes:
        vi=[]
        for p in vs:
            point=tuple(map(Q,p))
            vi.append(coords.setdefault(point,len(coords)))
        for f in fs:
            a,b,c=(vi[i] for i in f)
            if len({a,b,c})<3:raise ValueError('collapsed face')
            orientation=1 if ((a<b)+(b<c)+(c<a))==2 else -1
            chain[tuple(sorted((a,b,c)))]+=sign*orientation
    points=list(coords); residual=[]
    for tri,n in chain.items():
        if n: residual.extend([tri if n>0 else tri[::-1]]*abs(n))
    cert=oriented_planar_chain_signature(points,residual)
    return dict(triangles_left_after_literal_cancellation=len(residual), residual_support_planes=len(cert['signature']),
                oriented_boundary_chain_additive=not cert['signature'])


def exact_repr(vals):
    return [f'{v.numerator:#x}/{v.denominator:#x}' for v in vals]


def collect(folder, auditdir):
    roles=('A','F','Dprime','Fprime')
    nums={}; audit={}; meshes={}
    for k in roles:
        h,v,f=surface_data(folder/(k+'.json'))
        meshes[k]=(v,f)
        saved=json.loads((auditdir/(k+'.w1.json')).read_text())
        if h!=saved['source_sha256']:raise ValueError('MESH_SOURCE_DRIFT')
        nums[k]=list(map(token,(auditdir/(k+'.ten')).read_text().splitlines()))
        if len(nums[k])!=10 or sha(('\n'.join(exact_repr(nums[k]))+'\n').encode())!=saved['exact_ten_payload_sha256']:
            raise ValueError('EXACT_TEN_DRIFT')
        audit[k]=saved
    additive=all(a==b+c for a,b,c in zip(nums['A'],nums['Fprime'],nums['Dprime']))
    if not additive:raise ValueError('NONADDITIVE_EXACT_RAW_MOMENTS')
    check=partition_chain([(*meshes[k],s) for k,s in [('A',-1),('Fprime',1),('Dprime',1)]])
    if not check['oriented_boundary_chain_additive']:raise ValueError('NONADDITIVE_ORIENTED_BOUNDARY')
    vv,ff=meshes['Dprime'];raw=[tuple(map(Q,x)) for x in vv]
    component_faces={}
    for k in roles:
        if not audit[k]['embedding_passed']: raise ValueError('NON_EMBEDDED_SOLID:'+k)
        component_faces[k]=regular_orientation(*meshes[k])
    capacity={}
    factors=[]
    for axis, label in enumerate(('x','y')):
        lo,hi=interval(raw,ff,axis)
        if lo+hi!=nums['Dprime'][axis+1]:raise ValueError('HALF_DIVERGENCE_DISAGREES_WITH_TETRA')
        factors.extend((lo,hi))
        fixed=nums['Fprime'][axis+1]
        capacity[label]=dict(minimum_variable=float(lo),maximum_variable=float(hi), fixed=float(fixed),
                             total_range=[float(fixed+lo),float(fixed+hi)],zero_in_range=fixed+lo<=0<=fixed+hi)
    max_r2=max(x*x+y*y for x,y,z in raw)  # convex squared radius attains a max at a boundary vertex
    exact=(auditdir/'capacity.exact')
    exact.write_bytes(('\n'.join(exact_repr(factors+[max_r2]))+'\n').encode())
    full_inertia=sum(nums['A'][4:6]); fixed_inertia=sum(nums['Fprime'][4:6])
    floor=nums['Fprime'][0]+max(Q(0),(Q(9,10)*full_inertia-fixed_inertia)/max_r2)
    bound=dict(design_max_model_radius_squared=float(max_r2), exact_radius_squared_token=exact_repr([max_r2])[0],
               mass_lower_bound_fraction=float(floor/nums['A'][0]),fixed_mass_fraction=float(nums['Fprime'][0]/nums['A'][0]),
               fixed_inertia_fraction=float(fixed_inertia/full_inertia))
    return dict(schema='w1-m34-rational-noodles-partition',raw_order=ORDER,surfaces=audit,ten_exactly_additive=additive,
                boundary_chain=check, component_faces=component_faces, regular_solid_winding_verified=True,
                first_moment_capacity=capacity, minimum_mass_at_90pct_inertia=bound,
                capacity_exact_file_sha256=sha(exact.read_bytes()),
                interpretation='Oriented rational solid partition in the source frame; per-axis necessary ranges. Common witness, binary64 transport and Newton are assessed separately.')


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True)
    one=s.add_parser('mesh');one.add_argument('folder',type=Path);one.add_argument('auditdir',type=Path)
    one.add_argument('role',choices=('A','F','Dprime','Fprime'));one.add_argument('--ten',required=True,type=Path)
    one.add_argument('--embedding',required=True,type=Path);one.add_argument('--seconds',type=int,default=1500)
    all_=s.add_parser('collect');all_.add_argument('folder',type=Path);all_.add_argument('auditdir',type=Path)
    args=p.parse_args()
    if args.mode=='collect':print(json.dumps(collect(args.folder,args.auditdir),indent=2));return
    data,vals=run_surface(args.folder/(args.role+'.json'),args.ten.resolve(),args.embedding.resolve(),args.seconds)
    args.auditdir.mkdir(exist_ok=True,parents=True)
    payload=('\n'.join(exact_repr(vals))+'\n').encode()
    data['exact_ten_payload_sha256']=sha(payload)
    (args.auditdir/(args.role+'.ten')).write_bytes(payload)
    (args.auditdir/(args.role+'.w1.json')).write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__': main()
