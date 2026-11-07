"""Noodles local finite-cutter parameter perturbations against the archived native ten-moment derivative."""
import argparse
from array import array
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from vtkmodules.vtkCommonCore import reference, vtkVersion
from noodles_resolution import DESIGN_SHA, parse_mesh, stencil
from paired_cuts import retained_stencil

REPORT_SHA = '0872197d53d0744fee9968dbf0d80c8bdffab65b942620313d2c70aaf23061aa'
PACKETS = {1: ('anchor', '4d4bd291fbc7dee16415aca9251db2b19cc8b2768cca0067a14c0c8b4108f7dd'),
           6: ('exact/call-006', 'df456882833b0a6dd314becd450c80d984cc10fb33610e1439eecbe4c85ba7dc')}


def checked(path, expected):
    b = path.read_bytes()
    if sha256(b).hexdigest() != expected:
        raise ValueError(f'Unexpected input digest: {path}')
    return json.loads(b)


def quadric(q):
    a = np.array([[q[4],q[9],q[8]], [q[9],q[5],q[7]], [q[8],q[7],q[6]]])
    if np.max(np.linalg.eigvalsh(a)) >= 0:
        raise ValueError('fixed-normal reconstruction requires this negative-definite ellipsoid stratum')
    c = np.linalg.solve(a, -.5*q[1:4])
    return a, c, q[0]+.5*q[1:4]@c


def normals(q, x_tip):
    a, c, _ = quadric(q)
    n = (x_tip-c)@a
    return n/np.linalg.norm(n, axis=1, keepdims=True)


def vertices(q, n):
    a, c, level = quadric(q)
    m = np.linalg.solve(a, n.T).T
    return c + m*np.sqrt(-level/np.sum(m*n, axis=1))[:,None]


def vtk_runs(mask, n):
    row, lower, upper = array('I'), array('I'), array('I')
    lo, hi, it = reference(0), reference(0), reference(0)
    for z in range(n):
        for y in range(n):
            it.set(0)
            while mask.GetNextExtent(lo, hi, 0, n-1, y, z, it):
                row.append(n*z+y)
                lower.append(int(lo))
                upper.append(int(hi))
    return np.array(row), np.array(lower), np.array(upper)


def ray_hits(v, faces, y, z, *, details=False):
    """All intersections, including multiple disjoint cutter intervals on any ray."""
    row, hits = [], []
    triangles, bary_u, bary_w = [], [], []
    for index,(a,b,c) in enumerate(v[faces]):
        by,bz = b[1:]-a[1:]
        cy,cz = c[1:]-a[1:]
        det = by*cz-bz*cy
        if abs(det)<1e-15:
            continue
        iy = np.arange(np.searchsorted(y,min(a[1],b[1],c[1])),
                       np.searchsorted(y,max(a[1],b[1],c[1]),'right'))
        iz = np.arange(np.searchsorted(z,min(a[2],b[2],c[2])),
                       np.searchsorted(z,max(a[2],b[2],c[2]),'right'))
        dy,dz = y[iy][None,:]-a[1], z[iz][:,None]-a[2]
        u = (dy*cz-dz*cy)/det
        w = (by*dz-bz*dy)/det
        k,j = np.where((u>=0)&(w>=0)&(u+w<=1))
        row.append(iz[k]*len(y)+iy[j])
        hits.append(a[0]+u[k,j]*(b[0]-a[0])+w[k,j]*(c[0]-a[0]))
        if details:
            triangles.append(np.full(len(k),index,dtype=np.int32))
            bary_u.append(u[k,j])
            bary_w.append(w[k,j])
    row, hits = np.concatenate(row), np.concatenate(hits)
    order = np.lexsort((hits,row))
    row,hits = row[order],hits[order]
    # Shared triangle edges should be counted once; odd rows fail closed.
    keep = np.r_[True,(row[1:]!=row[:-1])|(np.abs(np.diff(hits))>1e-12)]
    row,hits = row[keep],hits[keep]
    if len(row)%2 or not np.array_equal(row[::2],row[1::2]):
        raise ValueError('unpaired cutter intersections on at least one ray')
    if details:
        return row,hits,*(np.concatenate(a)[order][keep] for a in (triangles,bary_u,bary_w))
    return row[::2],hits[::2],hits[1::2]


def ten(lo,hi,y,z,h):
    # Analytic prism integrals with h-wide transverse footprints centered on sample rays.
    dx = np.maximum(0., hi-lo)
    x = (lo+hi)/2
    w = dx*h*h
    d2 = h*h/12
    return np.array([w.sum(),w@x,w@y,w@z,w@(x*x+dx*dx/12),w@(y*y+d2),
                     w@(z*z+d2),w@(y*z),w@(x*z),w@(x*y)])


class Raster:
    def __init__(self, mask, n):
        self.n=n
        self.h=2/n
        self.y=-1+(np.arange(n)+.5)*self.h
        self.z=self.y+1   # tip coordinates (model z plus 0.16).
        self.row,a,b=vtk_runs(mask,n)
        self.lo=-1+self.h*a
        self.hi=-1+self.h*(b+1)
        self.dy=self.y[self.row%n]
        self.dz=self.z[self.row//n]

    def integrate(self, v, faces):
        rr,aa,bb=ray_hits(v,faces,self.y,self.z)
        if not len(rr):
            return {'hard':np.zeros(10),'subvoxel_x':np.zeros(10)}
        # There may be more than one interval on a ray. Process intervals by ordinal.
        ordinal=np.arange(len(rr))-np.maximum.accumulate(np.where(np.r_[True,rr[1:]!=rr[:-1]],np.arange(len(rr)),0))
        hard,sub=np.zeros(10),np.zeros(10)
        for i in range(int(ordinal.max())+1):
            which=ordinal==i
            left=np.full(self.n**2,1.,float)
            right=np.full(self.n**2,-1.,float)
            left[rr[which]]=aa[which]
            right[rr[which]]=bb[which]
            lo=np.maximum(self.lo,left[self.row])
            hi=np.minimum(self.hi,right[self.row])
            valid=hi>lo
            lo,hi=lo[valid],hi[valid]
            y,z=self.dy[valid],self.dz[valid]
            sub+=ten(lo,hi,y,z,self.h)
            # Center inclusion followed by integration of the complete selected cubes.
            a=-1 + self.h*np.ceil((lo+1)/self.h-.5)
            b=-1 + self.h*(np.floor((hi+1)/self.h-.5)+1)
            hard+=ten(a,b,y,z,self.h)
        return {'hard':hard,'subvoxel_x':sub}


def project(q, v):
    u=v-q*(q@v)
    return u/np.linalg.norm(u)


def load(handoff):
    report=checked(handoff/'online/exact/report.json',REPORT_SHA)
    states=[]
    for index,(where,receipt_hash) in PACKETS.items():
        base=handoff/'online'/where
        r=checked(base/'receipt.json',receipt_hash)
        s=checked(base/'source.json',r['source_sha256'])
        row=report['evaluations'][index-1]
        if (r['operation']!='intersection' or r['design_sha256']!=DESIGN_SHA or r['result_sha256']!=row['result_sha256'] or s['surface_type']!='ELLIPSOID' or s['raw_q']!=row['q'] or any(s['face_is_cap'])):
            raise ValueError('native binding mismatch')
        q=np.array(row['q'])
        x=np.array(s['vertices_tip'])
        f=np.array(s['faces'])
        if not np.allclose(x-[0,0,.16],s['vertices'],atol=1e-14,rtol=0):
            raise ValueError('wrong tip-to-model frame')
        cutter,_=parse_mesh(base/'cutter.txt',r['cutter_sha256'])
        states.append(dict(index=index,q=q,faces=f,reference=np.array(row['J_tip']),
                           native_ten=np.array(row['variable_ten_tip']),vel=np.array(s['native_jacobian']),
                           x=x,cutter=cutter,receipt=receipt_hash))
    return states


def rel(a,b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def run(handoff,sizes,steps,out,design_path=None):
    start=time.perf_counter()
    states=load(handoff)
    design,_=parse_mesh(design_path or handoff/'inputs/Dprime.txt')
    common=normals(states[0]['q'],states[0]['x'])
    result=dict(schema='noodles-native-ellipsoid-rle-jv-v1',vtk=vtkVersion.GetVTKVersion(),
                design_sha256=DESIGN_SHA,report_sha256=REPORT_SHA,
                moment_order='1 x y z xx yy zz yz xz xy (tip coordinates)',
                scope='centered geodesic finite differences vs pointwise cached M38 J_tip; fixed-normal finite ellipsoids only',
                tangent_directions='trajectory from q1 to q6 (forward orientation at both states), projected e0, projected e9 (cross xy)',
                hard='VTK center inclusion for D and native finite cutter; full voxel cube moments',
                subvoxel_x='same D voxel x-runs and y,z rays; fractional x endpoints of native triangle intersections',
                states=[],baseline=[],rows=[])
    for obj in states:
        q=obj['q']
        error=float(np.max(np.abs(vertices(q,common)-obj['x'])))
        if error>1e-11 or not np.array_equal(obj['faces'],states[0]['faces']):
            raise ValueError('native baseline not reproduced by fixed normals')
        vectors=[project(q,v) for v in (states[1]['q']-states[0]['q'],np.eye(10)[0],np.eye(10)[9])]
        obj['directions']=dict(zip(('trajectory','offset','xy'),vectors))
        h=1e-7
        v_err=[]
        for u in vectors:
            fx=lambda step:vertices(np.cos(step)*q+np.sin(step)*u,common)
            v_err.append(rel((fx(h)-fx(-h))/(2*h),obj['vel']@u))
        if max(v_err)>1e-6:
            raise ValueError('reconstruction derivative differs from native vertex velocities')
        result['states'].append(dict(index=obj['index'],q=q.tolist(),receipt_sha256=obj['receipt'],directions={k:v.tolist() for k,v in obj['directions'].items()},base_vertex_error=error,velocity_relative_errors=v_err))
    for n in sizes:
        started=time.perf_counter()
        design_mask=stencil(design,n)
        raster=Raster(design_mask,n)
        for obj in states:
            q=obj['q']
            computed=raster.integrate(obj['x'],obj['faces'])
            mask=retained_stencil(design_mask,stencil(obj['cutter'],n))
            rr,lo,hi=vtk_runs(mask,n)
            truth=ten(-1+raster.h*lo,-1+raster.h*(hi+1),raster.y[rr%n],raster.z[rr//n],raster.h)
            difference=float(np.max(abs(truth-computed['hard'])))
            if difference>1e-10:
                raise ValueError(f'center-sampled independent VTK mismatch: {difference}')
            result['baseline'].append(dict(n=n,state=obj['index'],vtk_max_abs_ten_error=difference,
                                             native_moment_errors={k:rel(v,obj['native_ten']) for k,v in computed.items()}))
            for name,u in obj['directions'].items():
                jv=obj['reference']@u
                izz=jv[4]+jv[5]
                for h in steps:
                    endpoints=[np.cos(h)*q+sign*np.sin(h)*u for sign in (-1,1)]
                    a,b=[raster.integrate(vertices(t,common),obj['faces']) for t in endpoints]
                    for method in a:
                        fd=(b[method]-a[method])/(2*h)
                        estimate=fd[4]+fd[5]
                        result['rows'].append(dict(n=n,state=obj['index'],direction=name,h=h,method=method,
                                                   jv_native=jv.tolist(),jv_estimate=fd.tolist(),
                                                   rel_ten=rel(fd,jv),rel_izz=abs(float(estimate/izz)-1),
                                                   izz_native=float(izz),izz_estimate=float(estimate)))
        result['elapsed_seconds']=time.perf_counter()-start
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(n,len(result['rows']),round(time.perf_counter()-started,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--handoff',type=Path,required=True)
    p.add_argument('--design',type=Path,help='optional path to the previous 5090 kit inputs/Dprime.txt')
    p.add_argument('--sizes',type=int,nargs='+',default=[128,256,512])
    p.add_argument('--steps',type=float,nargs='+',default=[1e-3,1e-4,1e-5,1e-6])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    run(args.handoff,args.sizes,args.steps,args.output,args.design)
