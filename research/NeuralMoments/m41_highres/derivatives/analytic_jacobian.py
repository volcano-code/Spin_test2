"""Differentiate the native-facet x-ray moment readout in a single local surface pass."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from native_jv import Raster, load, normals, parse_mesh, quadric, ray_hits, rel, stencil, vertices


def source_velocities(q,n):
    """Differentiate the fixed-normal map directly. No native Jacobian enters prediction."""
    a,c,level=quadric(q)
    inv=np.linalg.inv(a)
    m=n@inv
    norm_factor=(m*n).sum(1)
    scale=np.sqrt(-level/norm_factor)
    vel=np.empty((len(n),3,10))
    for i in range(10):
        e=np.eye(10)[i]
        da=np.array([[e[4],e[9],e[8]],[e[9],e[5],e[7]],[e[8],e[7],e[6]]])
        dc=-inv@(da@c+.5*e[1:4])
        dlevel=e[0]+c@e[1:4]+c@da@c
        dm=-m@da@inv
        dscale=.5*scale*(dlevel/level-(dm*n).sum(1)/norm_factor)
        vel[:,:,i]=dc+dm*scale[:,None]+m*dscale[:,None]
    return vel


def endpoint_jacobian(raster,v,face,vertex_vel):
    row,x,tri,u,w=ray_hits(v,face,raster.y,raster.z,details=True)
    # Look up the fixed design intervals containing each moving endpoint. Other
    # intervals have a static endpoint here and contribute no derivative.
    grid_x=np.floor((x+1)/raster.h).astype(np.int64)
    keys=row*raster.n+grid_x
    starts=raster.row.astype(np.int64)*raster.n+np.rint((raster.lo+1)/raster.h).astype(np.int64)
    ends=raster.row.astype(np.int64)*raster.n+np.rint((raster.hi+1)/raster.h).astype(np.int64)-1
    which=np.searchsorted(starts,keys,side='right')-1
    selected=(which>=0)&(grid_x>=0)&(grid_x<raster.n)&(keys<=ends[np.maximum(which,0)])
    s=np.where(np.flatnonzero(selected)%2==0,-1.,1.)
    row,x,tri,u,w=(a[selected] for a in (row,x,tri,u,w))
    y,z=raster.y[row%raster.n],raster.z[row//raster.n]
    facets=v[face]
    ns=np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0])
    # Each moving vertex contributes its barycentric weight times n.dot(v')/n.x.
    projected=np.einsum('fs,fjsp->fjp',ns,vertex_vel[face])
    projected=np.divide(projected,ns[:,None,0,None],where=(np.abs(ns[:,None,0,None])>=1e-15),out=np.zeros_like(projected))
    d2=raster.h*raster.h/12
    out=np.zeros((10,10))
    for i in range(0,len(x),32768):
        sl=slice(i,i+32768)
        dx=((1-u[sl]-w[sl])[:,None]*projected[tri[sl],0]+
            u[sl,None]*projected[tri[sl],1]+w[sl,None]*projected[tri[sl],2])
        a,b,c=x[sl],y[sl],z[sl]
        psi=np.column_stack((np.ones_like(a),a,b,c,a*a,b*b+d2,c*c+d2,b*c,a*c,a*b))
        out+=(psi*(s[sl]*raster.h*raster.h)[:,None]).T@dx
    return out,len(x)


def run(handoff,sizes,output,design_path=None):
    data=dict(schema='noodles-native-facet-single-axis-analytic-J-v1',
              frame='tip coordinates, raw ten (1,x,y,z,xx,yy,zz,yz,xz,xy)',
              readout='fixed cell-centered y,z rays, piecewise-constant voxel D x-runs, continuous moving x endpoints; barycentric facet normal velocity; prism width h along y,z',
              numeric_reference='M38 native cut-face J_tip (both accepted Noodles states); columns in raw q coordinates',
              timings='CPU single call, includes cutter ray intersections, endpoint Jacobian contraction, includes fixed-normal vertex and velocity evaluation; excludes design setup; no GPU, learned operator, Boolean, optimization or complete online timing',
              rows=[])
    objects=load(handoff)
    design,_=parse_mesh(design_path or handoff/'inputs/Dprime.txt')
    fixed_normals=normals(objects[0]['q'],objects[0]['x'])
    for n in sizes:
        start=perf_counter()
        rast=Raster(stencil(design,n),n)
        setup=perf_counter()-start
        for obj in objects:
            q=obj['q']
            start=perf_counter()
            moved_vertices=vertices(q,fixed_normals)
            velocity=source_velocities(q,fixed_normals)
            matrix,n_active=endpoint_jacobian(rast,moved_vertices,obj['faces'],velocity)
            elapsed=perf_counter()-start
            ref=obj['reference']
            # Exercise all ten coefficient columns using independent symmetric
            # differences of the implemented continuous readout; scaling q is immaterial.
            fd=[]
            h=1e-8
            for e in np.eye(10):
                plus=rast.integrate(vertices(q+h*e,fixed_normals),obj['faces'])['subvoxel_x']
                minus=rast.integrate(vertices(q-h*e,fixed_normals),obj['faces'])['subvoxel_x']
                fd.append((plus-minus)/(2*h))
            diffs=np.array(fd).T
            row=dict(n=n,state=obj['index'],elapsed_seconds=elapsed,shared_design_seconds=setup,
                     active_endpoints=n_active,source_velocity_error=rel(velocity,obj['vel']),J=matrix.tolist(),native_J=ref.tolist(),
                     rel_Frobenius=rel(matrix,ref),rel_izz_grad=rel(matrix[4]+matrix[5],ref[4]+ref[5]),
                     radial_defect=float(np.linalg.norm(matrix@q)/np.linalg.norm(matrix)),
                     fd_step=h,fd_relative_to_analytic=rel(diffs,matrix))
            data['rows'].append(row)
            output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
            print(n,obj['index'],row['rel_Frobenius'],row['rel_izz_grad'],row['fd_relative_to_analytic'],elapsed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--handoff',type=Path,required=True)
    p.add_argument('--design',type=Path,help='optional path to the previous 5090 kit inputs/Dprime.txt')
    p.add_argument('--sizes',type=int,nargs='+',default=[128,256,512,1024])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    run(args.handoff,args.sizes,args.output,args.design)
