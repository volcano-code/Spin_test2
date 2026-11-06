"""Exact occupied-cell witness for the archived reconstructed 2014 Ellipsoid."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

from .real_h8e_archive_m8 import require


def frozen_mask(runs):
    flat=[b for b,n in runs for _ in range(n)]
    require(len(flat)==16**3 and set(flat)=={0,1}, 'invalid RLE')
    return {(i%16,i//16%16,i//256) for i,b in enumerate(flat) if b}


def voxel_boundary(cells):
    boundary=set()
    for cell in cells:
        for axis,step in product(range(3),(-1,1)):
            adjacent=list(cell)
            adjacent[axis]+=step
            if tuple(adjacent) not in cells:
                base=list(cell)
                base[axis]+=int(step==1)
                boundary.add((axis,tuple(base),step))
    return boundary


def mesh_mask(path, orientation):
    """Recover interiors by oriented x crossings; then check every face along x,y,z."""
    verts=[]
    halves=defaultdict(list)
    for line in path.read_text().splitlines():
        head,*args=line.split()
        if head=='v':
            grid=[8*(F(val)+offset) for val,offset in zip(args,(1,1,0))]
            require(len(grid)==3 and all(n.denominator==1 and 0<=n<=16 for n in grid), 'off-grid vertex')
            verts.append(tuple(map(int,grid)))
            continue
        require(head=='f' and len(args)==3 and all(0<int(i)<=len(verts) for i in args), 'invalid face')
        t=tuple(verts[int(i)-1] for i in args)
        low=tuple(map(min,zip(*t)))
        span=tuple(max(x)-min(x) for x in zip(*t))
        require(sorted(span)==[0,1,1] and len(set(t))==3,'not half a voxel face')
        axis=span.index(0); u=(axis+1)%3; v=(axis+2)%3
        a,b,c=t
        normal=((b[u]-a[u])*(c[v]-a[v])-(b[v]-a[v])*(c[u]-a[u]))*orientation
        require(abs(normal)==1,'degenerate voxel face')
        halves[(axis,low,normal)].append(set(t))
    for pair in halves.values():
        require(len(pair)==2 and len(pair[0]|pair[1])==4,'missing halves')
        shared=list(pair[0]&pair[1])
        require(sum(abs(a-b) for a,b in zip(*shared))==2,'face halves overlap')
    cells=set()
    for y,z in product(range(16),repeat=2):
        inside=False
        for x in range(17):
            for sign in (-1,1):
                if (0,(x,y,z),sign) in halves:
                    require(inside==(sign==1),'orientation mismatch')
                    inside=not inside
            if inside: cells.add((x,y,z))
        require(not inside,'unbounded mask')
    require(set(halves)==voxel_boundary(cells),'boundary differs from recovered cells')
    return cells


def moments(cells):
    out=[F(0)]*10
    for ix,iy,iz in cells:
        x,y,z=F(2*ix-15,16),F(2*iy-15,16),F(2*iz+1,16)
        correction=F(1,768)  # side squared / 12
        values=(1,x,y,z,x*x+correction,y*y+correction,z*z+correction,y*z,x*z,x*y)
        for i,val in enumerate(values): out[i]+=F(val,512)
    return out


def verify(root):
    registry=json.loads((root/'spin2014/paper_cases.json').read_bytes())
    case=next(c for c in registry['cases'] if c['id']=='ellipsoid-top')
    report=json.loads((root/'viewer/spin2014-ellipsoid.json').read_bytes())['cases'][0]
    repeat=report['repeats'][0]
    directory=root/'viewer/spin2014/ellipsoid-top/repeat-1'
    data=(root/'spin2014'/case['occupancy']['path']).read_bytes()
    require(sha256(data).hexdigest()==case['occupancy']['sha256']==report['occupancy_sha256']==
            '6036f03ff83777a636cbb8cf67bd57675caa44b90c40b3d2e442f5fb27d85983', 'different asset')
    require(set(repeat['viewer_bundle'])=={'diagnostics.json','outer.obj','design.obj','material.obj','cavity.obj'}
            and all(sha256((directory/k).read_bytes()).hexdigest()==v for k,v in repeat['viewer_bundle'].items()),
            'invalid viewer bundle')
    r=json.loads(data)
    require(r['dimensions']==[16]*3 and r['minimum']==[-1,-1,0] and r['maximum']==[1,1,2],'changed grid')
    diag=json.loads((directory/'diagnostics.json').read_bytes())
    require(report['id']=='ellipsoid-top' and case['author_exact'] is False and case['target_axis']==[0,0,1]
            and diag['target_origin_model']==[0,0,0] and diag['target_axes_model']==[[1,0,0],[0,1,0],[0,0,1]]
            and all(repeat[k]==diag[k] for k in ('geometry_hash','input_hash','trace_hash','output_hash')),
            'changed target or archive')
    outer=frozen_mask(r['material_rle']); design=frozen_mask(r['carveable_rle'])
    require(outer==mesh_mask(directory/'outer.obj',1)==mesh_mask(directory/'material.obj',1)
            and design==mesh_mask(directory/'design.obj',1), 'paper asset/mesh mismatch')
    cavity=mesh_mask(directory/'cavity.obj',-1)
    require(cavity <= design <= outer, 'illegal excavation')
    retained=outer-cavity
    m,x,y,z,xx,yy,zz,yz,xz,xy=raw=moments(retained)
    require(all(abs(float(a)-b)<1e-12 for a,b in zip(raw,diag['final_raw_moments'])),'2014 moment mismatch')
    constraints=[x,y,-xz+x*z/m,-yz+y*z/m]
    require(constraints==[0]*4,'original physical equalities failed')
    return dict(schema='w0-reconstructed-ellipsoid-mask-witness-v1',case='ellipsoid-top',method='exact voxel integrals',
        source_asset_sha256=report['occupancy_sha256'], viewer_bundle_sha256=repeat['viewer_bundle'],
        input_hash=diag['input_hash'], target_origin=diag['target_origin_model'], target_axes=diag['target_axes_model'],
        outer_cells=len(outer),design_cells=len(design),locked_cells=len(outer-design),cavity_cells=len(cavity),
        retained_variable_cells=len(design-cavity),total_material_cells=len(retained),
        total_material_raw_moments=list(map(str,raw)), exact_constraints=list(map(str,constraints)),
        exact_center_of_mass=list(map(str,[x/m,y/m,z/m])), archived_2014_status=diag['status'],
        reconstructed_asset=True, measured_physical_validation=False, faster2024_trajectory_or_terminal_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_root',type=Path)
    print(json.dumps(verify(parser.parse_args().experiment_root),indent=2))
