"""A balanced-cuboid packing heuristic; returns feasible retained mass, not a quadric or an optimum."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from m34_noodles_cuboid_witness import (CENTERS, HALF, box_ten, point_winding, volume_solution)
from m34_noodles_partition import Q,sha,surface_data,token

# Inspect a finite box of dyadic lattice locations, not the entire design domain.
DENOM=32
RANGE=((-32,32),(-26,26),(14,45))
# The proportion of cell width used along each axis. The remaining gap is positive.
WIDTH=Q(7,8*DENOM)


def inside_cells(v,f):
    t=np.array(v,dtype=float)[np.array(f)]
    lower=np.nextafter(t.min(axis=1),-np.inf)
    upper=np.nextafter(t.max(axis=1),np.inf)
    lb=np.array([a for a,b in RANGE]);ub=np.array([b for a,b in RANGE])
    free=np.ones(ub-lb+1,dtype=bool)
    # Cell indices touching outward-rounded triangle AABBs. Factor is a power of two.
    # Round away from the triangle for the two half-cell arithmetic operations too.
    r0=np.maximum(np.ceil(np.nextafter(lower*DENOM-.5,-np.inf)).astype(int),lb)-lb
    r1=np.minimum(np.floor(np.nextafter(upper*DENOM+.5,np.inf)).astype(int),ub)-lb
    for a,b in zip(r0,r1):
        if np.any(a>b):continue
        free[a[0]:b[0]+1,a[1]:b[1]+1,a[2]:b[2]+1]=False
    cells,n=label(free)  # 6-neighbor full closed cells are connected across their common face.
    good=np.zeros(n+1,dtype=bool)
    for i in range(1,n+1):
        point=tuple(Q(int(a),DENOM) for a in np.argwhere(cells==i)[0]+lb)
        w=point_winding(v,f,point,lower,upper)
        if w not in (0,1):raise ValueError('INVALID_MATERIAL_WINDING')
        good[i]=(w==1)
    return {tuple(map(int,row)) for row in (np.argwhere(good[cells])+lb)},n


def extend(folder, auditdir, box_witness, pair_centers=None, eligible=None):
    witness=json.loads(box_witness.read_text())
    h,vs,fs=surface_data(folder/'Dprime.json')
    if h!=witness['Dprime_sha256']:raise ValueError('WITNESS_SOURCE_CHANGED')
    fixed=tuple(tuple(Q(s,16) for s in row) for row in CENTERS)
    raw=(auditdir/'A.ten').read_bytes()
    if sha(raw)!=witness['source_ten_sha256']:raise ValueError('A_MOMENT_DRIFT')
    full=list(map(token,raw.decode().splitlines()))
    exact_volumes=volume_solution(fixed,full)
    holes=[box_ten(c,vol) for c,vol in zip(fixed,exact_volumes)]
    Iref=full[4]+full[5]
    Iz_left=(Iref-sum(r[4]+r[5] for r in holes))-Q(9,10)*Iref
    candidates,num=inside_cells([tuple(map(Q,p)) for p in vs],fs)
    all_cells=''.join(f'{x} {y} {z}\n' for x,y,z in sorted(candidates)).encode('ascii')
    if eligible is not None:eligible.write_bytes(all_cells)
    candidates={t for t in candidates if all(max(abs(Q(s,DENOM)-coord) for s,coord in zip(t,b))>HALF+WIDTH/2 for b in fixed)}
    # (x,y)'s sign-reversed pair at the same z cancels x,y,xz,yz without consuming balance margin.
    pairs=[q for q in candidates if (q[0]>0 or q[0]==0 and q[1]>0) and (-q[0],-q[1],q[2]) in candidates]
    pairs.sort(key=lambda q:(q[0]**2+q[1]**2,q))
    allowed=Iz_left;spent=Q(0);count=0;vol=2*WIDTH**3
    for x,y,z in pairs:
        dz=vol*(Q(x*x+y*y,DENOM**2)+WIDTH**2/6)
        if spent+dz>allowed:break
        count+=1;spent+=dz
    mass=full[0]-sum(exact_volumes)-vol*count
    packet=json.dumps(pairs[:count],separators=(',',':')).encode()
    if pair_centers is not None:pair_centers.write_bytes(packet)
    return dict(schema='w1-m34-symmetric-cell-carve-witness',small_box_witness_sha256=sha(box_witness.read_bytes()),
                lattice_denominator=DENOM,center_index_range=RANGE,width=str(WIDTH),certified_free_components_inspected=num,
                all_interior_cell_count=all_cells.count(b'\n'),all_interior_cell_sha256=sha(all_cells),
                balanced_pairs_available=len(pairs),balanced_pairs_removed=count,removed_pair_centers_hash=sha(packet),distinct_removed_cavities=9+2*count,
                retained_mass_fraction=float(mass/full[0]),retained_axial_inertia_fraction=float((Q(9,10)*Iref+allowed-spent)/Iref),
                signed_first_x_first_y_xz_yz=['0']*4,unit_density=True, minimum_gap_between_lattice_boxes_mm=float((Q(1,DENOM)-WIDTH)*25),
                construction='9 exactly balanced cuboids plus equal-volume paired cubes. Grid occupancy is proven by conservative exclusion of triangle AABBs and exact winding on each connected set of remaining cells. One finite family sampled; mathematical only.')


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('folder',type=Path);a.add_argument('auditdir',type=Path)
    a.add_argument('box_witness',type=Path);a.add_argument('--pair-centers',type=Path);a.add_argument('--eligible',type=Path);args=a.parse_args()
    print(json.dumps(extend(args.folder,args.auditdir,args.box_witness,args.pair_centers,args.eligible),indent=2))
