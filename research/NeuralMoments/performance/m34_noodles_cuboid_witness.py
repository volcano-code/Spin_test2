"""A rational physical-equality witness carved out of M32; the cavities are independent boxes."""
import argparse
import json
from pathlib import Path
import numpy as np
from m34_noodles_partition import Q, point_winding, sha, surface_data, token
from exact_planar_patches import cross, dot, subtract

# Dyadic centers from a coarse search on the registered design. Four heights
# are solved exactly, the others use a fixed feasible volume. All box faces
# are rechecked against the current rational surface. No fitted mesh is used.
CENTERS = ((11,6,13),(12,-1,16),(12,-1,17),(12,1,14),(12,1,15),
           (12,1,16),(12,1,17),(12,4,14),(12,4,15))
UNKNOWN = (0,1,3,6)
HALF = Q(1,64)
FIXED_VOLUME = Q(9,10)*(2*HALF)**3


def solve_four(matrix, rhs):
    rows=[list(a)+[b] for a,b in zip(matrix,rhs)]
    for i in range(4):
        pivot=next(j for j in range(i,4) if rows[j][i])
        rows[i],rows[pivot]=rows[pivot],rows[i]
        rows[i]=[x/rows[i][i] for x in rows[i]]
        for j,row in enumerate(rows):
            if j!=i: rows[j]=[x-row[i]*y for x,y in zip(row,rows[i])]
    return [r[-1] for r in rows]


def box_column(center):
    x,y,z=center
    return x,y,x*z,y*z


def volume_solution(centers, total):
    cols=[box_column(c) for c in centers]
    target=[total[i] for i in (1,2,8,7)]
    rhs=[v-FIXED_VOLUME*sum(cols[i][j] for i in range(len(cols)) if i not in UNKNOWN)
         for j,v in enumerate(target)]
    solved=solve_four([[cols[i][j] for i in UNKNOWN] for j in range(4)],rhs)
    volumes=[FIXED_VOLUME]*len(cols)
    for i,val in zip(UNKNOWN,solved): volumes[i]=val
    if any(not 0<v<(2*HALF)**3 for v in volumes):raise ValueError('WITNESS_EXTENT_EXCEEDS_CERTIFIED_BOX')
    if target != [sum(v*col[j] for v,col in zip(volumes,cols)) for j in range(4)]:
        raise ValueError('EQUALITY_RESIDUAL_NONZERO')
    return volumes


def box_disjoint(tri, point, half):
    """Exact separating-axis test for a closed triangle and an axis-aligned box."""
    triangle=[subtract(q,point) for q in tri]
    edges=[subtract(b,a) for a,b in zip(triangle,triangle[1:]+triangle[:1])]
    units=[(1,0,0),(0,1,0),(0,0,1)]
    axes=units+[cross(edges[0],edges[1])]+[cross(u,e) for e in edges for u in units]
    for axis in axes:
        projections=[dot(axis,v) for v in triangle]
        radius=half*sum(map(abs,axis))
        if min(projections)>radius or max(projections)<-radius:return True
    return False


def certify_interior(vertices, faces, centers):
    # Binary64 only prunes candidates. Bounds expand one ulp in both directions.
    # Exact dyadic centers and the enclosing dyadic cubes are represented without rounding.
    vv=np.array(vertices,dtype=float)
    triangles=vv[np.array(faces)]
    lower=np.nextafter(triangles.min(axis=1),-np.inf)
    upper=np.nextafter(triangles.max(axis=1), np.inf)
    radius=float(HALF); windings=[]
    for point in centers:
        p=np.array(point,dtype=float)
        potential=np.flatnonzero(((lower<=p+radius)&(upper>=p-radius)).all(axis=1))
        for idx in potential:
            if not box_disjoint([vertices[i] for i in faces[idx]],point,HALF):
                raise ValueError('BOX_TOUCHES_DESIGN_BOUNDARY')
        winding=point_winding(vertices, faces, point, lower, upper)
        if winding!=1:raise ValueError('BOX_NOT_INSIDE_SINGLE_WINDING')
        windings.append(winding)
    for i,p in enumerate(centers):
        for q in centers[:i]:
            if max(abs(a-b) for a,b in zip(p,q))<=2*HALF:raise ValueError('BOXES_NOT_DISJOINT')
    return windings


def box_ten(center,volume):
    x,y,z=center; h=volume/(8*HALF**2)
    return tuple(volume*i for i in (1,x,y,z,x*x+HALF**2/3,y*y+HALF**2/3,z*z+h*h/3,y*z,x*z,x*y))


def witness(folder, auditdir, volume_packet=None):
    cert=json.loads((auditdir/'A.w1.json').read_text())
    a_hash,_,_=surface_data(folder/'A.json')
    if cert['source_sha256']!=a_hash:raise ValueError('A_INPUT_DRIFT')
    raw=(auditdir/'A.ten').read_bytes()
    if sha(raw)!=cert['exact_ten_payload_sha256']:raise ValueError('A_INTEGRAL_DRIFT')
    full=list(map(token,raw.decode().splitlines()))
    d_hash,v,faces=surface_data(folder/'Dprime.json')
    centers=[tuple(Q(s,16) for s in c) for c in CENTERS]
    volumes=volume_solution(centers,full)
    windings=certify_interior([tuple(map(Q,p)) for p in v],faces,centers)
    moments=[box_ten(c,vol) for c,vol in zip(centers,volumes)]
    retained=[u-sum(m[i] for m in moments) for i,u in enumerate(full)]
    if any(retained[i] for i in (1,2,7,8)):raise ValueError('FOUR_ORIGINAL_EQUALITIES_FAILED')
    ratio=(retained[4]+retained[5])/(full[4]+full[5])
    if ratio<Q(9,10):raise ValueError('NINETY_PERCENT_INERTIA_FAILED')
    packet=('\n'.join(f'{vol.numerator:#x}/{vol.denominator:#x}' for vol in volumes)+'\n').encode()
    if volume_packet is not None:volume_packet.write_bytes(packet)
    return dict(schema='w1-m34-general-cavity-physical-witness',A_sha256=a_hash,Dprime_sha256=d_hash,
                exact_ordered_volume_packet_sha256=sha(packet),source_ten_sha256=sha(raw),removal_box_count=len(volumes),windings=windings,
                mass_fraction=float(retained[0]/full[0]),axial_inertia_fraction=float(ratio),
                centroid_height_model=float(retained[3]/retained[0]),Cx_Cy_Ixz_Iyz=['0']*4,
                box_volume_model=list(map(float,volumes)),removed_volume_sha256=[
                    sha(f'{v.numerator:#x}/{v.denominator:#x}'.encode()) for v in volumes],
                exact_positive_volumes=True, disjoint_boxes_strictly_inside_Dprime=True,
                scope='Unrestricted axis-aligned box cavities, excludes strict quadric admissibility, native terminal, optimum, manufacturing and dynamical validation.')


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('folder',type=Path);a.add_argument('auditdir',type=Path)
    a.add_argument('--volume-packet',type=Path)
    opt=a.parse_args();print(json.dumps(witness(opt.folder,opt.auditdir,opt.volume_packet),indent=2))
