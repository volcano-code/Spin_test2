"""Primal construction and all-subsets dual bound for mass removal inside the fixed M32 D'."""
import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path

from cell_mass_lp import CELLS_SHA, inverse
from cylinder_enclosure import R, N, CYLINDER, CONE, ZREF, packet, ring
from single_quadric_witness import I, D_PRIME, A_TEN, enclose_hex

PAIRS = ((21,-7,34), (6,-17,22), (11,16,33), (20,2,23))
WIDTH = Q(7,256)
FOUR = (1,2,8,7)
EPS_VOLUME = Q(1,10**7)


def sup(i):
    return max(abs(i.lo),abs(i.hi))


def bounds(i):
    scale=10**14
    return [str(Q((i.lo*scale)//1,scale)), str(Q(-((-i.hi*scale)//1),scale))]


def check_hash(path, expected):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=expected:
        raise ValueError(f"input hash mismatch for {path.name}")
    return raw


def load_data(root, design_path, a_ten, cell_path):
    rows=check_hash(design_path,D_PRIME).splitlines()
    nv,nf=map(int,rows[0].split())
    pts=[tuple(Q(w.decode()) for w in line.split()) for line in rows[1:nv+1]]
    if len(rows)!=nv+nf+1:raise ValueError("invalid source mesh")
    extrema=[I(min(p[j] for p in pts),max(p[j] for p in pts)) for j in range(3)]
    if extrema[2].lo<=0 or extrema[2].hi>=2:raise ValueError("truncation intersects design")
    cells=set(tuple(map(int,l.split())) for l in check_hash(cell_path,CELLS_SHA).splitlines())
    whole=[enclose_hex(l.decode()) for l in check_hash(a_ten,A_TEN).splitlines()]
    return whole,cells,extrema


def read_packet(root,kind,shape):
    pinned=Path(__file__).with_name("m36_quadric")/(kind+".bounds")
    rec=json.loads(check_hash(root/(kind+'.bounds'),sha256(pinned.read_bytes()).hexdigest()))
    file=root/(kind+'.json')
    raw=check_hash(file, rec['source_sha256'])
    obj=json.loads(raw)
    if not obj['succeeded'] or obj['operation']!='intersection' or rec['bits']!=112:
        raise ValueError("invalid intersection or cubature metadata")
    if not (root/(kind+'.txt')).read_bytes()==packet(kind.endswith('outer'),shape=shape):
        raise ValueError("cutter bytes differ")
    nums=[I(Q(l),Q(h)) for l,h in rec['ten_bounds']]
    if len(nums)!=10 or any(x.hi<x.lo or x.hi-x.lo>Q(1,2**85) for x in nums):
        raise ValueError("moment enclosures invalid")
    return nums,{'result_sha256':sha256(raw).hexdigest(),
                 'cutter_sha256':sha256((root/(kind+'.txt')).read_bytes()).hexdigest(),
                 'bounds_sha256':sha256((root/(kind+'.bounds')).read_bytes()).hexdigest(),
                 'faces':len(obj['faces']),'vertices':len(obj['vertices'])}


def metrics(total,whole):
    m,x,y,z,xx,yy,zz,yz,xz,xy=total
    cxx=xx-x*x/m;cyy=yy-y*y/m;czz=zz-z*z/m
    cxy=xy-x*y/m;cxz=xz-x*z/m;cyz=yz-y*z/m
    target=cxx+cyy
    gap=target.lo-max((cxx+czz).hi,(cyy+czz).hi)-sup(cxy)
    if gap<=0:raise ValueError("dominant axial moment not certified")
    # For the top eigenvector, (lambda I-T)u = w v_z and lambda >= Izz.
    sin_limit=(sup(cxz)+sup(cyz))/gap
    return dict(removed_mass_fraction=bounds(1-m/whole[0]),
                inertia_retention=bounds((xx+yy)/(whole[4]+whole[5])),
                fixed_axis_lateral_centroid_L1_upper=bounds(I.exact((sup(x)+sup(y))/m.lo))[1],
                dominant_axis_angle_sine_upper=bounds(I.exact(sin_limit))[1],
                centroidal_spectral_separation_lower=bounds(I.exact(gap))[0],
                height_change_model_units=bounds(z/m-whole[3]/whole[0]),
                raw_balance_residuals=[bounds(total[k]) for k in FOUR])


def correction(inner,whole,cells,shape):
    # Equal widths, centered z heights. First and mixed moments are linear in
    # positive/negative volume difference for each of four opposite xy pairs.
    ctr=[[Q(a,32) for a in row] for row in PAIRS]
    col=[[x,y,x*z,y*z] for x,y,z in ctr]
    transform=inverse(list(zip(*col)))
    target=[whole[k]-inner[k] for k in FOUR]
    delta=[sum(t*b for t,b in zip(row,target)) for row in transform]
    removed=list(inner); pairs=[]
    rad,tx,ty,sx,sy,slope=shape
    for idx,cs,d in zip(PAIRS,ctr,delta):
        volumes=[]
        for sign in (1,-1):
            x,y,z=[cs[0]*sign,cs[1]*sign,cs[2]]
            if (idx[0]*sign,idx[1]*sign,idx[2]) not in cells:raise ValueError("uncertified correction cell")
            dz=I(z-WIDTH/2-ZREF,z+WIDTH/2-ZREF)
            nearest_x=max(0,abs(x)-WIDTH/2-sup(tx+sx*dz))
            nearest_y=max(0,abs(y)-WIDTH/2-sup(ty+sy*dz))
            if nearest_x**2+nearest_y**2<=(rad+slope*dz).hi**2:raise ValueError("cell intersects quadric")
            # Positive constant makes all eight volume variables strictly interior.
            v=I(EPS_VOLUME,max(Q(0),d.hi)+EPS_VOLUME) if sign==1 else I(EPS_VOLUME,max(Q(0),-d.lo)+EPS_VOLUME)
            # v_plus=e+max(0,d), v_minus=e+max(0,-d).
            v=I(EPS_VOLUME+max(Q(0),d.lo if sign==1 else -d.hi), v.hi)
            if v.hi>=WIDTH**3:raise ValueError("required correction too large")
            vol_ten=[v,v*x,v*y,v*z,v*(x*x+WIDTH**2/12),v*(y*y+WIDTH**2/12),
                     v*z*z+v*v*v/(12*WIDTH**4),v*y*z,v*x*z,v*x*y]
            removed=[u+w for u,w in zip(removed,vol_ten)]
            volumes.append(bounds(v))
        pairs.append(dict(positive_grid_index=idx,plus_minus_volume_bounds=volumes))
    retained=[a-b for a,b in zip(whole,removed)]
    if not all(r.lo<=0<=r.hi for i,r in enumerate(retained) if i in FOUR):
        raise ValueError("correction intervals do not enclose target")
    # The exact matrix inverse and v_plus-v_minus=d are symbolic identities.
    for k in FOUR:retained[k]=I.exact(0)
    if ((retained[4]+retained[5])/(whole[4]+whole[5])).lo<Q(9,10):
        raise ValueError("axial inertia budget exceeded by correction")
    return metrics(retained,whole),pairs


def dual(whole,inner,shell,extrema):
    rad,tx,ty,sx,sy,_=CYLINDER
    a,b=tx-sx*ZREF,ty-sy*ZREF; lam=1/rad**2
    q=[1-lam*(a*a+b*b),2*lam*a,2*lam*b,-2*lam*(a*sx+b*sy),-lam,-lam,
       -lam*(sx*sx+sy*sy),2*lam*sy,2*lam*sx,0]
    # q=(1-|p_xy-a(z)|^2/R^2). All points of the circle outside the
    # inner polygon have q <= 1-0.9999^2 by the rational inradius check.
    rho=Q(9999,10000);xy=ring(False)
    for p,t in zip(xy,xy[1:]+xy[:1]):
        edge=(t[0]-p[0],t[1]-p[1])
        if (p[0]*t[1]-t[0]*p[1])**2 < (rho*rad)**2*sum(s*s for s in edge):
            raise ValueError("inner radius bound invalid")
    positive_integral=sum(c*m for c,m in zip(q,inner)).hi+(1-rho**2)*shell
    # Bounding the positive square term by the *whole mass* makes the dual
    # valid for any measurable removable subset obeying the five constraints.
    z=extrema[2]
    bias=sup(a+sx*z)**2+sup(b+sy*z)**2
    correction=a*whole[1]+b*whole[2]+sx*whole[8]+sy*whole[7]
    upper=(positive_integral+lam*((whole[4]+whole[5])/10-2*correction+bias*whole[0]))/whole[0]
    return bounds(upper)[1]


def run(root,design,a_ten,cell_path):
    whole,cells,extrema=load_data(root,design,a_ten,cell_path)
    result={}
    for name,shape,prefix in [('cylinder',CYLINDER,''),('cone',CONE,'cone-')]:
        lo,r0=read_packet(root,prefix+'inner',shape)
        hi,r1=read_packet(root,prefix+'outer',shape)
        shell=hi[0].hi-lo[0].lo
        if shell<0:raise ValueError("inverted mass interval")
        x,y,z=extrema
        mono=(I.exact(1),x,y,z,x*x,y*y,z*z,y*z,x*z,x*y)
        # Monotone squared raw moments have positive shell contribution.
        clip=[l+I(min(0,f.lo)*shell,max(0,f.hi)*shell) for l,f in zip(lo,mono)]
        for k in (0,4,5,6):clip[k]=I(lo[k].lo,hi[k].hi)
        single=metrics([a-v for a,v in zip(whole,clip)],whole)
        # Preserve xz-h*x cancellation within the same unknown shell subset.
        # The true retained center h lies in the independently bounded interval.
        retained=[a-v for a,v in zip(whole,clip)]
        h=retained[3]/retained[0]
        inner_ret=[a-v for a,v in zip(whole,lo)]
        open_z=z-h
        radius,tx,ty,sx,sy,k=shape
        dz=z-ZREF
        xmax=sup(tx+sx*dz)+(radius+k*dz).hi
        ymax=sup(ty+sy*dz)+(radius+k*dz).hi
        cross=sup(inner_ret[8]-inner_ret[1]*h)+sup(inner_ret[7]-inner_ret[2]*h)
        cross+=(xmax+ymax)*sup(open_z)*shell
        gap=Q(single['centroidal_spectral_separation_lower'])
        single['dominant_axis_angle_sine_upper']=bounds(I.exact(cross/gap))[1]
        corrected,pairs=correction(lo,whole,cells,shape)
        result[name]=dict(shape=list(map(str,shape)),inner=r0,outer=r1,
                          true_quadric=single,balanced_inner_polyhedron=corrected,
                          correction_pairs=pairs,shell_volume_upper=bounds(I.exact(shell))[1])
        if name=='cylinder':result['arbitrary_subset_saving_upper']=dual(whole,lo,shell,extrema)
    m,x,y,z,xx,yy,zz,yz,xz,xy=whole
    centered_z=zz-z*z/m
    transverse=max((xx+centered_z).hi,(yy+centered_z).hi)+sup(xy)
    result['dominance_gap_for_all_balanced_90pct_designs_lower']=bounds(I.exact(Q(9,10)*(xx+yy).lo-transverse))[0]
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','design','whole_ten','eligible_cells'):p.add_argument(k,type=Path)
    args=p.parse_args()
    print(json.dumps(run(args.root,args.design,args.whole_ten,args.eligible_cells),indent=2,sort_keys=True))
