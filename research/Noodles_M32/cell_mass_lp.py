"""Mass optimum over a conservative family of disjoint height-variable cuboids inside M32 D'."""
import argparse
from fractions import Fraction as Q
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

CELLS_SHA = "be78242bb9675aa2c69a20c83f31ffef9b378946fab95f6b84cd2519a98be29d"
TEN_SHA = "5912e9f1dc44378919f5c00690983d284189edadc857ea3b354f09c4b86a1986"
TAU = Q(9, 10)
GRID = 32
TIP_Z = Q.from_float(-0.16)  # the registered binary64 model coordinate


def enclosure(s, digits=18):
    n, d = [int(x,0) for x in s.split("/")]
    if d <= 0:
        raise ValueError("invalid raw moment")
    scale = 10**digits
    return Q(n*scale//d,scale), Q(-((-n*scale)//d),scale)


def inverse(mat):
    n = len(mat)
    rows = [list(row) + [Q(i==j) for j in range(n)] for i,row in enumerate(mat)]
    for i in range(n):
        pivot = next((j for j in range(i,n) if rows[j][i]),None)
        if pivot is None:
            raise ValueError("singular basic matrix")
        rows[i], rows[pivot] = rows[pivot], rows[i]
        rows[i] = [x/rows[i][i] for x in rows[i]]
        for j in range(n):
            if j!=i and rows[j][i]:
                rows[j] = [x-rows[j][i]*y for x,y in zip(rows[j],rows[i])]
    return [row[n:] for row in rows]


def affine_interval(weights, intervals):
    bounds = [(a*lo,a*hi) if a>=0 else (a*hi,a*lo) for a,(lo,hi) in zip(weights,intervals)]
    return sum(x for x,_ in bounds),sum(y for _,y in bounds)


def grid_family(eligible, coarse):
    if not coarse:
        return sorted(eligible),Q(7,256)
    # Every third original cell along each axis keeps its W1 interior certificate.
    return sorted(p for p in eligible if all(i%3==0 for i in p)),Q(7,256)


def coefficients(index,width):
    x,y,z = [Q(i, GRID) for i in index]
    return x,y,x*z,y*z,x*x+y*y+width*width/6


def solve(centers, width, moments):
    if len(centers)<4:
        raise ValueError("insufficient certified cells")
    cap = width**3
    v0 = moments[0]
    inert = tuple(x+y for x,y in zip(moments[4],moments[5]))
    targets = [moments[i] for i in (1,2,8,7)] + [tuple((1-TAU)*x for x in inert)]
    cols = [coefficients(c,width) for c in centers]
    mat = np.array(cols,dtype=float).T
    rhs = np.array([float((a+b)/(2*cap)) for a,b in targets])
    fit = linprog(-np.ones(len(centers)),A_eq=mat[:4],b_eq=rhs[:4],A_ub=mat[4:],
                  b_ub=rhs[4:],bounds=(0,1),method="highs-ds")
    if not fit.success:
        raise ValueError(f"search LP has no candidate basis: {fit.message}")
    upper = [i for i,x in enumerate(fit.x) if x>1-1e-7]
    free = [i for i,x in enumerate(fit.x) if 1e-7 < x < 1-1e-7]
    n = len(free)
    if n not in (4,5):
        raise ValueError(f"uncertified degenerate basis ({n} free)")
    basic = [[cols[i][k] for i in free] for k in range(n)]
    inv = inverse(basic)
    residual = []
    for k in range(n):
        fixed = cap*sum(cols[i][k] for i in upper)
        residual.append(tuple(x-fixed for x in targets[k]))
    volumes = [affine_interval(row,residual) for row in inv]
    if any(not 0<lo<=hi<cap for lo,hi in volumes):
        raise ValueError("uncertified basic volume")
    # v_B = inv * (b - G_U * cap) in exact rational arithmetic.
    # Column sums are dual multipliers of the equality/active-inertia rows.
    dual = [sum(col) for col in zip(*inv)] + ([Q(0)] if n==4 else [])
    if dual[4] < 0:
        raise ValueError("inertia multiplier has wrong sign")
    ups,fr = set(upper),set(free)
    fixed_or_free = ups | fr
    for i,col in enumerate(cols):
        reduced = 1-sum(a*g for a,g in zip(dual,col))
        if (i in fr and reduced!=0) or (i in ups and reduced<0) or (i not in fixed_or_free and reduced>0):
            raise ValueError("exact reduced-cost certificate failed")
    if n==4:
        fixed_I=cap*sum(cols[i][4] for i in upper)
        rem_I=tuple(fixed_I+x for x in affine_interval([cols[i][4] for i in free],volumes))
        if rem_I[1]>targets[4][0]:
            raise ValueError("inactive inertia budget exceeded")
        retained_I=(1-rem_I[1]/inert[0],1-rem_I[0]/inert[1])
    else:
        retained_I=(TAU,TAU)
    removed=tuple(len(upper)*cap+sum(v[k] for v in volumes) for k in (0,1))
    ratio=(1-removed[1]/v0[0],1-removed[0]/v0[1])
    # Rotational observables from the same retained material. At exact x=y=0
    # its z principal inertia is Qxx+Qyy. The transverse block is bounded
    # above by its largest Gershgorin endpoint in the centroidal frame.
    def box_raw(i, v):
        x,y,z = [Q(k,GRID) for k in centers[i]]
        return (v*z,v*(x*x+width*width/12),v*(y*y+width*width/12),
                v*z*z+v**3/(12*width**4),v*x*y)
    upper_sum = [sum(vs) for vs in zip(*(box_raw(i,cap) for i in upper))] if upper else [Q(0)]*5
    removed_lo,removed_hi=upper_sum.copy(),upper_sum.copy()
    for i,(lo,hi) in zip(free,volumes):
        for k,(left,right) in enumerate(zip(box_raw(i,lo),box_raw(i,hi))):
            removed_lo[k]+=min(left,right)
            removed_hi[k]+=max(left,right)
    lz,xx,yy,zz,xy=[(moments[k][0]-high,moments[k][1]-low)
                      for k,high,low in zip((3,4,5,6,9),removed_hi,removed_lo)]
    retained_mass = v0[0]-removed[1],v0[1]-removed[0]
    # Use all corners to support arbitrary signed first moments.
    cz_vals = [v/m for v in lz for m in retained_mass]
    min_lz2 = 0 if lz[0]<=0<=lz[1] else min(v*v for v in lz)
    iz_lo=xx[0]+yy[0]
    transverse_hi = max(xx[1],yy[1]) + zz[1] - min_lz2/retained_mass[1]+max(map(abs,xy))
    lower_height=min(v[0] for v in volumes)/width**2
    return dict(centroid_height_above_registered_tip_mm_interval=[str((v-TIP_Z)*25) for v in (min(cz_vals),max(cz_vals))],
                axial_principal_dominance_gap_lower=str(iz_lo-transverse_hi),candidate_cells=len(centers),width_model=str(width),capacity_per_cell=str(cap),
                optimal_in_this_finite_family=True,unrestricted_or_quadric_optimum=False,
                full_height_index_ranges=[[g[0][1],g[-1][1]] for g in
                    (list(items) for _,items in groupby(enumerate(upper), key=lambda t:t[1]-t[0]))],
                variable_height_centers=[centers[i] for i in free],
                basic_inverse=[list(map(str,row)) for row in inv],
                dual=list(map(str,dual)), four_balance_equalities=["0"]*4,
                basic_volume_intervals=[[str(lo),str(hi)] for lo,hi in volumes],
                minimum_nonempty_cavity_height_mm_approx=float(lower_height*25),
                inertia_active=(n==5), retained_inertia_ratio=[str(x) for x in retained_I],
                retained_mass_ratio=[str(x) for x in ratio], retained_mass_ratio_approx=float(sum(ratio)/2),
                retained_inertia_ratio_approx=float(sum(retained_I)/2),removed_cavities=len(upper)+n)


def replay(cells_path,ten_path):
    data=cells_path.read_bytes()
    if sha256(data).hexdigest()!=CELLS_SHA:
        raise ValueError("unexpected W1 cell packet")
    rows=[tuple(map(int,s.split())) for s in data.splitlines()]
    if rows!=sorted(set(rows)) or not all(len(r)==3 for r in rows):
        raise ValueError("malformed or duplicate cell index")
    raw=ten_path.read_bytes()
    if sha256(raw).hexdigest()!=TEN_SHA:
        raise ValueError("unexpected full-body reference moments")
    moments=list(map(enclosure,raw.decode("ascii").splitlines()))
    out={}
    for coarse,name,stride in [(False,"fine",1),(True,"spaced",3)]:
        centers,width=grid_family(set(rows),coarse)
        case=solve(centers,width,moments)
        case["minimum_box_spacing_mm"]=str((Q(stride,GRID)-width)*25)
        out[name]=case
    return dict(schema="w4-m32-certified-cell-mass-lp-v1",cells_sha256=CELLS_SHA,
                initial_ten_sha256=TEN_SHA,unit_density_model=True,required_inertia=str(TAU),designs=out)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cells",type=Path);p.add_argument("A_ten",type=Path)
    a=p.parse_args()
    print(json.dumps(replay(a.cells,a.A_ten),indent=2))
