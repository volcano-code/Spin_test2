"""Exact oriented winding on each x-directed scanline of a uniform world cube."""
import argparse
import json
from pathlib import Path

import numpy as np

from m34_noodles_partition import Q, point_winding
from m35_noodles_crossing import packet

NOODLES = "c3633c74bc74c2be488eb22de108454276f28941f0477cfee5837f57db99d622"
BUNNY = "baaabf7f4ccfe8f4d4100ba2b977adb373f8e5bbc088a4b6469d3d1090dc7a13"


def sample(design, side):
    digest, vv, faces = packet(design)
    if digest == NOODLES:
        offset = (Q(0), Q(0), Q(-.16)+1)
        tip_center = [0, 0, 1]
    elif digest == BUNNY:
        offset = (Q(0),)*3
        tip_center = [0, 0, 0]
    else:
        raise ValueError("unregistered world frame")
    tri = np.asarray(vv, dtype=float)[faces]
    lower = np.nextafter(tri.min(1), -np.inf)
    upper = np.nextafter(tri.max(1), np.inf)
    nodes = np.linspace(-1,1,side)
    xs, ys, zs = [[s + Q(float(x)) for x in nodes] for s in offset]
    grid = np.zeros((side,side,side),dtype=np.uint8)
    for j,y in enumerate(ys):
        possible = (lower[:,1]<=float(y)) & (upper[:,1]>=float(y))
        for k,z in enumerate(zs):
            candidates = possible & (lower[:,2]<=float(z)) & (upper[:,2]>=float(z))
            crossings = []
            ambiguous = False
            for i in np.flatnonzero(candidates):
                a,b,c = (vv[v] for v in faces[i])
                n = (b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1])
                if n == 0:
                    # An edge-on triangle can be omitted only if the ray misses its projected line.
                    if all((b[1]-a[1])*(z-a[2])==(b[2]-a[2])*(y-a[1]) for a,b in ((a,b),(b,c),(c,a))):
                        ambiguous = True
                        break
                    continue
                b_weight = ((y-a[1])*(c[2]-a[2])-(z-a[2])*(c[1]-a[1]))/n
                c_weight = ((b[1]-a[1])*(z-a[2])-(b[2]-a[2])*(y-a[1]))/n
                weights = (1-b_weight-c_weight,b_weight,c_weight)
                if min(weights)<0:
                    continue
                if min(weights)==0:
                    ambiguous = True
                    break
                x_cut = sum(w*v[0] for w,v in zip(weights,(a,b,c)))
                crossings.append((x_cut,1 if n>0 else -1))
            for i,x in enumerate(xs):
                if ambiguous or any(h==x for h,_ in crossings):
                    value = point_winding(vv,faces,(x,y,z),lower,upper)
                else:
                    value = sum(s for h,s in crossings if h>x)
                if value not in (0,1):
                    raise ValueError("nonbinary design winding")
                grid[i,j,k] = value
    return dict(design_sha256=digest, axis=nodes.tolist(), indexing="ij", cube_center_in_tip=tip_center,
                cube_halfwidth=1, occupied=int(grid.sum()), occupancy=grid.tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("design",type=Path)
    p.add_argument("side",type=int)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.write_text(json.dumps(sample(a.design,a.side),separators=(",",":"))+"\n")
