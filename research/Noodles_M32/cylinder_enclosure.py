"""Build rational inner and outer prisms for one continuous oblique quadric cavity."""
from fractions import Fraction as Q
import argparse
import math
from pathlib import Path

# Frozen after a 512-sided design-only search, in the unchanged M32 model frame.
R, TX, TY, SX, SY = map(Q, ("0.501199840194", "0.000374697304", "-0.000369489108",
                              "-0.001561582753", "-0.000150924463"))
ZREF = Q(9, 10)
N = 256
CYLINDER = (R,TX,TY,SX,SY,Q(0))
CONE = tuple(map(Q, ("0.501132242410", "0.000361234915", "-0.000434042424",
                     "-0.001828120807", "-0.000000342463", "0.032733310987")))


def ring(outer, n=N):
    if n < 4:
        raise ValueError("at least four polygon sides required")
    scale = (1.000000001 / math.cos(math.pi/n)) if outer else 0.999999999
    points = [(Q(round(float(R)*scale*math.cos(2*math.pi*i/n)*2**44), 2**44),
               Q(round(float(R)*scale*math.sin(2*math.pi*i/n)*2**44), 2**44))
              for i in range(n)]
    # Verify the convex CCW polygon with exact predicates, independently of libm.
    for p, q in zip(points, points[1:]+points[:1]):
        dx, dy = q[0]-p[0], q[1]-p[1]
        if any(dx*(v-p[1])-dy*(u-p[0])<0 for u,v in points):
            raise ValueError("nonconvex ring")
        k = p[0]*q[1]-q[0]*p[1]
        if k <= 0:
            raise ValueError("origin not strictly interior")
        if outer and k*k <= R*R*(dx*dx+dy*dy):
            raise ValueError("outer support does not enclose circle")
        if not outer and p[0]*p[0]+p[1]*p[1] >= R*R:
            raise ValueError("inner ring outside circle")
    return points


def packet(outer, n=N, shape=CYLINDER):
    radius,tx,ty,sx,sy,slope=shape
    xy=ring(outer, n)
    tip,top=Q(0),Q(2)
    pts=[]
    for z in (tip,top):
        scale=(radius+slope*(z-ZREF))/R
        if scale<=0: raise ValueError("nonpositive radius in truncation slab")
        pts.extend((u*scale+tx+sx*(z-ZREF),v*scale+ty+sy*(z-ZREF),z) for u,v in xy)
    # End fans use existing perimeter vertices and do not enter the M32 slab.
    faces=[(0,i+1,i) for i in range(1,n-1)]+[(n,n+i,n+i+1) for i in range(1,n-1)]
    for i in range(n):
        j=(i+1)%n
        faces.extend([(i,j,n+j),(i,n+j,n+i)])
    return ("\n".join([f"{len(pts)} {len(faces)}",*(" ".join(f"{q.numerator}/{q.denominator}" for q in p) for p in pts),
                       *(" ".join(map(str,f)) for f in faces)])+"\n").encode()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind",choices=["inner","outer"])
    p.add_argument("output",type=Path)
    p.add_argument("--cone",action="store_true")
    a=p.parse_args()
    a.output.write_bytes(packet(a.kind=="outer",shape=CONE if a.cone else CYLINDER))
