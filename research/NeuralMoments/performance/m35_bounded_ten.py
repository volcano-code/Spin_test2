"""Signed-origin tetrahedron integration with rational per-term outward rounding."""
import argparse
import json
from pathlib import Path

from m34_noodles_partition import Q, surface_data
from exact_planar_patches import cross, dot


def integrate(vertices, faces, bits=112):
    if bits not in range(32,257):
        raise ValueError('unsupported enclosure precision')
    vv=[tuple(Q(x) for x in v) for v in vertices]
    pairs=((0,0),(1,1),(2,2),(1,2),(0,2),(0,1))
    scale=1<<bits
    # A common integer denominator 120*2^bits. Quantize each unscaled
    # monomial numerator outward before dividing by the tetrahedron factor.
    inner=[0]*10
    for ia,ib,ic in faces:
        a,b,c=vv[ia],vv[ib],vv[ic]
        v=dot(a,cross(b,c))
        s=tuple(x+y+z for x,y,z in zip(a,b,c))
        fs=[20*v, *[5*v*x for x in s],
            *[v*(s[i]*s[j]+a[i]*a[j]+b[i]*b[j]+c[i]*c[j]) for i,j in pairs]]
        for k,u in enumerate(fs):inner[k]+=(u.numerator*scale)//u.denominator
    den=120*scale
    return [(Q(v,den),Q(v+len(faces),den)) for v in inner]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('packet',type=Path);p.add_argument('--bits',type=int,default=112)
    a=p.parse_args();digest,v,f=surface_data(a.packet);out=integrate(v,f,a.bits)
    print(json.dumps(dict(source_sha256=digest,bits=a.bits,unit_density=True,
        frame='model origin, retained variable region',ten_bounds=[[str(l),str(r)] for l,r in out],
        ten=[float((l+r)/2) for l,r in out]),indent=2))
