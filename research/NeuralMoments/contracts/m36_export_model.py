"""Save the no-rise Noodles material with eight explicit rational, inward-oriented box surfaces."""
import argparse
from fractions import Fraction as Q
from itertools import product
import json
from pathlib import Path
import sys

from .m35_crossing_newton import digest, read_ten
from .m32_mass_bound import FULL_TEN_SHA

WIDTH = Q(7, 256)
# Vertices are in lexicographic (x,y,z) order. All six quads face into the cavity.
BOX_QUADS = ((0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5))


def box(center, volume):
    height = volume / WIDTH**2
    if not 0 < height < WIDTH:
        raise ValueError("box extends outside its certified cell")
    vertices = [[str(c + sign * half) for c,sign,half in zip(center,signs,(WIDTH/2,WIDTH/2,height/2))]
                for signs in product((-1,1),repeat=3)]
    faces = [list(tri) for a,b,c,d in BOX_QUADS for tri in ((a,b,c),(a,c,d))]
    return vertices, faces


def assemble(base, certificate):
    if not base["succeeded"] or base["operation"] != "difference":
        raise ValueError("base must be the exact full solid minus the inner conic clip")
    vertices,faces = list(base["vertices"]),list(base["faces"])
    removed_volumes = []
    pairs = certificate["cone"]["correction_pairs"]
    if len(pairs)!=4:
        raise ValueError("four independent pairs required")
    for pair in pairs:
        x,y,z = [Q(n,32) for n in pair["positive_grid_index"]]
        for sign,(lo,hi) in zip((1,-1),pair["plus_minus_volume_bounds"],strict=True):
            lo,hi=Q(lo),Q(hi)
            if not 0<lo<=hi or hi-lo>Q(1,10**13):
                raise ValueError("uncertified correction volume")
            volume=(lo+hi)/2
            v,f=box((sign*x,sign*y,z),volume)
            offset=len(vertices)
            vertices.extend(v)
            faces.extend([[offset+i for i in triangle] for triangle in f])
            removed_volumes.append(str(volume))
    return dict(schema="m36-rational-saved-no-rise-material-v1",vertices=vertices,faces=faces,
                extra_removed_volumes=removed_volumes)


def run(args):
    raw=args.base.read_bytes()
    if digest(raw)!="63894cb093e5956755e937120dcc9cc1413082a7a7b17579fc5568c825142bdd":
        raise ValueError("unexpected original solid minus conic cavity")
    model=assemble(json.loads(raw),json.loads(args.certificate.read_bytes()))
    args.output.write_text(json.dumps(model,separators=(",",":"))+"\n")
    # Re-read the saved mesh, independently integrate all tetrahedra including the eight cavities.
    sys.path.insert(0,str(args.w1.resolve()))
    from m35_bounded_ten import integrate
    from m34_noodles_partition import point_tip, Q as fast_q
    saved=json.loads(args.output.read_bytes())
    bounds=integrate(saved["vertices"],saved["faces"])
    bounds=[(Q(str(lo)),Q(str(hi))) for lo,hi in bounds]
    if digest(args.whole_ten.read_bytes())!=FULL_TEN_SHA:
        raise ValueError("unbound reference solid")
    whole=read_ten(args.whole_ten)
    mlo,mhi=bounds[0]
    if mlo<=0 or (bounds[4][0]+bounds[5][0])/(whole[4]+whole[5]+2*Q(1,2**190))<Q(9,10):
        raise ValueError("saved model misses inertia gate")
    if bounds[3][1]/mlo>=whole[3]/(whole[0]+Q(1,2**190)):
        raise ValueError("saved model violates no-rise")
    balance=max(abs(x) for k in (1,2,8,7) for x in bounds[k])
    if balance>Q(1,10**12):
        raise ValueError("unexpected saved-model balance error")
    tip=point_tip([tuple(map(fast_q,v)) for v in saved["vertices"]])
    if not tip["exact_fixed_tip"]:
        raise ValueError("registered unique tip changed")
    rec=dict(base_sha256=digest(raw),certificate_sha256=digest(args.certificate.read_bytes()),
             saved_sha256=digest(args.output.read_bytes()), faces=len(saved["faces"]),vertices=len(saved["vertices"]),
             tip=tip, bounds=[[str(lo),str(hi)] for lo,hi in bounds],
             maximum_absolute_four_balance_residual_upper=str(balance),
             removed_mass_fraction=float(1-(mhi+mlo)/2/whole[0]),
             fixed_axis_inertia_fraction=float((bounds[4][0]+bounds[5][0])/(whole[4]+whole[5])),
             height_change_model_units=float(bounds[3][1]/mlo-whole[3]/whole[0]),
             removed_box_heights_model_units=[float(Q(v)/WIDTH**2) for v in saved["extra_removed_volumes"]])
    args.receipt.write_text(json.dumps(rec,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("base","certificate","whole-ten","w1","output","receipt"):
        p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())
