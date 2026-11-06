"""Sample exact oriented M32 design membership at the frozen W6 8^3 lattice."""
import argparse
from itertools import product
import json
import sys
from pathlib import Path
import numpy as np

from .m35_crossing_newton import DSHA, digest


def sample(design_path, w1):
    sys.path.insert(0, str(w1.resolve()))
    from m34_noodles_partition import point_winding, Q

    raw = design_path.read_bytes()
    if digest(raw) != DSHA:
        raise ValueError("unexpected design operand")
    rows = raw.decode().splitlines()
    nv, nf = map(int, rows[0].split())
    vertices = [tuple(map(Q, row.split())) for row in rows[1:1+nv]]
    faces = [tuple(map(int, row.split())) for row in rows[1+nv:]]
    if len(faces) != nf:
        raise ValueError("incomplete triangle mesh")
    tip_z = Q(-.16)
    center = (Q(0), Q(0), tip_z + 1)
    if any(abs(x - c) > 1 for v in vertices for x, c in zip(v, center)):
        raise ValueError("voxel cube does not contain whole design")
    triangles = np.asarray(vertices, dtype=float)[faces]
    lower = np.nextafter(triangles.min(axis=1), -np.inf)
    upper = np.nextafter(triangles.max(axis=1), np.inf)
    axis = [Q(float(x)) for x in np.linspace(-1, 1, 8)]
    cells = []
    for p in product(axis, repeat=3):
        point = tuple(x+c for x,c in zip(p,center))
        w = point_winding(vertices, faces, point, lower, upper)
        if w not in (0, 1):
            raise ValueError("nonbinary design winding")
        cells.append(w)
    return dict(schema="W0_M35_DPRIME_8_CUBED", design_sha256=DSHA,
                axis=[float(x) for x in axis], indexing="ij", cube_center_in_tip=[0,0,1],
                cube_halfwidth=1, occupied=sum(cells), occupancy=np.reshape(cells,(8,8,8)).tolist())


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("design","w1","output"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    a.output.write_text(json.dumps(sample(a.design,a.w1),separators=(",",":"))+"\n")
