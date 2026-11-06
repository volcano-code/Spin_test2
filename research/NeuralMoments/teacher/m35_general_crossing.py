"""Evaluate the V7 general finite map on untouched exact-rational M32 design material."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from m35_exact_crossing import D_SHA, TIP, digest, dump

BOUNDS_SHA = "67ddc8327971dd0ca04ba344cc0011e3129fd2f14097ca891ae669e740715575"
DESIGN_LO = (-29, -24, 16)  # conservative model-frame dyadic AABB, denominator 32
DESIGN_HI = (29, 24, 41)
DESIGN_FACES = 94150


def bounds_input(design, dest):
    if digest(design) != D_SHA:
        raise ValueError("wrong exact model")
    rows = design.decode("ascii").splitlines()
    nv = int(rows[0].split()[0])
    rational = [[Fraction(x) for x in row.split()] for row in rows[1:nv+1]]
    if any(not lo <= 32*x <= hi for v in rational for lo,x,hi in zip(DESIGN_LO,v,DESIGN_HI)):
        raise ValueError("rational design outside the declared AABB")
    # The stock source needs bounds on the *tip-frame* design to truncate unbounded surfaces.
    # These rounded vertices never enter the Boolean operand.
    vertices = np.array(rational, dtype=float) - TIP
    np.savetxt(dest, vertices, header=str(nv), comments="", fmt="%.17g")
    if digest(dest.read_bytes()) != BOUNDS_SHA:
        raise ValueError("unexpected represented source bounds")


def cap_box_disjoint(source):
    """Count caps separated exactly from the conservative rational design AABB on some axis."""
    count = 0
    for face, cap in zip(source["faces"], source["face_is_cap"], strict=True):
        if cap:
            scaled = [[Fraction(x)*32 for x in source["vertices"][j]] for j in face]
            axes = zip(*scaled)
            count += any(max(a) < lo or min(a) > hi for a, lo, hi in zip(axes, DESIGN_LO, DESIGN_HI))
    return count


def general_evaluate(design, raw, folder, csg, source, bounds):
    if digest(design) != D_SHA or digest(bounds.read_bytes()) != BOUNDS_SHA:
        raise ValueError("incorrect exact design or represented source bounds")
    raw = np.array(raw, dtype=float)
    if raw.shape != (10,) or not np.isfinite(raw).all() or np.linalg.norm(raw) == 0:
        raise ValueError("a nonzero finite ten-vector is required")
    folder.mkdir(parents=True, exist_ok=True)
    source_hash = digest(source.read_bytes())
    csg_hash = digest(csg.read_bytes())
    text = subprocess.check_output([str(source), str(bounds)], input=" ".join(map(repr,map(float,raw))), text=True)
    lines = text.splitlines()
    head = lines[0].split()
    if len(head) != 18:
        raise ValueError("build the version reporting the actual Eigen-normalized coefficients")
    native_q = list(map(float, head[8:]))
    nv, nf, inside = map(int, head[:3])
    radius = float(head[3])
    table = np.array([list(map(float, row.split())) for row in lines[1:nv+1]])
    fc = np.array([list(map(int, row.split())) for row in lines[nv+1:]], dtype=int)
    if table.shape != (nv,33) or not np.isfinite(table).all() or fc.shape != (nf,4):
        raise ValueError("malformed source")
    tip_xyz = table[:,:3]
    xyz = tip_xyz + TIP
    jacobian = table[:,3:].reshape(nv,3,10)
    faces, cap = fc[:,:3], fc[:,3]
    if (faces < 0).any() or (faces >= nv).any() or not np.isin(cap,[0,1]).all() or sum(cap) != int(head[5]):
        raise ValueError("invalid original face or cap provenance")
    s = dict(raw_q=raw.tolist(), q=native_q, vertices_tip=tip_xyz.tolist(),
             vertices=xyz.tolist(), faces=faces.tolist(), native_jacobian=jacobian.tolist(), face_is_cap=cap.tolist(),
             interior_is_positive=bool(inside), truncation_radius=radius, surface_type=head[4],
             source_native_hash=head[6], triangulation_hash=head[7], bound_reference_sha256=BOUNDS_SHA)
    def token(v):
        n, d = float(v).as_integer_ratio()
        return f"{n}/{d}"
    packet = ("\n".join([f"{nv} {nf}", *(" ".join(map(token,p)) for p in xyz),
                        *(" ".join(map(str,f)) for f in faces)]) + "\n").encode()
    dump(folder/"source.json", s)
    (folder/"cutter.txt").write_bytes(packet)
    combined = folder/"input.txt"
    combined.write_bytes(design+packet)
    op = "intersection" if inside else "difference"
    pending, result = folder/"pending.json", folder/"result.json"
    start = time.monotonic()
    subprocess.run([str(csg), str(combined), str(pending), op], check=True)
    elapsed = time.monotonic() - start
    obj = json.loads(pending.read_bytes())
    parents = obj["birth_faces"]
    if not obj["succeeded"] or obj["operation"] != op or len(parents) != len(obj["faces"]):
        raise ValueError("bad Boolean")
    if any(p < 0 or p >= DESIGN_FACES + nf for p in parents):
        raise ValueError("parent outside operand range")
    if any(p >= DESIGN_FACES and cap[p - DESIGN_FACES] for p in parents):
        raise ValueError("truncation cap in output")
    pending.replace(result)
    combined.unlink()
    rec = dict(operation=op, design_sha256=D_SHA, cutter_sha256=digest(packet),
               source_sha256=digest((folder/"source.json").read_bytes()), source_executable=source_hash,
               csg_executable=csg_hash, result_sha256=digest(result.read_bytes()),
               faces=len(parents), vertices=len(obj["vertices"]), design_parent_faces=sum(i<DESIGN_FACES for i in parents),
               cutter_parent_faces=sum(i>=DESIGN_FACES for i in parents), cap_output_faces=0, seconds=elapsed,
               source_type=head[4])
    dump(folder/"receipt.json", rec)
    return rec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("q", "design", "csg", "source", "bounds", "output"):
        parser.add_argument("--"+key, required=True, type=Path)
    args = parser.parse_args()
    design = args.design.read_bytes()
    if not args.bounds.exists():
        bounds_input(design, args.bounds)
    print(general_evaluate(design, json.loads(args.q.read_text()), args.output, args.csg, args.source, args.bounds))
