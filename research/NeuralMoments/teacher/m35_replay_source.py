"""Check native source geometry, motion, orientation and exact normalized q without another Boolean."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from m35_exact_crossing import TIP, digest, dump
from m35_general_crossing import BOUNDS_SHA, cap_box_disjoint


def replay(folder, executable, bounds):
    if digest(bounds.read_bytes()) != BOUNDS_SHA:
        raise ValueError("incorrect source bounds")
    s = json.loads((folder/"source.json").read_bytes())
    native = subprocess.check_output([str(executable), str(bounds)], input=" ".join(map(repr,s["raw_q"])), text=True)
    rows = native.splitlines()
    head = rows[0].split()
    nv, nf, inside = map(int, head[:3])
    normalized = list(map(float, head[8:]))
    values = np.array([[float(x) for x in row.split()] for row in rows[1:nv+1]])
    faces = np.array([[int(x) for x in row.split()] for row in rows[nv+1:]])
    if len(normalized) != 10 or values.shape != (nv,33) or faces.shape != (nf,4):
        raise ValueError("unsupported native capture")
    if not (np.array_equal(s["vertices_tip"], values[:,:3])
            and np.array_equal(s["vertices"], values[:,:3]+TIP)
            and np.array_equal(s["native_jacobian"], values[:,3:].reshape(nv,3,10))
            and np.array_equal(s["faces"], faces[:,:3])):
        raise ValueError("native source differs from captured geometry/motion")
    caps = faces[:,3].tolist()
    if "face_is_cap" in s:
        if (s["face_is_cap"] != caps or s["interior_is_positive"] != bool(inside)
                or s["truncation_radius"] != float(head[3]) or s["source_native_hash"] != head[6]
                or s["triangulation_hash"] != head[7] or s["surface_type"] != head[4]):
            raise ValueError("native topology/side differs from capture")
    elif any(caps) or not inside:
        raise ValueError("ellipsoid capture has unexpected side or caps")
    return dict(native_q=normalized, raw_q=s["raw_q"], source_sha256=digest((folder/"source.json").read_bytes()),
                source_executable_sha256=digest(executable.read_bytes()), source_hash=head[6], triangulation_hash=head[7],
                stored_q_max_abs_error=float(max(abs(np.array(normalized)-s["q"]))),
                vertices=nv, velocity_scalars=30*nv, oriented_faces=nf, cap_faces=sum(caps),
                cap_aabbs_disjoint_from_exact_design=cap_box_disjoint(dict(s, face_is_cap=caps)),
                inside_positive=bool(inside), surface_type=head[4])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--bounds", required=True, type=Path)
    parser.add_argument("folders", nargs="+", type=Path)
    args = parser.parse_args()
    for case in args.folders:
        dump(case/"native-source-replay.json", replay(case, args.source, args.bounds))
