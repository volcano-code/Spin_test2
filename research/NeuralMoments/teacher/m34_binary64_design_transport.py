"""Audit IEEE binary64 input transport of an identified exact design operand."""

import argparse
from collections import Counter
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def normal(a, b, c):
    u = [y-x for x, y in zip(a, b)]
    v = [y-x for x, y in zip(a, c)]
    return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])


def audit(payload, expected_sha256, *, gmp_checker=None):
    if sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("design operand digest mismatch")
    rows = payload.decode("ascii").splitlines()
    nv, nf = map(int, rows[0].split())
    if nv < 3 or nf < 1 or len(rows) != 1 + nv + nf:
        raise ValueError("invalid operand dimensions")
    points = [tuple(map(Q, row.split())) for row in rows[1:1+nv]]
    faces = [tuple(map(int, row.split())) for row in rows[1+nv:]]
    if any(len(p) != 3 for p in points) or any(len(f) != 3 or min(f) < 0 or max(f) >= nv for f in faces):
        raise ValueError("invalid operand row")
    rounded = [tuple(Q.from_float(float(x)) for x in p) for p in points]
    drift, changed, moved = Q(0), 0, 0
    for p, q in zip(points, rounded):
        delta = [abs(x-y) for x, y in zip(p, q)]
        drift = max(drift, *delta)
        changed += sum(x != 0 for x in delta)
        moved += int(any(delta))

    # All represented denominators are powers of two; use a common integer lattice
    # to classify every projected triangle exactly, including collinear triples.
    scale = max(x.denominator for p in rounded for x in p)
    lattice = [tuple(x.numerator*(scale//x.denominator) for x in p) for p in rounded]
    lost_faces, witness = Counter(), None
    for i, face in enumerate(faces):
        vertices = [lattice[j] for j in face]
        if not any(normal(*vertices)):
            kind = "coincident_corners" if len(set(vertices)) < 3 else "distinct_collinear"
            original = [points[j] for j in face]
            if not any(normal(*original)):
                raise ValueError("exact source already contains degenerate triangle")
            lost_faces[kind] += 1
            if witness is None:
                witness = dict(face_index=i, vertex_ids=face,
                               exact_vertices=[[str(x) for x in p] for p in original],
                               rounded_vertices_hex=[[float(x).hex() for x in p] for p in original])
    different = len(set(points)) - len(set(lattice))
    result = dict(schema="W2_M34_BINARY64_DESIGN_OPERAND_AUDIT_V1", source_sha256=expected_sha256,
                  vertices=nv, faces=nf, exact_unique_points=len(set(points)), rounded_unique_points=len(set(lattice)),
                  distinct_vertices_lost=different, changed_coordinates=changed, moved_vertices=moved,
                  max_coordinate_error_model_exact=str(drift), max_coordinate_error_model_approx=float(drift),
                  formerly_nondegenerate_faces_lost=sum(lost_faces.values()), degeneracy_types=dict(lost_faces),
                  first_face_witness=witness, unchanged_coordinate_representation=(changed == 0),
                  geometry_and_jacobian_admitted=False)
    if gmp_checker:
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "operand.txt")
            with path.open("w", encoding="ascii", newline="\n") as out:
                out.write(rows[0]+"\n")
                for p in rounded:
                    out.write(" ".join(f"{x.numerator}/{x.denominator}" for x in p)+"\n")
                for row in rows[1+nv:]:
                    out.write(row+"\n")
            with path.open() as input_stream:
                probe = subprocess.run([str(gmp_checker), "--max-seconds", "180"], stdin=input_stream,
                                       capture_output=True, text=True, timeout=240)
            fields = dict(line.split("=", 1) for line in probe.stdout.splitlines() if "=" in line)
            if "status" not in fields:
                raise RuntimeError(f"checker produced no status: exit {probe.returncode}")
            if lost_faces and (probe.returncode == 0 or fields.get("reason") != "DEGENERATE_EXACT_TRIANGLE"):
                raise RuntimeError("checker disagrees with exact projected-face classification")
            result["gmp_replay"] = dict(rounded_operand_sha256=sha256(path.read_bytes()).hexdigest(),
                                        exit_code=probe.returncode, status=fields["status"], reason=fields.get("reason"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operand", type=Path)
    parser.add_argument("expected_sha256")
    parser.add_argument("--gmp-checker", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.operand.read_bytes(), args.expected_sha256,
                           gmp_checker=args.gmp_checker), indent=2, sort_keys=True))
