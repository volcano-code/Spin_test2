"""Cross-check the historical Heart point-000 subtraction with exact seam geometry.

Input operands alone determine the spatial intersections. The separately replayed
W2 face map supplies identities for a second pass over the saved output.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import gzip
import json
from math import lcm
from pathlib import Path

import numpy as np

from .native_teacher_replay import frac
from .real_h8e_archive_m8 import canonical_digest, open_pinned, read_json, read_member, require
from .real_h8f_81_m9 import ARCHIVE_SHA256


FACE_MAP_SHA = 'c1de258442374ba9678da93b3f12e4cfc150d82dabcbefcdc966a1d0ea2e0270'


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def normal(tri):
    a, b, c = tri
    return cross(sub(b, a), sub(c, a))


def section(tri, signed):
    points = {p for p, d in zip(tri, signed) if not d}
    for i, j in ((0, 1), (1, 2), (2, 0)):
        u, v = signed[i], signed[j]
        if u * v < 0:
            points.add(tuple(F(u * b - v * a, u - v) for a, b in zip(tri[i], tri[j])))
    return points


def intersection(t, nt, u, nu):
    """Intersect intervals on the common line; orient the edge for design minus cutter."""
    dt = [dot(nu, sub(p, u[0])) for p in t]
    du = [dot(nt, sub(p, t[0])) for p in u]
    if min(dt) > 0 or max(dt) < 0 or min(du) > 0 or max(du) < 0:
        return None
    tangent = cross(nu, nt)
    require(any(tangent), "coplanar candidate needs independent coplanar classification")
    axis = max(range(3), key=lambda i: abs(tangent[i]))
    s, r = [sorted(section(tri, d), key=lambda p: p[axis]) for tri, d in ((t, dt), (u, du))]
    a = max((s[0], r[0]), key=lambda p: p[axis])
    b = min((s[-1], r[-1]), key=lambda p: p[axis])
    if a[axis] > b[axis]:
        return None
    require(a != b, "point contact needs singular-event classification")
    # A carrier edge/vertex coincidence needs separate sheet and link bookkeeping.
    require(all(dt) and all(du), "nontransverse contact at an input vertex")
    require(not (set(s) & set(r)), "coincident input edge endpoints on intersection line")
    return (a, b) if tangent[axis] > 0 else (b, a)


def input_seams(design, cutter, dnorm, cnorm, original):
    # The operands are the unmodified binary64 input values on a common dyadic
    # lattice. Inclusive component extrema below discard only separated boxes.
    dmin, dmax = original[0].min(axis=1), original[0].max(axis=1)
    expected = {}
    candidates = 0
    for ci, tri in enumerate(cutter):
        ct = original[1][ci]
        ids = np.flatnonzero(((dmax >= ct.min(axis=0)) & (dmin <= ct.max(axis=0))).all(axis=1))
        candidates += len(ids)
        for idx in ids:
            di = int(idx)
            pair = intersection(design[di], dnorm[di], tri, cnorm[ci])
            if pair is not None:
                expected[di, ci] = pair
    return expected, candidates


def observed_seams(faces, ownership, points):
    require(len(faces) == len(ownership), "incomplete output ownership")
    incidences = defaultdict(list)
    for face, owner in zip(faces, ownership):
        for a, b in zip(face, face[1:] + face[:1]):
            incidences[min(a,b), max(a,b)].append((owner, a, b))
    observed = {}
    for incident in incidences.values():
        require(len(incident) == 2, "open or nonmanifold output edge")
        (p, a, b), (q, v, u) = sorted(incident)
        require(a == u and b == v, "inconsistent output winding")
        if p[0] == q[0]:
            continue
        require((p[0], q[0]) == (0, 1), "unexpected operand labels")
        key = (p[1], q[1])
        require(key not in observed, "multiple seam edges for one pair")
        observed[key] = points[a], points[b]
    return observed


def cycles(segments):
    successor = dict(segments)
    require(len(successor) == len(segments) and set(successor) == set(successor.values()),
            "seam is not a collection of simple directed cycles")
    sizes = []
    while successor:
        start = p = next(iter(successor))
        size = 0
        while p in successor:
            size += 1
            p = successor.pop(p)
        require(p == start, "open seam walk")
        sizes.append(size)
    return sorted(sizes)


def orient2(a, b, p):
    return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])


def winding(p, design, normals, boxes):
    """Signed positive-X ray. Reject an ambiguous ray so another anchor is used."""
    x, y, z = p
    total = 0
    for tri, n, (xmax, ymin, ymax, zmin, zmax) in zip(design, normals, boxes):
        if xmax < x or not ymin <= y <= ymax or not zmin <= z <= zmax:
            continue
        side = dot(n, sub(p, tri[0]))
        if n[0] == 0:
            if side == 0:
                return None
            continue
        if side * n[0] > 0:
            continue  # intersection is strictly behind the anchor
        yz = [a[1:] for a in tri]
        signs = [orient2(a, b, (y,z))*n[0] for a,b in zip(yz, yz[1:]+yz[:1])]
        if min(signs) < 0:
            continue
        if not side or min(signs) == 0:
            return None
        total += 1 if n[0] > 0 else -1
    return total


def anchor_check(design, cutter, dnorm, cnorm, output, faces, owners):
    boxes = [(max(x), min(y), max(y), min(z), max(z))
             for x, y, z in (zip(*tri) for tri in design)]
    by_parent = defaultdict(list)
    for face, (owner, fi) in zip(faces, owners):
        if owner == 1:
            by_parent[fi].append([output[i] for i in face])
    results = Counter()
    for i, tri in enumerate(cutter):
        axis = max(range(3), key=lambda k: abs(cnorm[i][k]))
        def project(point):
            return point[:axis] + point[axis+1:]
        projected = [[project(p) for p in t] for t in by_parent[i]]
        for weights in ((1,1,1), (1,2,4), (4,3,1), (2,7,5), (11,3,7)):
            p = tuple(F(dot(weights, col), sum(weights)) for col in zip(*tri))
            count = winding(p, design, dnorm, boxes)
            if count is None:
                continue
            require(count in (0,1), "ambiguous design solid winding")
            covers = 0
            qp = project(p)
            for t in projected:
                signs = [orient2(a, b, qp) for a,b in zip(t, t[1:]+t[:1])]
                if min(signs) < 0 < max(signs):
                    continue
                if 0 in signs:
                    break
                covers += 1
            else:
                require(covers == count, "wrong saved coverage at a cutter anchor")
                results[count] += 1
                break
        else:
            raise ValueError("no unambiguous cutter anchor")
    return results


def replay(request, material, ownership):
    require(material["boolean_operation"] == "subtraction", "wrong Boolean operation")
    require(canonical_digest(ownership) == FACE_MAP_SHA, "wrong face map")
    coords = [request['design_vertices'], material['quadric_source_vertices']]
    fs = [request['design_faces'], material['quadric_source_faces']]
    scale = lcm(*(x.as_integer_ratio()[1] for mesh in coords for p in mesh for x in p))
    orig = [np.asarray(v, dtype=np.float64)[f] for v, f in zip(coords, fs)]
    soups = []
    for mesh, faces in zip(coords, fs):
        vs = [tuple(int(F(x) * scale) for x in p) for p in mesh]
        soups.append([tuple(vs[k] for k in f) for f in faces])
    design, cutter = soups
    dnorm, cnorm = [list(map(normal, t)) for t in soups]
    expected, candidates = input_seams(design, cutter, dnorm, cnorm, orig)
    output = [tuple(frac(s)*scale for s in v) for v in material["material_proof"]["exact_vertices"]]
    faces = material['variable_faces']
    observed = observed_seams(faces, ownership, output)
    require(expected and expected == observed, "input/output directed seam mismatch")
    loop_sizes = cycles(list(expected.values()))
    anchors = anchor_check(design, cutter, dnorm, cnorm, output, faces, ownership)
    return dict(point="H8f/point-000", archive_sha256=ARCHIVE_SHA256, face_map_sha256=FACE_MAP_SHA,
                candidate_triangle_pairs=candidates, oriented_seam_segments=len(observed),
                seam_loop_lengths=loop_sizes, cutter_anchors_inside=anchors[1],
                cutter_anchors_outside=anchors[0], new_native_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('face_map', type=Path)
    args = parser.parse_args()
    with open_pinned(args.archive, ARCHIVE_SHA256) as z:
        request = read_json(read_member(z, "artifacts/h8-request/prepared-request.json"))
        material = read_json(gzip.decompress(read_member(z, "artifacts/h8f-all81-points/point-000-material.json.gz")))
    print(json.dumps(replay(request["request"], material, read_json(args.face_map.read_bytes())), indent=2))


if __name__ == "__main__":
    main()
