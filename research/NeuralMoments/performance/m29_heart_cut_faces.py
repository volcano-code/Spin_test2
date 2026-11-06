"""Replay H8f point-000 design-parent area coverage for D minus its convex cutter.

The archive contains model-frame exact Cherchi output and the two Boolean inputs.
Use exact arithmetic after decoding. NumPy only narrows the saved-triangle search.
"""
import argparse
from collections import Counter
from fractions import Fraction as Q
import gzip
from hashlib import sha256
import json
from math import gcd, lcm
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from m28_cut_cell_coverage import audit, contains, cross, edges, sub

ARCHIVE = '99a050f4604109e0323c0110a029aa5d7601d729d755e6956ad48ed8c77ccda9'
REQUEST = '1d2b06063c5b2c7010fcb2764518c05c9707f013d6914bc4e20298f8516e061c'
PACKET = 'c372f749bf7b85015e2dcdad4a445aeab4f2f6fbd25d622089ec58d471992973'


def checked(data, digest):
    if sha256(data).hexdigest() != digest:
        raise ValueError('input SHA-256 mismatch')
    return data


def homogeneous(point):
    den = lcm(*(x.denominator for x in point))
    return tuple(x.numerator * (den // x.denominator) for x in point) + (den,)


def plane(tri):
    a, b, c = tri
    u = tuple(y-x for x, y in zip(a, b))
    v = tuple(y-x for x, y in zip(a, c))
    n = tuple(u[(i+1)%3]*v[(i+2)%3] - u[(i+2)%3]*v[(i+1)%3] for i in range(3))
    if not any(n):
        raise ValueError('degenerate input face')
    values = (*n, -sum(x*y for x, y in zip(n, a)))
    den = lcm(*(x.denominator for x in values))
    integers = [int(x * den) for x in values]
    g = gcd(*integers)
    return tuple(i // g for i in integers)


def dot(p, h):
    return p[0]*h[0] + p[1]*h[1] + p[2]*h[2] + p[3]*h[3]


def convex_planes(vertices, faces):
    # A connected closed sphere, with each oriented facet supporting the whole mesh.
    if len(set(vertices)) != len(vertices) or set(range(len(vertices))) != {i for f in faces for i in f}:
        raise ValueError('unexpected cutter vertex set')
    directed = Counter((a, b) for f in faces for a, b in edges(tuple(f)))
    if (any(directed[(b, a)] != 1 or count != 1 for (a, b), count in directed.items())
            or len(vertices) - len(directed)//2 + len(faces) != 2):
        raise ValueError('cutter must be an oriented closed sphere')
    neighbours = [[] for _ in vertices]
    for a, b in directed:
        neighbours[a].append(b)
    seen, todo = {0}, [0]
    while todo:
        for j in neighbours[todo.pop()]:
            if j not in seen:
                seen.add(j)
                todo.append(j)
    if len(seen) != len(vertices):
        raise ValueError('disconnected cutter')
    hv = list(map(homogeneous, vertices))
    planes = [plane(tuple(vertices[i] for i in face)) for face in faces]
    for p in planes:
        d = [dot(p, h) for h in hv]
        if max(d) > 0 or min(d) >= 0:
            raise ValueError('cutter is nonconvex, inward, or degenerate')
    return planes


def clip(poly, ds):
    result = []
    for (a, b), (da, db) in zip(edges(poly), edges(ds)):
        if da <= 0:
            result.append(a)
        if (da < 0 < db) or (db < 0 < da):
            t = da / (da - db)
            result.append(tuple(x + t * (y-x) for x, y in zip(a, b)))
    return result


def projected(p, axis):
    return p[:axis] + p[axis+1:]


def area2(poly):
    return sum(cross(a, b) for a, b in edges(poly))


def cut_polygon(tri, htri, planes):
    """Exact intersection of triangle and cutter, empty for zero-area contacts."""
    poly = list(tri)
    for p in planes:
        ds = [dot(p, h) for h in htri]
        if min(ds) == max(ds) == 0:
            # A plane shared with a cutter facet needs a separate coincident-surface audit.
            # Test actual overlap later; an arbitrary remote supporting plane is harmless.
            continue
        if min(ds) >= 0:
            return []
        if max(ds) <= 0:
            continue
        hp = list(map(homogeneous, poly))
        poly = clip(poly, [Q(dot(p, h), h[-1]) for h in hp])
        if len(poly) < 3:
            return []
    return poly


def projected_overlap(parent, out):
    sign = 1 if area2(parent) > 0 else -1
    poly = list(out)
    for a, b in edges(parent):
        poly = clip(poly, [-sign * cross(sub(b, a), sub(p, a)) for p in poly])
        if len(poly) < 3:
            return False
    return bool(area2(poly))


def audit_cut(tri, hp, footprint, planes, output, output_h, faces, lo, hi):
    equation = plane(tri)
    axis = max(range(3), key=lambda i: abs(equation[i]))
    parent = tuple(projected(p, axis) for p in tri)
    polygon = [projected(p, axis) for p in footprint]
    if len(polygon) < 3 or area2(polygon) == 0 or abs(area2(polygon)) == abs(area2(parent)):
        return None
    if any(all(dot(p,h)==0 for h in hp) for p in planes):
        raise ValueError('positive-area cutter facet coplanarity requires a separate surface audit')
    # Binary64 rounding is monotone. Inclusive overlap admits rounding ties.
    pts = np.array(tri, dtype='float64')
    candidates = np.flatnonzero(((hi >= pts.min(axis=0)) & (lo <= pts.max(axis=0))).all(axis=1))
    triangles, face_ids = [], []
    for oi in candidates:
        face = faces[oi]
        if any(dot(equation, output_h[k]) for k in face):
            continue
        out = tuple(projected(output[k], axis) for k in face)
        if not projected_overlap(parent, out):
            continue
        if not all(contains(parent, p) for p in out):
            raise ValueError(f'output triangle crosses parent perimeter: {oi}')
        if area2(out)*area2(parent) <= 0:
            raise ValueError(f'wrong winding on design-owned output triangle: {oi}')
        triangles.append(out)
        face_ids.append(int(oi))

    def keep(p):
        xyz = list(p)
        xyz.insert(axis, 0)
        xyz[axis] = -(sum(a*b for a,b in zip(xyz, equation)) + equation[3]) / equation[axis]
        h = homogeneous(xyz)
        vals = [dot(n, h) for n in planes]
        if max(vals) == 0:
            raise ValueError('witness on cutter boundary')
        return max(vals) > 0  # D minus closed cutter, away from its boundary

    result = audit(parent, list(edges(polygon)), triangles, keep)
    return dict(output_faces=face_ids, projection_drop_axis=axis,
                exact_projected_cutter_area=str(abs(area2(polygon))/2),
                **{k: str(v) if isinstance(v,Q) else v for k,v in result.items()})


def run(path, limit=None):
    checked(path.read_bytes(), ARCHIVE)
    with ZipFile(path) as z:
        req = json.loads(checked(z.read('artifacts/h8-request/prepared-request.json'), REQUEST))['request']
        blob = checked(z.read('artifacts/h8f-all81-points/point-000-material.json.gz'), PACKET)
    mat = json.loads(gzip.decompress(blob))
    if mat['boolean_operation'] != 'subtraction':
        raise ValueError('unexpected Boolean operation')
    def vertices(data):
        return [tuple(map(Q, p)) for p in data]
    def token(x):
        n,d = x.split('/')
        return Q(int(n,0), int(d,0))
    design = vertices(req['design_vertices'])
    dh = list(map(homogeneous, design))
    cutter = vertices(mat['quadric_source_vertices'])
    planes = convex_planes(cutter, mat['quadric_source_faces'])
    out = [tuple(map(token,p)) for p in mat['material_proof']['exact_vertices']]
    oh = list(map(homogeneous,out))
    faces = mat['variable_faces']
    coords = np.asarray(out, dtype='float64')[np.asarray(faces)]
    lo, hi = coords.min(axis=1), coords.max(axis=1)
    cutter_bounds = np.asarray(cutter,dtype='float64')
    boxes = np.asarray(design, dtype='float64')[np.asarray(req['design_faces'])]
    close = np.flatnonzero(((boxes.max(axis=1) >= cutter_bounds.min(axis=0)) &
                            (boxes.min(axis=1) <= cutter_bounds.max(axis=0))).all(axis=1))
    records = []
    for fi in close:
        idx = req['design_faces'][fi]
        tri, hp = tuple(design[i] for i in idx), tuple(dh[i] for i in idx)
        footprint = cut_polygon(tri, hp, planes)
        if len(footprint) < 3 or footprint == list(tri):
            continue
        rec = audit_cut(tri, hp, footprint, planes, out, oh, faces, lo, hi)
        if rec is not None:
            records.append(dict(design_face=int(fi), **rec))
            if len(records) == limit:
                break
    return dict(archive_sha256=ARCHIVE, prepared_request_sha256=REQUEST,
                historical_compressed_material_sha256=PACKET, evaluation='H8f/point-000',
                operation='design_minus_convex_cutter', frame='model', design_faces=len(boxes),
                cutter_faces=len(planes), saved_faces=len(faces), scan_all_design_faces=limit is None,
                positive_area_cut_parents_audited=len(records), passed=sum(r['passes'] for r in records),
                failed=sum(not r['passes'] for r in records), global_region_certified=False,
                w1_m28_archive_cut_parents_audited=0, records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    result = run(args.archive, args.limit)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='records'})
    if not result['positive_area_cut_parents_audited'] or result['failed']:
        raise SystemExit(1)
