"""Replay H8f point-000 design-parent area coverage for D minus its convex cutter.

The archive contains model-frame exact Cherchi output and the two Boolean inputs.
Use exact arithmetic after decoding. NumPy only narrows the saved-triangle search.
"""
import argparse
from collections import Counter, defaultdict
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


def saved_triangles(tri, parent, equation, axis, output, output_h, faces, lo, hi, indices=None):
    # Binary64 rounding is monotone. Inclusive overlap admits rounding ties.
    pts = np.array(tri, dtype='float64')
    lower = lo if indices is None else lo[indices]
    upper = hi if indices is None else hi[indices]
    ix = np.flatnonzero(((upper >= pts.min(axis=0)) & (lower <= pts.max(axis=0))).all(axis=1))
    triangles, face_ids = [], []
    for oi in ix if indices is None else indices[ix]:
        face = faces[oi]
        if any(dot(equation, output_h[k]) for k in face):
            continue
        out = tuple(projected(output[k], axis) for k in face)
        if not all(contains(parent, p) for p in out):
            if projected_overlap(parent, out):
                raise ValueError(f'output triangle crosses parent perimeter: {oi}')
            continue
        if area2(out)*area2(parent) <= 0:
            raise ValueError(f'wrong winding on design-owned output triangle: {oi}')
        triangles.append(out)
        face_ids.append(int(oi))
    return triangles, face_ids


def audit_cut(tri, hp, footprint, planes, output, output_h, faces, lo, hi, indices=None):
    equation = plane(tri)
    axis = max(range(3), key=lambda i: abs(equation[i]))
    parent = tuple(projected(p, axis) for p in tri)
    polygon = [projected(p, axis) for p in footprint]
    if len(polygon) < 3 or area2(polygon) == 0 or abs(area2(polygon)) == abs(area2(parent)):
        return None
    if any(all(dot(p,h)==0 for h in hp) for p in planes):
        raise ValueError('positive-area cutter facet coplanarity requires a separate surface audit')
    triangles, face_ids = saved_triangles(tri,parent,equation,axis,output,output_h,faces,lo,hi,indices)

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


def canonical(p):
    return p if next(x for x in p if x) > 0 else tuple(-x for x in p)


def verify_whole(parent, triangles, keep):
    if not keep:
        return not triangles
    if len(triangles) == 1 and set(parent) == set(triangles[0]):
        return True
    return audit(parent, [], triangles, lambda _: True)['passes']


def run(path, limit=None, ledger_path=None):
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
    all_design = ledger_path is not None
    seen_outputs = set()
    counts = Counter()
    all_rows = []
    if all_design:
        close = set(close)
        by_plane = defaultdict(list)
        for oi, face in enumerate(faces):
            by_plane[canonical(plane(tuple(out[k] for k in face)))].append(oi)
        by_plane = {p: np.array(ids, dtype=int) for p, ids in by_plane.items()}
        cutter_plane_keys = set(map(canonical, planes))
        empty = np.array([], dtype=int)
    for fi in range(len(boxes)) if all_design else close:
        idx = req['design_faces'][fi]
        tri, hp = tuple(design[i] for i in idx), tuple(dh[i] for i in idx)
        footprint = cut_polygon(tri, hp, planes) if not all_design or fi in close else []
        if all_design:
            equation = plane(tri)
            axis = max(range(3), key=lambda i: abs(equation[i]))
            parent = tuple(projected(p,axis) for p in tri)
            polygon = [projected(p,axis) for p in footprint]
            cut_area = abs(area2(polygon))
            key = canonical(equation)
            if cut_area and key in cutter_plane_keys:
                raise ValueError(f'positive-area overlap with a cutter facet on parent {fi}')
            indices = by_plane.get(key, empty)
            if cut_area == 0 or cut_area == abs(area2(parent)):
                triangles, ids = saved_triangles(tri,parent,equation,axis,out,oh,faces,lo,hi,indices)
                keep = cut_area == 0
                if not verify_whole(parent,triangles,keep):
                    raise ValueError(f'whole parent coverage mismatch on {fi}')
                status = 'retained' if keep else 'removed'
            else:
                rec = audit_cut(tri, hp, footprint, planes, out, oh, faces, lo, hi, indices)
                if not rec or not rec['passes']:
                    raise ValueError(f'cut parent coverage mismatch on {fi}')
                records.append(dict(design_face=int(fi), **rec))
                ids, status = rec['output_faces'], 'cut'
            if seen_outputs.intersection(ids):
                raise ValueError(f'nonunique design-side output on parent {fi}')
            seen_outputs.update(ids)
            counts[status] += 1
            all_rows.append([int(fi),status,ids])
        else:
            if len(footprint) < 3 or footprint == list(tri):
                continue
            rec = audit_cut(tri, hp, footprint, planes, out, oh, faces, lo, hi)
            if rec is not None:
                records.append(dict(design_face=int(fi), **rec))
                if len(records) == limit:
                    break
    if all_design:
        data = ''.join(json.dumps(row,separators=(',',':'))+'\n' for row in all_rows).encode()
        ledger_path.write_bytes(data)
        return dict(archive_sha256=ARCHIVE, prepared_request_sha256=REQUEST,
                    historical_compressed_material_sha256=PACKET, evaluation='H8f/point-000',
                    expected_region='design_minus_convex_cutter', scope='design_face_relative_interiors',
                    design_faces=len(boxes), full_retained=counts['retained'], full_removed=counts['removed'],
                    partially_cut=counts['cut'], assigned_output_faces=len(seen_outputs),
                    cells_on_partially_cut_faces=sum(x['cells'] for x in records),
                    failed=0, design_parent_ledger_sha256=sha256(data).hexdigest(),
                    global_region_certified=False)
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
    parser.add_argument('--all-design-ledger',type=Path,help='audit every design face and write ordered JSONL')
    args = parser.parse_args()
    if args.limit and args.all_design_ledger:
        parser.error('choose a partial cut scan or an all-design ledger')
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    result = run(args.archive, args.limit, args.all_design_ledger)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='records'})
    if not result.get('positive_area_cut_parents_audited',result.get('partially_cut',0)) or result['failed']:
        raise SystemExit(1)
