"""Exact cutter-side positive-area coverage replay for historical H8f/point-000."""
import argparse
from collections import Counter
from fractions import Fraction as Q
import json
from pathlib import Path

import numpy as np

from m28_cut_cell_coverage import audit, cross, edges, sub
from m29_heart_cut_faces import (ARCHIVE, PACKET, REQUEST, area2, convex_planes, dot,
                                 exact_output_points, homogeneous, input_points, load_point0,
                                 plane, projected, projected_overlap, saved_triangles)


def clipped_segment(a, b, parent):
    lo, hi = Q(0), Q(1)
    sign = 1 if area2(parent) > 0 else -1
    for u, v in edges(parent):
        edge = sub(v, u)
        m, n = (sign * cross(edge, sub(p, u)) for p in (a, b))
        if m < 0 and n < 0:
            return None
        if m < 0:
            lo = max(lo, m / (m-n))
        if n < 0:
            hi = min(hi, m / (m-n))
    if lo >= hi:
        return None
    d = sub(b, a)
    return tuple(tuple(x+t*y for x,y in zip(a,d)) for t in (lo,hi))


def section_segment(tri, homogeneous_tri, equation, axis, parent):
    """All positive-length cuts of one input triangle on the cutter parent."""
    signs = [dot(equation, p) for p in homogeneous_tri]
    if min(signs) > 0 or max(signs) < 0:
        return None
    pts = [projected(p, axis) for p in tri]
    if not any(signs):
        if projected_overlap(parent, pts):
            raise ValueError('coplanar positive-area operand overlap')
        # The triangles can then touch only at the parent perimeter.
        return None
    hits = {p for p,s in zip(pts,signs) if not s}
    ds = [Q(s,h[3]) for s,h in zip(signs,homogeneous_tri)]
    for (a,b),(m,n) in zip(edges(pts),edges(ds)):
        if m*n < 0:
            t = m / (m-n)
            hits.add(tuple(x+t*(y-x) for x,y in zip(a,b)))
    if len(hits) < 2:
        return None
    a,b = sorted(hits)
    return clipped_segment(a,b,parent)


class DesignSolid:
    """Nonzero oriented winding of the frozen design surface by an exact axis ray."""
    def __init__(self, vertices, faces):
        directed = Counter(e for face in faces for e in edges(tuple(face)))
        if not faces or any(v != 1 or directed[b,a] != 1 for (a,b),v in directed.items()):
            raise ValueError('design must have a closed consistently oriented triangle boundary')
        hv = list(map(homogeneous, vertices))
        self.triangles = [tuple(vertices[i] for i in face) for face in faces]
        self.h_triangles = [tuple(hv[i] for i in face) for face in faces]
        self.planes = [plane(t) for t in self.triangles]
        coords = np.asarray(vertices,dtype='float64')[np.asarray(faces)]
        self.lo = coords.min(axis=1)
        self.hi = coords.max(axis=1)

    def section(self, tri, equation, axis, parent):
        box = np.asarray(tri,dtype='float64')
        candidates = np.flatnonzero(((self.hi >= box.min(axis=0)) &
                                     (self.lo <= box.max(axis=0))).all(axis=1))
        cuts = set()
        for k in candidates:
            seg = section_segment(self.triangles[k],self.h_triangles[k],equation,axis,parent)
            if seg:
                cuts.add(tuple(sorted(seg)))
        return cuts

    def inside(self, point):
        hp = homogeneous(point)
        x = tuple(map(float,point))
        for axis in (2,0,1):
            u,v = (i for i in range(3) if i != axis)
            # Inclusive monotone binary64 bounds only cull provably disjoint rays.
            candidates = np.flatnonzero((self.lo[:,u]<=x[u]) & (self.hi[:,u]>=x[u]) &
                                        (self.lo[:,v]<=x[v]) & (self.hi[:,v]>=x[v]) &
                                        (self.hi[:,axis]>=x[axis]))
            q = projected(point,axis)
            winding = 0
            for k in candidates:
                eq = self.planes[k]
                side = dot(eq,hp)
                if not eq[axis]:
                    if not side:
                        break  # coplanar ray, try another axis
                    continue
                if side * eq[axis] > 0:
                    continue  # plane crossing is behind this ray
                pts = [projected(p,axis) for p in self.triangles[k]]
                s = [cross(sub(b,a),sub(q,a)) for a,b in edges(pts)]
                if min(s)<0<max(s):
                    continue
                if not side:
                    raise ValueError('cell witness on design boundary')
                if 0 in s:
                    break  # ray through an edge or vertex
                winding += 1 if eq[axis]>0 else -1
            else:
                if winding not in (0,1):
                    raise ValueError(f'unexpected design winding {winding}')
                return bool(winding)
        raise ValueError('no transverse axis ray for cell witness')


def audit_parent(solid, tri, equation, output, oh, faces, lo, hi):
    axis = max(range(3),key=lambda i: abs(equation[i]))
    # A subtraction uses the reversed cutter face winding in its output.
    parent = tuple(projected(p,axis) for p in tri)[::-1]
    cuts = solid.section(tri,equation,axis,parent)
    triangles,ids = saved_triangles(tri,parent,equation,axis,output,oh,faces,lo,hi)
    states = set()

    def keep(p):
        xyz = list(p)
        xyz.insert(axis,0)
        xyz[axis] = -(sum(a*b for a,b in zip(xyz,equation)) + equation[3]) / equation[axis]
        result = solid.inside(tuple(xyz))
        states.add(result)
        return result

    report = audit(parent,cuts,triangles,keep)
    status = 'partial' if len(states)==2 else 'retained' if True in states else 'removed'
    return dict(status=status, cuts=len(cuts), cells=report['cells'], output_faces=ids,
                projection_drop_axis=axis, missing=str(report['missing']), extra=str(report['extra']),
                overlap=str(report['overlap']), passes=report['passes'])


def run(archive):
    request,material = load_point0(archive)
    design = DesignSolid(input_points(request['design_vertices']),request['design_faces'])
    cutter = input_points(material['quadric_source_vertices'])
    cf = material['quadric_source_faces']
    planes = convex_planes(cutter,cf)
    out = exact_output_points(material)
    oh = list(map(homogeneous,out))
    faces = material['variable_faces']
    boxes = np.asarray(out,dtype='float64')[np.asarray(faces)]
    lo,hi = boxes.min(axis=1),boxes.max(axis=1)
    rows,owners = [],set()
    for k,(face,eq) in enumerate(zip(cf,planes)):
        rec = audit_parent(design,tuple(cutter[i] for i in face),eq,out,oh,faces,lo,hi)
        if owners.intersection(rec['output_faces']):
            raise ValueError(f'nonunique cutter-side output at parent {k}')
        owners.update(rec['output_faces'])
        rows.append(dict(parent=k,**rec))
    return dict(archive_sha256=ARCHIVE, prepared_request_sha256=REQUEST, material_gzip_sha256=PACKET,
                point='H8f/point-000', operation='design_minus_cutter',
                scope='cutter_face_relative_interiors', cutter_parents=len(rows),
                design_input_faces=len(design.triangles), saved_faces=len(faces),
                owned_output_faces=len(owners), status_counts=dict(Counter(r['status'] for r in rows)),
                cells=sum(r['cells'] for r in rows), failed=sum(not r['passes'] for r in rows),
                new_native_calls=0, global_region_certified=False, records=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report=run(args.archive)
    with args.out.open('x') as f:
        json.dump(report,f,indent=2)
        f.write('\n')
    print({k:v for k,v in report.items() if k!='records'})
    raise SystemExit(bool(report['failed']))
