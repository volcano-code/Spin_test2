"""Independent, exact XZ section of the frozen H8f point-000 design face 3456."""
import argparse
import gzip
import json
from fractions import Fraction as F

from .native_teacher_replay import frac
from .real_h8e_archive_m8 import check_closed_indexed_surface, open_pinned, read_json, read_member, require
from .real_h8f_81_m9 import ARCHIVE_SHA256


def edges(polygon):
    return zip(polygon, polygon[1:] + polygon[:1])


def turn(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def blend(a, b, t):
    return tuple(x + t * (y - x) for x, y in zip(a, b))


def overlaps(a, b):
    # Positive-area triangle intersection by an exact separating-axis test.
    for tri, other in ((a, b), (b, a)):
        sign = turn(*tri)
        require(sign != 0, "degenerate section triangle")
        if any(all(sign * turn(u, v, p) <= 0 for p in other) for u, v in edges(tri)):
            return False
    return True


def clip_segment(a, b, tri):
    lo, hi = F(0), F(1)
    sign = turn(*tri)
    for u, v in edges(tri):
        k, m = sign * turn(u, v, a), sign * turn(u, v, b)
        if k < 0 and m < 0:
            return None
        if k < 0:
            lo = max(lo, F(k, k - m))
        if m < 0:
            hi = min(hi, F(k, k - m))
    return (blend(a, b, lo), blend(a, b, hi)) if lo < hi else None


def crossings(p, sections):
    """Even/odd membership using a positive X ray in the exact plane section."""
    hits = 0
    for a, b in sections:
        if turn(a, b, p) == 0 and all(min(x, y) <= z <= max(x, y) for x, y, z in zip(a, b, p)):
            raise ValueError("corner on cutter")
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (b[0] - a[0]) * F(p[1] - a[1], b[1] - a[1])
            hits += x > p[0]
    return hits


def witness(request, material):
    require(material["boolean_operation"] == "subtraction", "incorrect operation")
    corners = [tuple(map(F, request["design_vertices"][i])) for i in request["design_faces"][3456]]
    height = corners[0][1]
    require(all(p[1] == height for p in corners), "face is not horizontal")
    parent = [(x, z) for x, _, z in corners]
    a, b, c = parent
    mesh = [tuple(map(F, p)) for p in material["quadric_source_vertices"]]
    facets = material["quadric_source_faces"]
    check_closed_indexed_surface(facets, len(mesh))
    require(all(p[1] != height for p in mesh), "section passes through a cutter vertex")
    sections, cut = [], []
    for index, face in enumerate(facets):
        ends = []
        for u, v in edges([mesh[i] for i in face]):
            if (u[1] > height) != (v[1] > height):
                x, _, z = blend(u, v, (height - u[1]) / (v[1] - u[1]))
                ends.append((x, z))
        if ends:
            sections.append(ends)
            clipped = clip_segment(*ends, parent)
            if clipped is not None:
                cut.append((index, clipped))
    require(len(cut) == 1, "expected one boundary segment through the parent")
    counts = [crossings(p, sections) for p in parent]
    require([n % 2 for n in counts] == [0, 1, 0], "unexpected corner membership")
    facet, (p, q) = cut[0]
    if turn(a, b, q) == 0:
        p, q = q, p
    require(turn(a, b, p) == turn(b, c, q) == 0 and p not in parent and q not in parent,
            "incorrect intersections with parent edges")

    # Only coplanar faces can cover positive area of this parent. Compare the
    # two oriented triangles with the exact quadrilateral remaining outside.
    vertices = [tuple(map(frac, row)) for row in material["material_proof"]["exact_vertices"]]
    actual = {}
    for i, face in enumerate(material["variable_faces"]):
        pts = [vertices[j] for j in face]
        if all(x[1] == height for x in pts):
            tri = [(x, z) for x, _, z in pts]
            if overlaps(parent, tri):
                actual[i] = tri
    def oriented(tri):
        return min(tuple(tri[n:] + tri[:n]) for n in range(3))
    require(len(actual) == 2 and {oriented(t) for t in actual.values()} ==
            {oriented([p, q, c]), oriented([c, a, p])}, "oriented saved triangulation differs")
    residual = abs(turn(*parent)) - abs(turn(p, b, q)) - sum(abs(turn(*t)) for t in actual.values())
    require(residual == 0, "nonzero projected area residual")
    return dict(archive_sha256=ARCHIVE_SHA256, point="H8f/point-000", design_face=3456,
                input_design_vertex_ids=request["design_faces"][3456],
                cutter_facets_considered=len(facets), cutter_sections=len(sections),
                active_cutter_facet=facet, corner_ray_crossings=counts,
                saved_output_face_ids=sorted(actual), projected_area_residual=str(residual / 2),
                global_boolean_set_equality=False, new_native_calls=0, training_admitted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    with open_pinned(parser.parse_args().archive, ARCHIVE_SHA256) as archive:
        request = read_json(read_member(archive, "artifacts/h8-request/prepared-request.json"))
        member = read_member(archive, "artifacts/h8f-all81-points/point-000-material.json.gz")
        print(json.dumps(witness(request["request"], read_json(gzip.decompress(member))), indent=2))


if __name__ == "__main__":
    main()
