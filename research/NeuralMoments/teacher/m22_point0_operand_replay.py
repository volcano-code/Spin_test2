"""Reconstruct H8f point-000 vertices from input operands, independently of saved coordinates."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
from zipfile import ZipFile

from m16_native_acceptance import HEART_ARTIFACT_SHA, REQUEST_SHA, file_hash, h256, strict_json
from m22_audit_heart_label import require
from m22_recover_h8f_point0 import TEACHER, read_material
from native_cherchi_teacher import digest


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def exact_point(row):
    require(len(row) == 3, 'POINT_DIMENSION_MISMATCH')
    point = []
    for token in row:
        numerator, denominator = token.split('/')
        point.append(Fraction(int(numerator, 0), int(denominator, 0)))
    return tuple(point)


def segment_triangle_point(points):
    """Intersect a closed segment and a closed triangle using rational arithmetic."""
    a, b, c, d, e = points
    direction = sub(b, a)
    normal = cross(sub(d, c), sub(e, c))
    denominator = dot(normal, direction)
    require(denominator != 0, 'NO_UNIQUE_LINE_PLANE_INTERSECTION')
    t = Fraction(dot(normal, sub(c, a)), denominator)
    require(0 <= t <= 1, 'INTERSECTION_OUTSIDE_SOURCE_SEGMENT')
    point = tuple(x+t*y for x, y in zip(a, direction))
    for u, v in ((c, d), (d, e), (e, c)):
        require(dot(normal, cross(sub(v, u), sub(point, u))) >= 0,
                'INTERSECTION_OUTSIDE_SOURCE_TRIANGLE')
    return point


def audit_vertices(request, material):
    meshes = ((request['design_vertices'], request['design_faces']),
              (material['quadric_source_vertices'], material['quadric_source_faces']))
    positions, triangles, edges = [], [], []
    for vertices, faces in meshes:
        positions.append([tuple(map(Fraction, point)) for point in vertices])
        triangles.append({tuple(sorted(face)) for face in faces})
        edges.append({tuple(sorted((u, v))) for a, b, c in faces
                      for u, v in ((a, b), (b, c), (c, a))})
    proof = material['material_proof']
    records, resolved = proof['construction_records'], proof['resolved_construction_records']
    physical_points = proof['exact_vertices']
    target_points = material['variable_rational_vertices']
    require(len(records) == len(resolved) == len(physical_points) == len(target_points),
            'RECORD_COUNT_MISMATCH')
    origin = tuple(map(Fraction, request['target_origin']))
    axes = [tuple(map(Fraction, axis)) for axis in zip(*request['target_axes'])]
    counts = Counter()
    for i, (raw, linked) in enumerate(zip(records, resolved)):
        require(raw['vertex_index'] == linked['vertex_index'] == i, 'VERTEX_INDEX_MISMATCH')
        kind = raw['point_type']
        require(kind == linked['point_type'] and kind in ('EXPLICIT3D', 'LPI'),
                'UNSUPPORTED_CONSTRUCTION')
        exact = [exact_point(v) for v in raw['operand_coordinates_exact']]
        require(len(exact) == len(linked['operands']) == (1 if kind == 'EXPLICIT3D' else 5),
                'OPERAND_COUNT_MISMATCH')
        refs = []
        for coords, operand in zip(exact, linked['operands']):
            require(operand['status'] == 'UNIQUE' and len(operand['source_vertices']) == 1,
                    'AMBIGUOUS_OR_MISSING_OPERAND')
            ref = operand['source_vertices'][0]
            obj, vertex = ref['input_object'], ref['input_vertex']
            require(type(obj) is int and obj in (0, 1) and type(vertex) is int
                    and 0 <= vertex < len(positions[obj]), 'INVALID_OPERAND_INDEX')
            require(positions[obj][vertex] == coords, 'SOURCE_OPERAND_POSITION_MISMATCH')
            refs.append((obj, vertex))
        obj = refs[0][0]
        if kind == 'LPI':
            require([key for key, _ in refs] == [obj, obj] + [1-obj]*3,
                    'CROSS_OPERAND_CONSTRUCTION_REQUIRED')
            require(tuple(sorted(v for _, v in refs[:2])) in edges[obj], 'NOT_A_SOURCE_EDGE')
            require(tuple(sorted(v for _, v in refs[2:])) in triangles[1-obj], 'NOT_A_SOURCE_FACE')
            point = segment_triangle_point(exact)
        else:
            point = exact[0]
        require(point == exact_point(physical_points[i]), 'PHYSICAL_VERTEX_MISMATCH')
        target = tuple(dot(sub(point, origin), axis) for axis in axes)
        require(target == exact_point(target_points[i]), 'TARGET_VERTEX_MISMATCH')
        counts[f'{kind}_' + ('design' if obj == 0 else 'quadric')] += 1
    return dict(sorted(counts.items()))


def replay(pilot_dir, archive):
    entries = strict_json((TEACHER/'M22_M21_HEART_ARCHIVE_AUDIT_20261106.json').read_bytes())['files']
    docs = {}
    for name in ('spec', 'label'):
        path = pilot_dir / entries[name]['path']
        require(file_hash(path) == entries[name]['sha256'], 'M21_DIGEST_MISMATCH:'+name)
        docs[name] = strict_json(path.read_bytes())
    require(file_hash(archive) == HEART_ARTIFACT_SHA, 'H8F_ARCHIVE_SHA_MISMATCH')
    with ZipFile(archive) as z:
        raw_request = z.read('artifacts/h8-request/prepared-request.json')
        require(h256(raw_request) == REQUEST_SHA, 'PREPARED_REQUEST_BYTES_MISMATCH')
        archived_request = strict_json(raw_request)['request']
    spec, label = docs['spec'], docs['label']['payload']
    require(digest(spec['request']) == digest(archived_request),
            'HISTORICAL_M21_REQUEST_MISMATCH')
    # Only the default identity normalization for this pinned Heart request is supported.
    require('normalization' not in spec and 'design_vertices_physical' not in spec,
            'UNSUPPORTED_PHYSICAL_NORMALIZATION')
    baseline = strict_json((TEACHER/'M22_P2_HEART_BOUNDARY_REPLAY_20261106.json').read_bytes())
    material = read_material(archive, baseline['historical_source']['member_sha256'], HEART_ARTIFACT_SHA)
    require(digest(material) == baseline['recomputed_boundary']['historical_material_sha256'], 'HISTORICAL_MATERIAL_MISMATCH')
    require(material['boolean_operation'] == 'subtraction', 'UNEXPECTED_BOOLEAN_OPERATION')
    require(digest(spec['native_c']) == digest(material['quadric_coefficients']),
            'NATIVE_C_MISMATCH')
    source = dict(vertices=material['quadric_source_vertices'], faces=material['quadric_source_faces'],
                  coefficients=material['quadric_coefficients'])
    require(digest(source) == material['quadric_source_identity'], 'QUADRIC_SOURCE_IDENTITY_MISMATCH')
    proof = label['exact_proof']
    require(all(proof[k] == material[k] for k in ('geometry_hash', 'topology_hash')),
            'HISTORICAL_M21_OUTPUT_IDENTITY_MISMATCH')
    counts = audit_vertices(archived_request, material)
    return dict(schema='w2-m22-p3-heart-point000-operand-replay-v1',
                status='HISTORICAL_CONSTRUCTION_INCIDENCE_VERIFIED', request_sha256=digest(archived_request),
                prepared_request_file_sha256=REQUEST_SHA,
                archive_sha256=HEART_ARTIFACT_SHA, m21_label_sha256=entries['label']['sha256'],
                historical_point_gzip_sha256=baseline['historical_source']['member_sha256'],
                historical_material_sha256=digest(material),
                m21_material_sha256=proof['same_call_attachment']['serialized_material_sha256'],
                historical_evaluation_map_identity=material['evaluation_map_identity'],
                m21_evaluation_map_identity=proof['evaluation_map_identity'],
                design_operand_sha256=digest(dict(vertices=archived_request['design_vertices'],
                                                   faces=archived_request['design_faces'])),
                quadric_source_identity=digest(source), counts=counts, vertices_verified=sum(counts.values()),
                real_native_calls_this_replay=0, fresh_same_call_material_recovered=False,
                boolean_set_equality_proved=False, J_D=None, split='excluded', training_admitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', type=Path, required=True)
    parser.add_argument('--h8f', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.pilot_dir, args.h8f), sort_keys=True, indent=2))
