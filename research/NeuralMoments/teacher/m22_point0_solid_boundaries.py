"""Verify the three H8f point-000 surfaces in the same physical coordinate frame."""
import argparse
import json
from fractions import Fraction
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from independent_reference import _triple
from m16_native_acceptance import HEART_ARTIFACT_SHA, REQUEST_SHA, file_hash, h256, strict_json
from m22_audit_heart_label import require
from m22_point0_operand_replay import exact_point
from m22_recover_h8f_point0 import TEACHER, read_material
from native_cherchi_teacher import digest

CHECKER_SHA = '2536f7783ff6552ac930c294879e69b8430fc8e527873859b3ec1187d39c6ea0'
PARENT = 'e702344d4f9037fd8f8f55bbdc785a1f494ddaeb'


def oriented_shell(vertices, faces):
    """Count components and calculate volume; embedding and manifold checks are separate."""
    neighbours = [set() for _ in vertices]
    volume6 = Fraction()
    for a, b, c in faces:
        volume6 += _triple(vertices[a], vertices[b], vertices[c])
        neighbours[a].update((b, c))
        neighbours[b].update((c, a))
        neighbours[c].update((a, b))
    seen, todo = set(), [0]
    while todo:
        i = todo.pop()
        if i not in seen:
            seen.add(i)
            todo.extend(neighbours[i] - seen)
    require(len(seen) == len(vertices), 'MULTIPLE_OR_UNREFERENCED_COMPONENTS')
    require(volume6 > 0, 'NONPOSITIVE_SIGNED_VOLUME')
    edges = sum(map(len, neighbours)) // 2
    euler = len(vertices) - edges + len(faces)
    require(euler <= 2 and euler % 2 == 0, 'INVALID_ORIENTABLE_SHELL_EULER')
    return dict(components=1, vertices=len(vertices), edges=edges, faces=len(faces),
                euler=euler, genus=(2-euler)//2, volume_exact=str(volume6/6),
                volume_approx=float(volume6/6))


def audit_surface(binary, vertices, faces):
    rows = [f'{len(vertices)} {len(faces)}']
    rows.extend(' '.join(map(str, p)) for p in vertices)
    rows.extend(' '.join(map(str, f)) for f in faces)
    stream = ('\n'.join(rows)+'\n').encode('ascii')
    run = subprocess.run([str(binary), '--max-seconds', '120'], input=stream,
                         capture_output=True, timeout=150)
    fields = dict(line.split('=', 1) for line in run.stdout.decode().splitlines())
    require(run.returncode == 0 and fields.get('status') == 'EXACT_POST_BOOLEAN_SELF_INTERSECTION_FREE',
            'SURFACE_CHECK_FAILED:'+str(fields))
    counts = {k:int(fields[k]) for k in ('candidate_pairs', 'exact_pairs_checked',
                                         'adjacent_pairs', 'nonadjacent_pairs')}
    require(counts['candidate_pairs'] == counts['exact_pairs_checked'] ==
            counts['adjacent_pairs'] + counts['nonadjacent_pairs'] and
            (int(fields['vertices']), int(fields['faces'])) == (len(vertices),len(faces)),
            'INCOMPLETE_SURFACE_CHECK')
    return dict(checker_input_sha256=h256(stream), surface_status=fields['status'],
                **counts, **oriented_shell(vertices, faces))


def replay(archive, checker_source):
    previous = strict_json((TEACHER / 'M22_P3_POINT000_OPERAND_REPLAY.json').read_bytes())
    material = read_material(archive, previous['historical_point_gzip_sha256'], HEART_ARTIFACT_SHA)
    with ZipFile(archive) as z:
        req = z.read('artifacts/h8-request/prepared-request.json')
    require(h256(req) == REQUEST_SHA and file_hash(archive) == HEART_ARTIFACT_SHA, 'REQUEST_OR_ARCHIVE_DRIFT')
    req = strict_json(req)['request']
    require(digest(material) == previous['historical_material_sha256'] and
            digest(req) == previous['request_sha256'] and material['boolean_operation'] == 'subtraction',
            'EVIDENCE_IDENTITY_OR_OPERATION_MISMATCH')
    require(REQUEST_SHA == previous['prepared_request_file_sha256'], 'PREPARED_REQUEST_DRIFT')

    with TemporaryDirectory() as folder:
        # Compile exactly the verified bytes, independent of an existing executable.
        source = checker_source.read_bytes()
        require(h256(source) == CHECKER_SHA, 'CHECKER_SOURCE_SHA_MISMATCH')
        path = Path(folder) / 'checker.cpp'
        path.write_bytes(source)
        binary = Path(folder) / 'checker'
        subprocess.run(['c++','-std=c++17','-O2',str(path),'-o',str(binary),'-lgmpxx','-lgmp'], check=True)
        meshes = (
            ('design', [tuple(map(Fraction,v)) for v in req['design_vertices']], req['design_faces']),
            ('cutter', [tuple(map(Fraction,v)) for v in material['quadric_source_vertices']],
             material['quadric_source_faces']),
            ('saved_difference', [exact_point(v) for v in material['material_proof']['exact_vertices']],
             material['variable_faces']))
        surfaces = {name:audit_surface(binary, vertices, faces) for name, vertices, faces in meshes}
        binary_hash = file_hash(binary)
    # Compare the physical-frame signed volume to the saved exact per-component volume.
    orientation = material['material_proof']['orientation']
    require(orientation['component_count'] == 1 and len(orientation['components']) == 1,
            'ARCHIVED_COMPONENT_COUNT_MISMATCH')
    volume = Fraction(surfaces['saved_difference']['volume_exact'])
    require(volume == Fraction(orientation['components'][0]['exact_emitted_signed_volume']),
            'PHYSICAL_VOLUME_REPLAY_MISMATCH')
    return dict(schema='w2-h8f-point000-physical-solid-boundaries-v1', archive_sha256=HEART_ARTIFACT_SHA,
                material_sha256=digest(material), request_sha256=digest(req),
                operation='design_minus_quadric_source', frame='physical_model',
                w2_operand_replay_commit=PARENT, checker_source_sha256=CHECKER_SHA,
                checker_binary_sha256=binary_hash, surfaces=surfaces,
                fresh_native_boolean_calls=0, whole_boolean_region_certified_by_this_check=False,
                J_D=None, training_admitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h8f', type=Path, required=True)
    parser.add_argument('--checker-source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.h8f, args.checker_source), indent=2, sort_keys=True))
