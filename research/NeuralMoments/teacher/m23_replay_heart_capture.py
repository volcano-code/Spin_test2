"""Offline replay of the persisted output of one freshly identified Heart native call."""
import argparse
from fractions import Fraction
import json
import sys
from pathlib import Path

from frozen_model_binding import validate_model_binding
from independent_reference import compare_predeclared_budget, integrate_exact_boundary
from m16_native_acceptance import MODULES, SOURCE_SHA, file_hash, strict_json
from m22_audit_heart_label import require
from m22_point0_operand_replay import exact_point
from native_cherchi_teacher import digest, verify

W0_SOLID_COMMIT = '219b9bc9b5efbe97b6f41060afb48c752c1bb7b6'
FROZEN_REQUEST = '7084fecb6bbd4d17bca842bfe68299f9711ecead09eb01593364424a3e0c4a8a'
FROZEN_NATIVE_C = '8be318f8e19909a2c63ba5287d712665b9723b1b7a9cca42dc4ba23c16819a47'
FROZEN_SETTINGS = '3e1dab1b74ab0c4c60bd5d54845df590e908f7b2c11f4003e89cc0284b31b8da'
FROZEN_FIXED = '8152490745a60b34bdfaaeb1f2586644ed5f97114e97b90a66fa6eb7dbfb8d16'
HISTORICAL_CUTTER = '596d2573abbfee52d77090544f34b11bae17a9a45199df00fe848aa905908850'
# Computed by surface_identity on the original H8f canonical material 6be4faf5...
HISTORICAL_ORIENTED_SURFACE = '0b0152079cbfa7bc555e93575caa1971bd5b5b17061e6fc3c08a66353477175a'
REQUIRED_FILES = {'spec.json', 'label.json', 'material.json', 'module-manifest.json',
                  'frozen_normalized_outer.obj'}


def capture_files(directory, expected_run_id):
    capture = strict_json((directory / 'capture.json').read_bytes())
    require(capture['schema'] == 'w2-m23-fresh-heart-capture-v1' and expected_run_id
            and capture['run_id'] == expected_run_id, 'WRONG_CAPTURE_OR_RUN_ID')
    require(capture['fresh_evaluation_verified'] is True and type(capture['native_cherchi_calls']) is int
            and capture['native_cherchi_calls'] == 1 and capture['training_admitted'] is False
            and capture['J_D'] is None, 'NOT_A_COMPLETED_SINGLE_CALL')
    require(REQUIRED_FILES <= set(capture['files']), 'INCOMPLETE_CAPTURE')
    for name, sha in capture['files'].items():
        path = directory / name
        require(Path(name).name == name and path.is_file() and not path.is_symlink()
                and file_hash(path) == sha, 'CAPTURE_FILE_HASH_MISMATCH:' + name)
    return capture


def surface_identity(vertices, faces):
    # Canonicalize vertex numbering, triangle order and cyclic permutations.
    # Keep triangle multiplicity and winding (a reversal changes the digest).
    coords = [tuple(map(str, point)) for point in vertices]
    triangles = []
    for face in faces:
        require(len(face) == 3 and all(type(i) is int and 0 <= i < len(coords) for i in face),
                'INVALID_PHYSICAL_FACE')
        tri = [coords[i] for i in face]
        triangles.append(min(tuple(tri[i:] + tri[:i]) for i in range(3)))
    return digest(sorted(triangles))


def check_attachment(capture, payload, material):
    proof = payload['exact_proof']
    require(proof['native_cherchi'] is True and proof['kind'] == 'cherchi_same_call_discrete_map'
            and type(proof['native_stage_call_counts']['boolean']) is int
            and proof['native_stage_call_counts']['boolean'] == 1
            and proof['native_stage_call_counts'] == capture['native_stage_call_counts'],
            'MISSING_NATIVE_PROOF')
    sha = digest(material)
    binding = proof['same_call_attachment']
    require(binding['schema'] == 'w2-same-call-material-binding-v1'
            and binding['native_c_sha256'] == FROZEN_NATIVE_C
            and sha == binding['serialized_material_sha256'] == capture['same_call_material_sha256'],
            'WRONG_SAME_CALL_ATTACHMENT')
    for key in ('evaluation_map_identity', 'boolean_backend_identity', 'derivative_identity',
                'geometry_hash', 'topology_hash'):
        require(isinstance(material[key], str) and material[key] and proof[key] == material[key],
                'ATTACHMENT_MISMATCH:' + key)
    require(proof['evaluation_map_identity'] == capture['evaluation_map_identity'], 'WRONG_EVALUATION_MAP')
    require(material['boolean_operation'] == 'subtraction'
            and digest(material['quadric_coefficients']) == FROZEN_NATIVE_C, 'WRONG_BOOLEAN_OR_QUADRIC')
    return sha


def replay(directory, expected_run_id):
    capture = capture_files(directory, expected_run_id)
    spec, label, material, build = [strict_json((directory / name).read_bytes()) for name in
                                    ('spec.json', 'label.json', 'material.json', 'module-manifest.json')]
    req = spec['request']
    require(spec['schema'] == 'w2-one-cherchi-request-v3' and spec['source_sha'] == SOURCE_SHA
            and ('normalization' not in spec and 'design_vertices_physical' not in spec),
            'WRONG_FROZEN_SPEC_OR_NORMALIZATION')
    require(digest(req) == capture['request_sha256'] == FROZEN_REQUEST
            and digest(spec['native_c']) == spec['expected_native_c_sha256'] == FROZEN_NATIVE_C
            and digest(spec['evaluator_settings']) == FROZEN_SETTINGS
            and spec['reference_absolute_tolerances'] == [1e-10]*10, 'WRONG_FROZEN_POINT_OR_BUDGET')
    require(spec['geometry']['capture_run_id'] == expected_run_id, 'SPEC_RUN_ID_MISMATCH')
    model_name = 'frozen_normalized_outer.obj'
    require(capture['files'][model_name] == spec['expected_model_sha256'], 'WRONG_SAVED_MODEL')
    binding = validate_model_binding(directory / model_name, req, spec['model_geometry_binding'])
    payload = verify(label, expected_q=spec['native_c'], expected_model_sha256=spec['expected_model_sha256'],
                     expected_native_module_sha256=spec['expected_native_module_sha256'])
    proof = payload['exact_proof']
    require(payload['split'] == 'excluded' and payload['J_D'] is None and payload['derivative_contract'] is None
            and payload['derivative_status'] == 'UNKNOWN' and payload['topology_status'] == 'OBSERVED_DISCRETE_MAP',
            'EXPECTED_EXCLUDED_DIAGNOSTIC')
    geo = dict(payload['geometry'])
    sealed = geo.pop('_w2_frozen_bindings')
    require(geo == spec['geometry'] and sealed['request_sha256'] == FROZEN_REQUEST
            and sealed['native_c_sha256'] == FROZEN_NATIVE_C and sealed['evaluator_settings_sha256'] == FROZEN_SETTINGS
            and sealed['source_sha'] == SOURCE_SHA and sealed['model_to_request_geometry_proof'] == binding
            and sealed['spec_sha256'] == digest(dict(spec, _verified_model_binding=binding)),
            'LABEL_SPEC_MISMATCH')
    material_sha = check_attachment(capture, payload, material)
    require(build['schema'] == 'w2-m16-externally-frozen-ELF-extensions-v1'
            and set(build['extensions']) == set(MODULES), 'WRONG_MODULE_MANIFEST')
    modules = {name: build['extensions'][name]['sha256'] for name in MODULES}
    require(modules == capture['module_sha256'] and spec['expected_native_module_sha256'] == modules[MODULES[0]]
            and geo['cherchi_bridge_sha256'] == modules[MODULES[1]]
            and geo['gmp_backend_sha256'] == modules[MODULES[2]]
            and material['boolean_backend_identity'] ==
            'cherchi-reference-7bd6c2697695f555f9a15d7e154e084d841316f0:' + modules[MODULES[1]]
            and material['rational_arithmetic_identity']['module_sha256'] == modules[MODULES[2]],
            'MODULE_IDENTITY_MISMATCH')

    require(payload['frame'] == spec['frame'] == dict(kind='INPUT_NORMALIZED_TARGET_V1',
            native_target_origin=req['target_origin'], native_target_axes=req['target_axes'])
            and payload['density'] == req['material_density'],
            'DENSITY_OR_FRAME_MISMATCH')
    target = material['variable_rational_vertices']
    faces = material['variable_faces']
    physical = [exact_point(row) for row in material['material_proof']['exact_vertices']]
    require(len(target) == len(physical), 'FRAME_VERTEX_COUNT_MISMATCH')
    o = tuple(map(Fraction, req['target_origin']))
    axes = [tuple(map(Fraction, column)) for column in zip(*req['target_axes'])]
    for point, row in zip(physical, target):
        shifted = tuple(x - a for x, a in zip(point, o))
        require(tuple(sum(v*c for v, c in zip(shifted, axis)) for axis in axes) == exact_point(row),
                'TARGET_FRAME_MISMATCH')
    exact, detail = integrate_exact_boundary(target, faces, payload['density'])
    compare = compare_predeclared_budget(payload['r_D'], exact, spec['reference_absolute_tolerances'])
    require(compare['within_tolerance'] and dict(detail, comparison=compare) ==
            proof['independent_ten_moment_audit']
            and detail['exact_boundary_sha256'] == capture['exact_boundary_sha256'], 'TEN_MOMENT_REPLAY_MISMATCH')
    require(digest(payload['r_F']) == FROZEN_FIXED and material['variable_raw_moments'] == payload['r_D'] and
            material['raw_moments'] == [a+b for a,b in zip(payload['r_D'],payload['r_F'])], 'MATERIAL_MOMENT_MISMATCH')

    cutter = digest(dict(vertices=material['quadric_source_vertices'], faces=material['quadric_source_faces'],
                         coefficients=material['quadric_coefficients']))
    require(material['quadric_source_identity'] == cutter, 'CUTTER_IDENTITY_MISMATCH')
    surface = surface_identity(physical, faces)
    return dict(schema='w2-m23-fresh-heart-offline-replay-v1', run_id=expected_run_id,
                capture_file_sha256=file_hash(directory/'capture.json'), request_sha256=FROZEN_REQUEST,
                material_sha256=material_sha, evaluation_map_identity=material['evaluation_map_identity'],
                module_sha256=modules, native_calls_in_offline_replay=0,
                stored_ten_exact_moments_matched=True, exact_boundary_sha256=detail['exact_boundary_sha256'],
                vertices=detail['vertex_count'], faces=detail['face_count'],
                maximum_absolute_error_vs_rounded_reference=compare['maximum_absolute_error'],
                maximum_exact_error=float(max(abs(Fraction(x)-r) for x, r in zip(payload['r_D'],exact))),
                physical_oriented_triangles_sha256=surface, faceted_cutter_sha256=cutter,
                historical_solid_commit=W0_SOLID_COMMIT,
                fresh_regularized_solid_verified_by_identity=(cutter==HISTORICAL_CUTTER and
                                                              surface==HISTORICAL_ORIENTED_SURFACE),
                training_admitted=False, J_D=None)


if __name__ == '__main__':
    # Exact integrated rational moments on this authenticated mesh exceed 4,300 digits.
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(100000)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--expected-run-id', required=True)
    args = p.parse_args()
    print(json.dumps(replay(args.capture_dir, args.expected_run_id), sort_keys=True, indent=2))
