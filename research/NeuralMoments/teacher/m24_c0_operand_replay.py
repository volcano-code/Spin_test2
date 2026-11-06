"""Exact source-operand reconstruction and face-support audit of saved Heart coefficient perturbations."""
import argparse
import json
from pathlib import Path

from m16_native_acceptance import file_hash, require, strict_json
from m22_point0_operand_replay import audit_geometry
from m24_capture_c0 import plan
from m25_capture_linear import COLUMNS
from m26_capture_c7 import plan as event_plan
from m27_capture_base_column import COLUMNS as BASE_COLUMNS, plan as base_column_plan
from native_cherchi_teacher import digest


def replay(group, group_id, expected_result_sha256, *, linear=False, c7=False, base_column=None):
    require(sum((linear, c7, base_column is not None)) <= 1, "SPECIFY_ONE_STENCIL_PROTOCOL")
    protocol = "w2-m26-c7" if c7 else "w2-m25-linear" if linear else "w2-m24-c0"
    if base_column is not None:
        protocol = "w2-m27-base-column"
    count = 6 if c7 else 8 if linear else 4
    result_file = group / 'stencil-result.json'
    require(file_hash(result_file) == expected_result_sha256, 'WRONG_EXTERNAL_RESULT_DIGEST')
    result = strict_json(result_file.read_bytes())
    stencil_file = group / 'stencil.json'
    require(file_hash(stencil_file) == result['stencil_sha256'], 'STENCIL_DIGEST_MISMATCH')
    stencil = strict_json(stencil_file.read_bytes())
    require(result['schema'] == protocol+'-result-v1' and result['training_admitted'] is False
            and result['J_D'] is None and Path(group_id).name == group_id and group_id not in ('', '.', '..')
            and group_id == result['group_id'] and len(result['samples']) == count
            and result['fresh_native_boolean_calls'] == count, 'STENCIL_GROUP_ID_OR_COUNT')
    args = (group_id, stencil['module_sha256'], stencil['source_checkout'])
    if base_column is not None:
        expected_plan = base_column_plan(*args, column=base_column)
    else:
        expected_plan = event_plan(*args) if c7 else plan(*args, columns=COLUMNS if linear else None)
    require(stencil == expected_plan, 'UNEXPECTED_STENCIL')
    samples = []
    for expected, entry in zip(stencil['samples'], result['samples']):
        if linear or c7 or base_column is not None:
            require(entry['column'] == expected['column'], 'COLUMN_MISMATCH')
        if c7:
            require(all(entry[k] == expected[k] for k in ('side','location')), 'EVENT_SIDE_MISMATCH')
        run_id = expected['run_id']
        require(entry['run_id'] == run_id and entry['index'] == expected['index'], 'POINT_ID_MISMATCH')
        folder = group / run_id
        cap_file = folder / 'capture.json'
        require(file_hash(cap_file) == entry['capture_sha256'], 'CAPTURE_DIGEST_MISMATCH')
        capture = strict_json(cap_file.read_bytes())
        require(capture['schema'] == protocol+'-sample-v1' and capture['run_id'] == run_id
                and capture['source_checkout'] == stencil['source_checkout']
                and capture['module_sha256'] == stencil['module_sha256']
                and capture['native_stage_call_counts']['boolean'] == 1, 'POINT_CAPTURE_MISMATCH')
        docs = []
        for name in ('spec.json', 'material.json'):
            f = folder / name
            require(file_hash(f) == capture['files'][name], 'CAPTURE_FILE_DRIFT:' + name)
            docs.append(strict_json(f.read_bytes()))
        spec, mat = docs
        req = spec['request']
        require(digest(req) == capture['request_sha256'] == stencil['request_sha256']
                and 'normalization' not in spec and 'design_vertices_physical' not in spec
                and spec['geometry']['capture_run_id'] == run_id, 'WRONG_REQUEST')
        require(digest(spec['native_c']) == digest(mat['quadric_coefficients'])
                == capture['native_c_sha256'] == expected['native_c_sha256'], 'WRONG_COEFFICIENT')
        require(mat['boolean_operation'] == 'subtraction' and digest(mat) == entry['same_call_material_sha256']
                == capture['same_call_material_sha256'], 'WRONG_SAME_CALL_MATERIAL')
        src = dict(vertices=mat['quadric_source_vertices'], faces=mat['quadric_source_faces'],
                   coefficients=mat['quadric_coefficients'])
        require(digest(src) == mat['quadric_source_identity'], 'WRONG_CUTTER_IDENTITY')
        n = len(mat['material_proof']['exact_vertices'])
        require(n > 0 and mat['variable_faces'] and all(len(face) == 3 and all(type(i) is int and 0 <= i < n
                for i in face) for face in mat['variable_faces']), 'BAD_OUTPUT_TRIANGLE_INDICES')
        counts, faces, assignments = audit_geometry(req, mat)
        samples.append(dict(run_id=run_id, native_c_sha256=expected['native_c_sha256'],
                            material_sha256=capture['same_call_material_sha256'],
                            vertices_in_model_and_target=sum(counts.values()), construction_counts=counts,
                            faces_with_exact_support_and_winding=sum(faces.values()), face_counts=faces,
                            ordered_face_to_operand_sha256=digest(assignments)))
        samples[-1].update({k:expected[k] for k in ('column','side','location') if k in expected})
    return dict(schema=protocol+'-operands-v1', group_id=group_id, stencil_result_sha256=expected_result_sha256,
                samples=samples, fresh_native_calls_in_replay=0, whole_boolean_set_equality_verified=False,
                smooth_interval_verified=False, J_D=None, training_admitted=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('group', type=Path)
    p.add_argument('group_id')
    p.add_argument('result_sha256', help='SHA-256 of the independently retrieved public stencil-result.json')
    protocol = p.add_mutually_exclusive_group()
    protocol.add_argument('--c7', action='store_true', help='verify the M26 six event-side samples')
    protocol.add_argument('--linear', action='store_true', help='verify the registered M25 eight-point linear stencil')
    protocol.add_argument('--base-column', type=int, choices=BASE_COLUMNS, help='verify one registered M27 base column')
    args = p.parse_args()
    print(json.dumps(replay(args.group, args.group_id, args.result_sha256, linear=args.linear, c7=args.c7,
                            base_column=args.base_column), indent=2, sort_keys=True))
