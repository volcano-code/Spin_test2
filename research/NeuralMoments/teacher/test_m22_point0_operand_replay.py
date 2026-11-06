import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from m22_point0_operand_replay import audit_vertices, segment_triangle_point
from native_cherchi_teacher import digest
from run_native_smoke import _write_bound_material, run


class Point0OperandReplayTests(unittest.TestCase):
    def test_intersection_must_be_on_both_primitives(self):
        triangle = ((0, 0, 0), (4, 0, 0), (0, 4, 0))
        self.assertEqual(segment_triangle_point(((1, 1, -2), (1, 1, 3), *triangle)), (1, 1, 0))
        for segment, reason in (
                (((3, 3, -1), (3, 3, 1)), 'OUTSIDE_SOURCE_TRIANGLE'),
                (((1, 1, 2), (1, 1, 3)), 'OUTSIDE_SOURCE_SEGMENT'),
                (((0, 0, 0), (1, 1, 0)), 'NO_UNIQUE_LINE_PLANE')):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                segment_triangle_point((*segment, *triangle))

    def test_reject_wrong_source_index_and_target_frame(self):
        request = dict(design_vertices=[[1, 2, 3]], design_faces=[], target_origin=[0, 0, 1],
                       target_axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        point = ['1/1', '2/1', '3/1']
        ref = dict(input_object=0, input_vertex=0)
        raw = dict(vertex_index=0, point_type='EXPLICIT3D', operand_coordinates_exact=[point])
        linked = dict(vertex_index=0, point_type='EXPLICIT3D',
                      operands=[dict(status='UNIQUE', source_vertices=[ref])])
        proof = dict(construction_records=[raw], resolved_construction_records=[linked], exact_vertices=[point])
        material = dict(quadric_source_vertices=[], quadric_source_faces=[], material_proof=proof,
                        variable_rational_vertices=[['1/1', '2/1', '2/1']])
        self.assertEqual(audit_vertices(request, material), {'EXPLICIT3D_design': 1})
        ref['input_vertex'] = -1
        with self.assertRaisesRegex(ValueError, 'INVALID_OPERAND_INDEX'):
            audit_vertices(request, material)
        ref['input_vertex'] = 0
        material['variable_rational_vertices'] = [point]
        with self.assertRaisesRegex(ValueError, 'TARGET_VERTEX_MISMATCH'):
            audit_vertices(request, material)

    def test_full_attachment_persisted_only_against_original_digest(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'material.json'
            material = {'evaluation_map_identity':'map', 'exact_vertices':[['1/2']]}
            proof = dict(evaluation_map_identity='map',
                         same_call_attachment=dict(serialized_material_sha256=digest(material)))
            with self.assertRaisesRegex(ValueError, 'CAPTURED_MATERIAL_BINDING'):
                _write_bound_material(path, dict(material, exact_vertices=[['1/3']]), proof)
            self.assertFalse(path.exists())
            _write_bound_material(path, material, proof)
            self.assertEqual(digest(json.loads(path.read_text())), digest(material))
            with self.assertRaises(FileExistsError):
                _write_bound_material(path, material, proof)
            with self.assertRaises(FileExistsError):
                run(Path(tmp)/'spec.json', Path(tmp)/'label.json', material_output=path)

    def test_new_label_and_material_must_have_distinct_paths(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'output.json'
            with self.assertRaisesRegex(FileExistsError, 'DISTINCT_NEW_PATH'):
                run(Path(tmp)/'spec.json', output, material_output=output)
            self.assertFalse(output.exists())
