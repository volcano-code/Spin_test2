"""Rebuild operand provenance and winding for saved native Newton prefix materials."""
import argparse
import json
from pathlib import Path

from m16_native_acceptance import REQUEST_SHA, file_hash, require, strict_json
from m22_point0_operand_replay import audit_geometry
from native_cherchi_teacher import digest


def replay(folder, expected_result_sha256):
    require(file_hash(folder/'result.json') == expected_result_sha256, 'WRONG_EXTERNAL_RESULT_DIGEST')
    receipt = strict_json((folder/'result.json').read_bytes())
    require(receipt['schema'] == 'w2-m29-first-newton-step-v1' and
            file_hash(folder/'request.json') == receipt['file_sha256']['request.json'] == REQUEST_SHA and
            file_hash(folder/'evaluations.jsonl') == receipt['file_sha256']['evaluations.jsonl'],
            'RECEIPT_FILE_DRIFT')
    req = strict_json((folder/'request.json').read_bytes())['request']
    require(digest(req)==receipt['request_sha256'], 'REQUEST_DRIFT')
    results = []
    with (folder/'evaluations.jsonl').open('rb') as stream:
        for index, line in enumerate(stream,1):
            row = strict_json(line)
            require(row['index']==index and row['status'] in ('returned','exception'), 'EVALUATION_ID_OR_STATUS')
            item = dict(index=index, status=row['status'])
            if row['status']=='returned':
                m = row['material']
                require(digest(m)==row['material_sha256'] and m['quadric_coefficients']==row['native_c']
                        and m['boolean_operation'] in ('intersection','subtraction'), 'MATERIAL_IDENTITY_OR_OPERATION')
                source=dict(coefficients=row['native_c'],faces=m['quadric_source_faces'],vertices=m['quadric_source_vertices'])
                require(digest(source)==m['quadric_source_identity'], 'CUTTER_IDENTITY_DRIFT')
                counts, face_counts, parents = audit_geometry(req,m)
                item.update(operation=m['boolean_operation'],material_sha256=row['material_sha256'],
                            constructions=counts, face_support_and_winding=face_counts,
                            ordered_parent_map_sha256=digest(parents),vertices=len(m['variable_rational_vertices']),
                            faces=len(m['variable_faces']))
            else:
                item.update(error_type=row['error_type'], error=row['error'])
            results.append(item)
    require(len(results)==receipt['evaluation_count'], 'EVALUATION_COUNT_DRIFT')
    return dict(schema='w2-m29-native-prefix-operands-v1',run_id=receipt['run_id'],
                external_receipt_sha256=expected_result_sha256,evaluations=results,
                native_calls_in_replay=0,whole_boolean_set_equality_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('group', type=Path)
    p.add_argument('result_sha256')
    args=p.parse_args()
    print(json.dumps(replay(args.group,args.result_sha256),indent=2))
