"""Replay only the terminal or best accepted material witnesses from the native trajectory."""
import argparse
import json
from pathlib import Path
from m16_native_acceptance import REQUEST_SHA, file_hash, require, strict_json
from m22_point0_operand_replay import audit_geometry
from native_cherchi_teacher import digest


def replay(folder, expected_result_sha256):
    require(file_hash(folder/'result.json')==expected_result_sha256,'WRONG_EXTERNAL_RESULT_DIGEST')
    receipt = strict_json((folder/'result.json').read_bytes())
    require(receipt['schema']=='w2-m30-trajectory-v1','WRONG_SCHEMA')
    for name in ('request.json','events.jsonl','evaluations.jsonl'):
        require(file_hash(folder/name)==receipt['file_sha256'][name], 'FILE_DRIFT_'+name)
    req = strict_json((folder/'request.json').read_bytes())['request']
    require(digest(req)==receipt['request_sha256'] and file_hash(folder/'request.json')==REQUEST_SHA, 'REQUEST_DRIFT')
    with (folder/'events.jsonl').open('rb') as stream:
        events=[strict_json(line) for line in stream]
    accepted=[e for e in events if e['type']=='evaluation' and e['outcome']=='accepted']
    require(bool(accepted),'NO_ACCEPTED_EVENT')
    trials=[e for e in accepted if e['iteration']>=1 and e['reason'] in ('residual_decreased','topology_transition_relinearized')]
    require([e['evaluation_index'] for e in trials]==receipt['accepted_evaluation_indices'], 'ACCEPTED_SEQUENCE_DRIFT')
    selected={}
    for role,e in [('initial_accepted',accepted[0]), ('previous_accepted',accepted[-2] if len(accepted)>1 else accepted[-1]),
                   ('last_accepted',accepted[-1]),('best_accepted_raw_residual',min(accepted,key=lambda e:e['raw_equation_residual'])),
                   ('best_accepted_physical_residual',min(accepted,key=lambda e:e['physical_relative_residual']))]:
        selected.setdefault(e['evaluation_index'],(e,[]))[1].append(role)
    audits=[]
    with (folder/'evaluations.jsonl').open('rb') as stream:
        for index,line in enumerate(stream,1):
            if index not in selected: continue
            event,roles = selected[index]
            row = strict_json(line)
            require(row['index']==index and row['status']=='returned' and row['native_c']==event['quadric_coefficients'], 'EVALUATION_ID_OR_Q_DRIFT')
            material=row['material']
            require(digest(material)==row['material_sha256'] and material['quadric_coefficients']==row['native_c'] and
                    material['geometry_hash']==event['material_geometry_hash'] and material['topology_hash']==event['material_topology_hash'],
                    'MATERIAL_IDENTITY_OR_MAP_DRIFT')
            source=dict(coefficients=row['native_c'],faces=material['quadric_source_faces'],vertices=material['quadric_source_vertices'])
            require(digest(source)==material['quadric_source_identity'],'CUTTER_IDENTITY_DRIFT')
            counts,faces,parents=audit_geometry(req,material)
            audits.append(dict(index=index,roles=roles,restart=event['restart'],iteration=event['iteration'],operation=material['boolean_operation'],
                               material_sha256=row['material_sha256'],constructions=counts,face_support_and_winding=faces,
                               ordered_parent_map_sha256=digest(parents),vertices=len(material['variable_rational_vertices']),faces=len(material['variable_faces'])))
    require({a['index'] for a in audits}==set(selected),'MISSING_SELECTED_EVALUATION')
    return dict(schema='w2-m30-selected-trajectory-operands-v1',run_id=receipt['run_id'],external_receipt_sha256=expected_result_sha256,
                accepted_evaluation_count=len(accepted),accepted_trial_count=len(trials),selected_evaluations=audits,
                native_calls_in_replay=0,whole_boolean_set_equality_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('group',type=Path)
    p.add_argument('result_sha256')
    args=p.parse_args()
    print(json.dumps(replay(args.group,args.result_sha256),indent=2))
