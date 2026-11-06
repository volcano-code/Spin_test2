"""Rebuild the pinned GMP checker and test both saved meshes of an accepted M29 step."""
import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from m16_native_acceptance import file_hash, require, strict_json
from m22_point0_operand_replay import exact_point
from m22_point0_solid_boundaries import CHECKER_SHA, audit_surface
from native_cherchi_teacher import digest


def replay(directory, public_result_sha256, checker):
    require(file_hash(directory/'result.json') == public_result_sha256, 'PUBLIC_RECEIPT_MISMATCH')
    receipt = strict_json((directory/'result.json').read_bytes())
    require(receipt['status'] == 'FIRST_TRIAL_ACCEPTED', 'NO_ACCEPTED_STEP_TO_AUDIT')
    for name in ('evaluations.jsonl','events.jsonl','request.json'):
        require(file_hash(directory/name) == receipt['file_sha256'][name], 'EVIDENCE_FILE_CHANGED')
    events = [strict_json(line) for line in (directory/'events.jsonl').read_bytes().splitlines()]
    target = receipt['accepted_evaluation_index']
    accepted = [e for e in events if e['type']=='evaluation' and e['outcome']=='accepted']
    chosen = [e for e in accepted if e['evaluation_index']==target and e['iteration']==1
              and e['reason'] in ('topology_transition_relinearized','residual_decreased')]
    require(len(chosen)==1, 'NATIVE_TRIAL_EVENT_MISMATCH')
    seed = [e for e in accepted if e['restart']==chosen[0]['restart'] and e['iteration']==0 and
            e['evaluation_index']<target and e['reason']=='initial_evaluation_succeeded']
    require(len(seed)==1, 'NATIVE_INITIAL_EVENT_MISMATCH')
    expected = {seed[0]['evaluation_index']:seed[0], target:chosen[0]}
    source = checker.read_bytes()
    require(sha256(source).hexdigest()==CHECKER_SHA, 'WRONG_CHECKER_SOURCE')
    out = []
    with TemporaryDirectory() as tmp:
        cpp=Path(tmp)/'checker.cpp'; cpp.write_bytes(source)
        binary=Path(tmp)/'checker'
        subprocess.run(['c++','-std=c++17','-O2',str(cpp),'-o',str(binary),'-lgmpxx','-lgmp'],check=True)
        with (directory/'evaluations.jsonl').open('rb') as f:
            for line in f:
                row=strict_json(line)
                if row['index'] not in expected:
                    continue
                e=expected[row['index']]; mat=row['material']
                require(row['status']=='returned' and row['native_c']==e['quadric_coefficients']==mat['quadric_coefficients']
                        and row['material_sha256']==digest(mat) and
                        row['map']['geometry_hash']==mat['geometry_hash']==e['material_geometry_hash'],
                        'WRONG_EVALUATION_AT_ENDPOINT')
                points=[exact_point(p) for p in mat['material_proof']['exact_vertices']]
                details=audit_surface(binary,points,mat['variable_faces'])
                exact=details.pop('volume_exact')
                require(Fraction(exact)==Fraction(mat['material_proof']['orientation']['components'][0]['exact_emitted_signed_volume']),
                        'VOLUME_DIFFERS_FROM_ORIGINAL_MATERIAL')
                out.append(dict(evaluation_index=row['index'],material_sha256=row['material_sha256'],
                                boolean_operation=mat['boolean_operation'],volume_exact_sha256=sha256(exact.encode()).hexdigest(),
                                **details))
    require([e['evaluation_index'] for e in out]==[seed[0]['evaluation_index'],target],'MISSING_ENDPOINT_SURFACE')
    return dict(schema='w2-m29-endpoint-surface-replay-v1', run_id=receipt['run_id'],
                result_sha256=public_result_sha256,checker_source_sha256=CHECKER_SHA,endpoints=out,
                new_native_boolean_calls=0,regularized_set_equality_certified=False,terminal_convergence_certified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('public_result_sha256')
    parser.add_argument('checker_source',type=Path)
    args=parser.parse_args()
    import sys
    sys.set_int_max_str_digits(100000)
    print(json.dumps(replay(args.directory,args.public_result_sha256,args.checker_source),indent=2))
