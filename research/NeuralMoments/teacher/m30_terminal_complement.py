"""Postprocess a converged native terminal using its saved cut surface."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

from m16_native_acceptance import MODULES, REQUEST_SHA, file_hash, require, strict_json, validate_module_manifest
from m23_fresh_point0 import HERE, check_core, qualify, resource_snapshot
from m29_native_first_step import json_array
from native_cherchi_teacher import digest


def terminal_index(result, events):
    require(result['status']=='NATIVE_TERMINAL' and result['native_terminal']=='CONVERGED', 'NOT_CONVERGED')
    terminal = events[-1]
    require(terminal['type']=='attempt_terminal' and terminal['reason']=='converged', 'NO_CONVERGED_RESTART')
    accepted = [e for e in events if e['type']=='evaluation' and e['outcome']=='accepted'
                and e['restart']==terminal['restart']]
    require(bool(accepted), 'NO_ACCEPTED_MATERIAL')
    last = accepted[-1]
    q = result['native_terminal_summary']['quadric_coefficients']
    require(last['quadric_coefficients']==q and last['evaluation_index']==terminal['evaluation_index'],
            'TERMINAL_ACCEPTED_IDENTITY_MISMATCH')
    return last['evaluation_index'], q, last


def outer_minus_void(request, carved_vertices, carved_faces):
    # Frozen Heart is a strictly nested design domain with unit scale and identity axes.
    require(request['target_axes']==[[1,0,0],[0,1,0],[0,0,1]], 'UNSUPPORTED_TARGET_AXES')
    origin = list(map(Fraction,request['target_origin']))
    def token(value):return f'{value.numerator}/{value.denominator}'
    outer = [[token(Fraction(x)-a) for x,a in zip(p,origin)] for p in request['outer_vertices']]
    offset = len(outer)
    return dict(frame='M23_INPUT_NORMALIZED_TARGET', vertices=outer+carved_vertices,
                faces=request['outer_faces']+[[offset+c,offset+b,offset+a] for a,b,c in carved_faces])


def capture(folder):
    from make_it_stand import spin
    from make_it_stand.verification.cherchi_inprocess import InProcessCherchiEvaluator
    import _cherchi_core_boolean as bridge
    import _cherchi_rational_moments as arithmetic

    sys.set_int_max_str_digits(100000)
    result = strict_json((folder/'result.json').read_bytes())
    if result['status']!='NATIVE_TERMINAL' or result['native_terminal']!='CONVERGED':
        return 'NO_TERMINAL_COMPLEMENT_REQUIRED'
    require(result['schema']=='w2-m30-trajectory-v1' and result['run_id']==folder.name,
            'WRONG_TRAJECTORY_RECEIPT')
    check_core()
    before = resource_snapshot()
    qualify(before,startup=True)
    files = result['file_sha256']
    require(all(file_hash(folder/k)==files[k] for k in ('request.json','module-manifest.json','events.jsonl','evaluations.jsonl'))
            and files['request.json']==REQUEST_SHA, 'CONVERGED_EVIDENCE_DRIFT')
    events=[strict_json(row) for row in (folder/'events.jsonl').read_bytes().splitlines()]
    index,q,event=terminal_index(result,events)
    with (folder/'evaluations.jsonl').open('rb') as stream:
        row=next((r for r in map(strict_json,stream) if r['index']==index),None)
    require(row is not None and row['status']=='returned' and row['native_c']==q,'NO_TERMINAL_MATERIAL')
    kept=row['material']
    require(row['material_sha256']==digest(kept) and kept['quadric_coefficients']==q
            and kept['geometry_hash']==event['material_geometry_hash'] and
            kept['topology_hash']==event['material_topology_hash'], 'TERMINAL_RETAINED_MAP_MISMATCH')
    entries=validate_module_manifest(folder/'module-manifest.json')
    pinned=strict_json((HERE/'M23_HEART_POINT000_FRESH.json').read_bytes())['runner_replay']['module_sha256']
    require({name:v['sha256'] for name,v in entries.items()}==result['module_sha256']==pinned,'ELF_DRIFT')
    for name,module in zip(MODULES,(spin._native_module(),bridge,arithmetic)):
        require(Path(module.__file__).resolve()==Path(entries[name]['path']), 'WRONG_NATIVE_MODULE')
    prepared=strict_json((folder/'request.json').read_bytes())
    request=prepared['request']
    require(digest(request)==result['request_sha256'], 'REQUEST_MISMATCH')
    settings=request['settings']
    terminal=result['native_terminal_summary']
    require(terminal['multiplier_mu']>0 and terminal['raw_equation_residual']<=settings['equation_tolerance']
            and terminal['physical_relative_residual']<=settings['physical_relative_tolerance'],
            'CONVERGENCE_TOLERANCE_OR_MULTIPLIER_FAILED')
    data=dict(request, settings=spin.SpinSolverConfig(**request['settings']))
    req=spin.SpinItFasterRequest(**data)
    evaluator=InProcessCherchiEvaluator(req,spin.SpinEvaluatorConfig(**prepared['evaluator_settings']),
                                        rational_arithmetic_backend='NATIVE_GMP_SURFACE_V1')
    carved=evaluator.complement_of_saved_source(kept)
    require(evaluator.call_counts.get('boolean')==1 and evaluator.call_counts.get('source',0)==0
            and carved['quadric_source_identity']==kept['quadric_source_identity']
            and carved['evaluation_map_identity']==kept['evaluation_map_identity']
            and carved['boolean_operation']!=kept['boolean_operation'], 'TERMINAL_OPPOSITE_SOURCE_MISMATCH')
    mesh=outer_minus_void(request,carved['variable_rational_vertices'],carved['variable_faces'])
    after=resource_snapshot()
    qualify(after,startup=False)
    require(all(after['memory_events'][k]==before['memory_events'][k] for k in ('oom','oom_kill','max')),
            'POSTPROCESS_SCOPE_EXCEEDED')
    packet=dict(schema='w2-m30-converged-complement-v1',run_id=result['run_id'],
                result_sha256=file_hash(folder/'result.json'), evaluation_index=index,
                retained_material_sha256=row['material_sha256'],complement_stage_counts=evaluator.call_counts,
                complement=carved, final_mesh=mesh, outer_vertex_count=len(request['outer_vertices']),
                outer_face_count=len(request['outer_faces']), before=before,after=after,
                whole_boolean_set_equality_verified=False, final_mesh_independently_verified=False)
    with (folder/'terminal-complement.json').open('x') as f:
        json.dump(packet,f,default=json_array,allow_nan=False,separators=(',',':'))
    return 'CONVERGED_COMPLEMENT_CAPTURED'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    print(capture(parser.parse_args().folder))
