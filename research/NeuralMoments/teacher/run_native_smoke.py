"""Run one frozen native Cherchi request, with fail-closed pre-W0 evidence.

This is NOT a Newton optimizer and does not release training labels. A compiled
Spin/Cherchi build and previously verified sealed input are prerequisites.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

try:
    from .native_cherchi_teacher import canonical, digest, load, vector
    from .cherchi_single_request import evaluate_one, sha256_file
    from .independent_reference import compare_predeclared_budget
    from .frozen_model_binding import validate_model_binding
except ImportError:
    from native_cherchi_teacher import canonical, digest, load, vector
    from cherchi_single_request import evaluate_one, sha256_file
    from independent_reference import compare_predeclared_budget
    from frozen_model_binding import validate_model_binding

FROZEN_SOURCE_SHA='0bcc187b3005478baa586a6d6f450e73811ca342'
SCHEMA='w2-one-cherchi-request-v3'
LEGACY_SCHEMA='w2-one-cherchi-request-v2'
_SHA=re.compile(r'^[0-9a-f]{64}$')


def _require_sha(token, field):
    if not isinstance(token,str) or _SHA.fullmatch(token) is None:
        raise ValueError('INVALID_'+field+'_SHA256')
    return token


def parse_spec(path):
    spec=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(spec,dict) or spec.get('schema') not in (SCHEMA,LEGACY_SCHEMA):
        raise ValueError('W2_V2_OR_V3_FROZEN_SPEC_REQUIRED')
    if spec.get('source_sha')!=FROZEN_SOURCE_SHA:
        raise ValueError('SOURCE_REVISION_MISMATCH')
    if not isinstance(spec.get('request'),dict) or not isinstance(spec.get('evaluator_settings'),dict):
        raise ValueError('REQUEST_OR_SETTINGS_MISSING')
    if 'q' in spec:
        raise ValueError('AMBIGUOUS_Q_VS_NATIVE_C;SPECIFY_NATIVE_C_ONLY')
    spec['native_c']=vector(spec.get('native_c'))
    if all(x==0.0 for x in spec['native_c']):
        raise ValueError('DEGENERATE_ZERO_NATIVE_C')
    if digest(spec['native_c'])!=_require_sha(spec.get('expected_native_c_sha256'),'NATIVE_C'):
        raise ValueError('FROZEN_NATIVE_C_IDENTITY_MISMATCH')
    model=Path(spec.get('model_path',''))
    if not model.is_file():
        raise ValueError('FROZEN_MODEL_FILE_NOT_FOUND')
    if sha256_file(model)!=_require_sha(spec.get('expected_model_sha256'),'MODEL'):
        raise ValueError('FROZEN_MODEL_SHA256_MISMATCH')
    _require_sha(spec.get('expected_native_module_sha256'),'NATIVE_MODULE')
    if not isinstance(spec.get('geometry'),dict) or not isinstance(spec.get('frame'),dict):
        raise ValueError('GEOMETRY_OR_FRAME_MISSING')
    if spec.get('derivative_contract') is not None:
        raise ValueError('W0_DERIVATIVE_CONTRACT_UNAVAILABLE')
    # The 10 separately declared absolute budgets are in the native target
    # frame's mass, first and second moment units, not a universal percentage.
    compare_predeclared_budget([0.0]*10,[0.0]*10,
                               spec.get('reference_absolute_tolerances'))
    if spec['schema']==SCHEMA:
        spec['_verified_model_binding']=validate_model_binding(
            model, spec['request'],spec.get('model_geometry_binding'),
            spec.get('normalization'))
    return spec


def evaluate_from_spec(spec, *, capture_material=None):
    """Lazy imports allow infrastructure failures to be logged without mock data."""
    import numpy as np
    # V2 intentionally remains parseable for historical test-only failures,
    # but has no geometrical source binding and can NEVER launch native work.
    if spec.get('schema')!=SCHEMA or not spec.get('_verified_model_binding'):
        raise ValueError('NATIVE_REQUIRES_V3_VERIFIED_MODEL_REQUEST_BINDING')
    binding=validate_model_binding(spec['model_path'],spec['request'],
        spec['model_geometry_binding'],spec.get('normalization'))
    if binding!=spec['_verified_model_binding']:
        raise ValueError('FROZEN_OBJ_CHANGED_SINCE_PREFLIGHT')
    from make_it_stand.spin import SpinItFasterRequest, SpinSolverConfig, SpinEvaluatorConfig
    from make_it_stand.verification.cherchi_inprocess import InProcessCherchiEvaluator
    request_data=dict(spec['request'])
    if isinstance(request_data.get('settings'),dict):
        request_data['settings']=SpinSolverConfig(**request_data['settings'])
    request=SpinItFasterRequest(**request_data)
    settings=SpinEvaluatorConfig(**spec['evaluator_settings'])
    from make_it_stand import spin
    native=spin._native_module()
    if sha256_file(native.__file__)!=spec['expected_native_module_sha256']:
        raise ValueError('FROZEN_NATIVE_MODULE_SHA256_MISMATCH')
    # Source design boundary validity is necessary but not sufficient.
    boundary=native.audit_exact_triangle_boundary(
        np.asarray(request.design_vertices,dtype=np.float64),
        np.asarray(request.design_faces,dtype=np.int32),'AnyNonzero')
    if not boundary['boundary_valid']:
        raise ValueError('OPEN_OR_ILLEGAL_DESIGN_BOUNDARY')
    evaluator=InProcessCherchiEvaluator(
        request,settings,normalization=spec.get('normalization'),
        design_vertices_physical=spec.get('design_vertices_physical'),
        rational_arithmetic_backend=spec.get('rational_arithmetic_backend','PYTHON_FRACTION_V1'))
    if sha256_file(evaluator.native.__file__)!=spec['expected_native_module_sha256']:
        raise ValueError('EVALUATOR_NATIVE_MODULE_IDENTITY_CHANGED')
    # This binds declared request/settings to the output. It does *not* prove
    # that model_path's bytes represent that source mesh: a separate frozen
    # provenance audit is still necessary before any scientific release.
    geometry=dict(spec['geometry'])
    geometry['_w2_frozen_bindings']={
        'source_sha':spec['source_sha'],
        'request_sha256':digest(spec['request']),
        'evaluator_settings_sha256':digest(spec['evaluator_settings']),
        'native_c_sha256':spec['expected_native_c_sha256'],
        'spec_sha256':digest(spec),
        'model_to_request_geometry_verified':False,
        'outer_mesh_binary64_correspondence_verified':True,
        'model_to_request_geometry_proof':binding,
    }
    record = evaluate_one(evaluator,spec['native_c'],model_path=spec['model_path'],
        native_module_path=evaluator.native.__file__,geometry=geometry,
        density=request.material_density,frame=spec['frame'],split='excluded',
        reference_absolute_tolerances=spec['reference_absolute_tolerances'],
        exclusion_reason='W0_VERIFIER_AND_INDEPENDENT_SOLID_PROOF_NOT_VERIFIED')
    if capture_material is not None and record['payload']['topology_status'] == 'OBSERVED_DISCRETE_MAP':
        # Read the evaluator's same-call cache before it goes out of scope.
        capture_material.update(evaluator.material(tuple(spec['native_c'])))
    return record


def _write_new_atomically(path,record):
    """Fsync a complete record before publishing it; never replace evidence."""
    if os.path.lexists(path):
        raise FileExistsError('REFUSE_OVERWRITE_EXISTING_EVIDENCE')
    tmp=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,
             prefix='.'+path.name+'.',suffix='.partial',delete=False) as stream:
            tmp=Path(stream.name)
            stream.write(canonical(record)+'\n')
            stream.flush();os.fsync(stream.fileno())
        os.link(tmp,path)  # atomic exclusive publication, never overwrite
        fd=os.open(path.parent,os.O_RDONLY | getattr(os,'O_DIRECTORY',0))
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _write_bound_material(path, material, proof):
    expected = proof['same_call_attachment']['serialized_material_sha256']
    if digest(material) != expected or material['evaluation_map_identity'] != proof['evaluation_map_identity']:
        raise ValueError('CAPTURED_MATERIAL_BINDING_MISMATCH')
    _write_new_atomically(path, material)
    # Compare the canonical bytes actually persisted, excluding the single trailing LF.
    if path.read_bytes() != (canonical(material) + '\n').encode('utf-8'):
        raise ValueError('PERSISTED_MATERIAL_BYTES_MISMATCH')


def run(spec_path,output_path, *, material_output=None):
    path=Path(output_path)
    if os.path.lexists(path):
        raise FileExistsError('REFUSE_OVERWRITE_EXISTING_EVIDENCE')
    material_path = Path(material_output) if material_output is not None else None
    if material_path is not None:
        if material_path.resolve() == path.resolve() or os.path.lexists(material_path):
            raise FileExistsError('MATERIAL_OUTPUT_REQUIRES_DISTINCT_NEW_PATH')
    start=time.perf_counter()
    phase='PREFLIGHT'
    try:
        spec_content_before=sha256_file(spec_path)
        spec=parse_spec(spec_path)
        phase='NATIVE_SETUP_OR_EVALUATION'
        material = {}
        record = (evaluate_from_spec(spec, capture_material=material) if material_path is not None
                  else evaluate_from_spec(spec))
        phase='POSTFLIGHT'
        if sha256_file(spec_path)!=spec_content_before:
            raise ValueError('FROZEN_SPEC_CHANGED_DURING_EVALUATION')
        if spec['schema']==SCHEMA:
            binding_after=validate_model_binding(spec['model_path'],spec['request'],
                spec['model_geometry_binding'],spec.get('normalization'))
            if binding_after!=spec['_verified_model_binding']:
                raise ValueError('MODEL_REQUEST_GEOMETRY_CHANGED_DURING_EVALUATION')
        if record['payload'].get('exact_proof',{}).get('native_cherchi') is not True:
            raise ValueError('NON_NATIVE_EVALUATION_RETURN')
        if (record['identity']['model_sha256']!=spec['expected_model_sha256'] or
            record['identity']['native_module_sha256']!=spec['expected_native_module_sha256'] or
            record['identity']['q_sha256']!=digest(spec['native_c'])):
            raise ValueError('RETURNED_LABEL_SOURCE_IDENTITY_MISMATCH')
        if material_path is not None and record['payload']['topology_status'] == 'OBSERVED_DISCRETE_MAP':
            _write_bound_material(material_path, material, record['payload']['exact_proof'])
        # Save only after complete checks and then independently reopen it.
        _write_new_atomically(path,record)
        body=load(path,expected_q=spec['native_c'],
             expected_model_sha256=spec['expected_model_sha256'],
             expected_native_module_sha256=spec['expected_native_module_sha256'])
        return {'status':'LABEL_RECORDED', 'label_path':str(path),
                'native_cherchi':body['exact_proof']['native_cherchi'],
                'topology_status':body['topology_status'],
                'derivative_status':body['derivative_status'],
                'training_admitted':False,
                'wall_seconds':time.perf_counter()-start,
                'failure_reason':body['failure_reason']}
    except FileExistsError:
        raise
    except Exception as error:
        # Import failures are preflight failures: zero native evaluations.
        kind=('FAILED_PREFLIGHT' if (phase=='PREFLIGHT' or isinstance(error,ImportError) or
               str(error)=='NATIVE_REQUIRES_V3_VERIFIED_MODEL_REQUEST_BINDING')
              else 'FAILED_'+phase)
        failure={'schema':'w2-native-smoke-preflight-v2','status':kind,
                 'phase':phase,'error':f'{type(error).__name__}: {error}',
                 'elapsed_seconds':time.perf_counter()-start,
                 'training_admitted':False,'native_cherchi_verified':False}
        _write_new_atomically(path,failure)
        return failure


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--material-output', help='New path for the full native same-call JSON attachment')
    args=parser.parse_args(argv)
    result=run(args.spec,args.output,material_output=args.material_output)
    print(canonical(result))
    return 0 if result.get('status')=='LABEL_RECORDED' and result.get('failure_reason') is None else 2


if __name__=='__main__':
    sys.exit(main())
