"""Single-request Cherchi adapter; unverified derivatives never become truth.

A caller passes an already constructed InProcessCherchiEvaluator. Exactly one
native call is made; no Newton solve or shared core mutation is performed.
"""
import hashlib
import importlib
import math
import resource
import time
from pathlib import Path

try:
    from .native_cherchi_teacher import seal, vector, canonical
    from .independent_reference import integrate_exact_boundary, compare_exact_reference, compare_predeclared_budget
except ImportError:
    from native_cherchi_teacher import seal, vector, canonical
    from independent_reference import integrate_exact_boundary, compare_exact_reference, compare_predeclared_budget


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rss_bytes():
    # ru_maxrss is a process lifetime high-water mark on Linux, not a per-call peak.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


def _authenticated_native(evaluator, native_module_path):
    cls=type(evaluator)
    if (cls.__module__!="make_it_stand.verification.cherchi_inprocess" or
        cls.__name__!="InProcessCherchiEvaluator"):
        return False
    try:
        module=importlib.import_module(cls.__module__)
        actual=module.InProcessCherchiEvaluator
        native_file=evaluator.native.__file__
    except (ImportError, AttributeError):
        return False
    return (cls is actual and
            Path(native_file).resolve()==Path(native_module_path).resolve())


def audit_same_call_exact_boundary(evaluator, output, density, *, reference_absolute_tolerances=None):
    """Compare a Cherchi same-call material with an independent exact formula.

    Does NOT establish non-self-intersection; an independently sealed geometry
    and legal Boolean material proof remain separate requirements.
    """
    last=evaluator.last
    if not isinstance(last,dict) or not last.get("geometry_hash"):
        raise ValueError("MISSING_SAME_CALL_BOUNDARY")
    if (last["geometry_hash"]!=output.get("geometry_hash") or
        last.get("topology_hash")!=output.get("topology_hash")):
        raise ValueError("STALE_OR_WRONG_BRANCH_BOUNDARY")
    if float(evaluator.request.material_density)!=float(density):
        raise ValueError("REFERENCE_DENSITY_MISMATCH")
    faces=last["faces"]
    if hasattr(faces,"tolist"):
        faces=faces.tolist()
    reference,proof=integrate_exact_boundary(last["exact_vertices"],faces,density)
    if reference_absolute_tolerances is None:
        # Direct analytic unit-test compatibility only; production native calls
        # below are forced to supply an a-priori per-case ten-component budget.
        comparison=compare_exact_reference(output["variable_raw_moments"],reference)
    else:
        comparison=compare_predeclared_budget(output["variable_raw_moments"],reference,
                                              reference_absolute_tolerances)
    if not comparison["within_tolerance"]:
        raise ValueError("INDEPENDENT_TEN_MOMENT_MISMATCH:"+
                         str(comparison["maximum_normalized_error"]))
    return dict(proof,comparison=comparison)



def audit_same_call_attachment(evaluator, output, native_c):
    """Require the producer's persisted material to match *this* evaluation.

    Only invoke for authenticated InProcessCherchiEvaluator. The producer's own
    validator protects its serialized material against mismatched same-call
    packets. This adds independent cross-checks of all returned moment values
    and of the native input coefficient order, without rebuilding the surface.
    """
    key=tuple(vector(native_c))
    attachment=evaluator.material(key)
    evaluator.validate_same_call_material(attachment,key,evaluator.request)
    if vector(attachment.get('quadric_coefficients'))!=list(key):
        raise ValueError('SAME_CALL_NATIVE_C_MISMATCH')
    for field in ('raw_moments','variable_raw_moments'):
        if vector(attachment.get(field))!=vector(output.get(field)):
            raise ValueError('SAME_CALL_MOMENT_ATTACHMENT_MISMATCH:'+field)
    jacobian=attachment.get('moment_jacobian')
    observed=output.get('moment_jacobian')
    if jacobian is None or observed is None or len(jacobian)!=10 or len(observed)!=10:
        raise ValueError('SAME_CALL_JACOBIAN_SHAPE_MISMATCH')
    if any(vector(a)!=vector(b) for a,b in zip(jacobian,observed)):
        raise ValueError('SAME_CALL_JACOBIAN_ATTACHMENT_MISMATCH')
    for field in ('geometry_hash','topology_hash','evaluation_map_identity',
                  'boolean_backend_identity','derivative_identity','derivative_defined'):
        if attachment.get(field)!=output.get(field):
            raise ValueError('SAME_CALL_ATTACHMENT_IDENTITY_MISMATCH:'+field)
    if not isinstance(attachment.get('material_proof'),dict) or not attachment.get('variable_rational_vertices'):
        raise ValueError('SAME_CALL_EXACT_MATERIAL_PROOF_MISSING')
    return {'schema':'w2-same-call-material-binding-v1',
            'serialized_material_sha256':hashlib.sha256(canonical(attachment).encode()).hexdigest(),
            'native_c_sha256':hashlib.sha256(canonical(list(key)).encode()).hexdigest(),
            'coefficient_order':'CONST,X,Y,Z,XX,YY,ZZ,YZ,XZ,XY',
            'native_c_to_polynomial_q':'Bdiag=1,1,1,1,1,1,1,2,2,2',
            'source':'producer.material + producer.validate_same_call_material'}


def evaluate_one(evaluator, q, *, model_path, native_module_path,
                 geometry, density, frame, derivative_contract=None,
                 split="excluded", exclusion_reason=None,
                 reference_absolute_tolerances=None):
    """Create replayable pre-W0 label for one evaluation, including failures.

    Passing claimed derivative contracts never grants a VALID Jacobian until
    W0's independently authenticating verifier is integrated. The valid
    *physical moments* survive this derivative exclusion.
    """
    model_sha=sha256_file(model_path)
    native_sha=sha256_file(native_module_path)
    q=vector(q)
    native_authenticated=_authenticated_native(evaluator,native_module_path)
    stage_before=dict(getattr(evaluator,"stage_seconds",{}) or {})
    start=time.perf_counter()
    rss_before=_rss_bytes()
    failure=None
    extra_proof={}
    try:
        output=evaluator(tuple(q))
        if not isinstance(output,dict):
            raise ValueError("EVALUATOR_RETURN_NOT_MAPPING")
        variable=vector(output["variable_raw_moments"])
        total=vector(output["raw_moments"])
        fixed=vector(evaluator.fixed)
        if any(not math.isclose(total[i],variable[i]+fixed[i],rel_tol=1e-11,abs_tol=1e-11)
               for i in range(10)):
            raise ValueError("DOUBLE_COUNT_OR_FIXED_MISMATCH")
        if output.get("evaluation_map_identity")!=evaluator.context_identity:
            raise ValueError("EVALUATION_MAP_IDENTITY_MISMATCH")
        if not output.get("geometry_hash") or not output.get("topology_hash"):
            raise ValueError("MISSING_GEOMETRY_TOPOLOGY_PROOF")
        if native_authenticated:
            if reference_absolute_tolerances is None:
                raise ValueError('PREDECLARED_TEN_MOMENT_ERROR_BUDGET_REQUIRED')
            extra_proof['same_call_attachment']=audit_same_call_attachment(evaluator,output,q)
            if 'capture_run_id' in geometry:
                counts = dict(evaluator.call_counts)
                if counts.get('source') != 1 or counts.get('boolean') != 1:
                    raise ValueError('CAPTURE_REQUIRES_ONE_NATIVE_BOOLEAN')
                extra_proof['native_stage_call_counts'] = counts
            exact_start=time.perf_counter()
            extra_proof["independent_ten_moment_audit"]=audit_same_call_exact_boundary(
                evaluator,output,density,
                reference_absolute_tolerances=reference_absolute_tolerances)
            extra_proof["reference_elapsed_seconds"]=time.perf_counter()-exact_start
        # Reading both files again catches replacement during a long native call.
        # Merely supplying their original hashes cannot bless changed binary bytes.
        if sha256_file(model_path)!=model_sha or sha256_file(native_module_path)!=native_sha:
            raise ValueError('FROZEN_MODEL_OR_NATIVE_MODULE_CHANGED_DURING_CALL')
        topology="OBSERVED_DISCRETE_MAP"
        derivative="UNKNOWN"
        jacobian=None
        reason=("W0_VERIFIER_NOT_INTEGRATED" if derivative_contract is not None
                else "NO_W0_NEIGHBORHOOD_CERTIFICATE")
        proof={"kind":"cherchi_same_call_discrete_map" if native_authenticated else "mock_unverified_map",
               "geometry_hash":output["geometry_hash"],
               "topology_hash":output["topology_hash"],
               "evaluation_map_identity":evaluator.context_identity,
               "boolean_backend_identity":output.get("boolean_backend_identity"),
               "derivative_identity":output.get("derivative_identity"),
               "derivative_admission_reason":reason,
               "native_cherchi":native_authenticated,**extra_proof}
    except Exception as exc:
        variable=jacobian=fixed=None
        topology="FAILED"
        derivative="INVALID"
        proof={"kind":"failed_same_call","native_cherchi":native_authenticated,**extra_proof}
        failure=f"{type(exc).__name__}: {exc}"
    elapsed=time.perf_counter()-start
    raw_stages=dict(getattr(evaluator,"stage_seconds",{}) or {})
    stages={key:max(0.0,float(value)-float(stage_before.get(key,0)))
            for key,value in raw_stages.items()}
    stages["w2_total_wall_seconds"]=elapsed
    if 'reference_elapsed_seconds' in extra_proof:
        stages['w2_independent_reference_seconds']=extra_proof['reference_elapsed_seconds']
    rss_after=_rss_bytes()
    proof["memory_measurement"]={"kind":"PROCESS_LIFETIME_RSS_HIGH_WATER_LINUX",
        "before_bytes":rss_before,"after_bytes":rss_after,
        "is_per_call_peak":False}
    if failure is not None or derivative!="VALID" or not native_authenticated:
        split="excluded"
    return seal(model_sha256=model_sha,native_module_sha256=native_sha,
        geometry=geometry,density=density,frame=frame,q=q,r_d=variable,j_d=jacobian,
        r_f=fixed,topology_status=topology,derivative_status=derivative,
        derivative_contract=None,stage_seconds=stages,peak_rss_bytes=rss_after,
        exact_proof=proof,split=split,failure_reason=failure or exclusion_reason)
