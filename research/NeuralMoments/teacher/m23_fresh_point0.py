"""Run one fresh historical Heart input in the guarded 9 GiB diagnostic profile."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from m16_native_acceptance import (HEART_SOURCE_SHA, MODULES, REQUEST_SHA, SOURCE_SHA,
                                   file_hash, h256, require, strict_json, validate_module_manifest)
from native_cherchi_teacher import digest
from run_native_smoke import run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NATIVE_C = [0.3701323978732728, -0.15863761828595124, -0.0, -0.7608730119328398,
            0.293773367196206, 0.293773367196206, 0.293773367196206, 0.0, 0.0, 0.0]
C_SHA = '8be318f8e19909a2c63ba5287d712665b9723b1b7a9cca42dc4ba23c16819a47'
MODEL_SHA = 'cde5dbdf894c181c27abf22abe8bce8b82a2a3a68b141cfb24749ef07e864062'
GIB = 1024**3


# Trees of the frozen numerical implementation. The W2 harness may evolve.
CORE_TREES = {
    'python': 'bbc921e8722d2f72c0a91038f3267f9ef67d6f5f',
    'src': '82a6db5297f5c09d36d01c8f29975f8eb7a66183',
    'bindings': '68943bb2b41d6649580ca914b1c15016f7a6b7b2',
    'CMakeLists.txt': '29ce142935cac9ec1f183450391311445e25bd5f',
}


def check_core():
    tree = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', 'HEAD', '--', *CORE_TREES],text=True)
    observed = {row.split()[3]:row.split()[2] for row in tree.splitlines()}
    require(observed == CORE_TREES, 'FROZEN_CORE_IDENTITY_CHANGED')
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--',*CORE_TREES],check=True)


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False, separators=(',', ':'))
        f.write('\n')


def resource_snapshot():
    mount = Path('/sys/fs/cgroup')
    relative = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                    if line.startswith('0::'))
    scope = mount / relative.lstrip('/')
    require(scope.resolve().is_relative_to(mount) and (scope/'cgroup.procs').is_file(),
            'CGROUP_V2_SCOPE_REQUIRED')
    paths = [scope, *[p for p in scope.parents if p.is_relative_to(mount)]]

    def limit(field):
        values = [(p / field).read_text().strip() for p in paths if (p / field).is_file()]
        return min((int(v) for v in values if v != 'max'), default=None)

    host = {line.split(':')[0]: int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemAvailable:', 'SwapTotal:'))}
    events = dict(line.split() for line in (scope / 'memory.events').read_text().splitlines())
    return dict(cgroup=relative, memory_high=limit('memory.high'), memory_max=limit('memory.max'),
                swap_max=limit('memory.swap.max'), available=host['MemAvailable'],
                host_swap=host['SwapTotal'], memory_events={k:int(v) for k,v in events.items()},
                peak=int((scope / 'memory.peak').read_text()))


def qualify(snapshot, *, startup):
    require(snapshot['memory_max'] == 9*GIB and snapshot['memory_high'] == 7*GIB
            and snapshot['swap_max'] == snapshot['host_swap'] == 0, '9GIB_NO_SWAP_PROFILE_REQUIRED')
    if startup:
        require(snapshot['available'] >= 11*GIB, '11GIB_HOST_AVAILABLE_REQUIRED')


def prepare(out, modules, prepared_request, *, native_c=NATIVE_C, expected_c_sha=C_SHA):
    raw = prepared_request.read_bytes()
    require(h256(raw) == REQUEST_SHA and digest(native_c) == expected_c_sha, 'FROZEN_REQUEST_OR_C_DRIFT')
    frozen = strict_json(raw)
    request = frozen['request']
    model = out/'frozen_normalized_outer.obj'
    with model.open('x', encoding='utf-8') as f:
        f.write('# W2 M16 exact binary64 normalized H8f outer VIEW; not source CAD\n')
        for p in request['outer_vertices']:
            f.write('v '+' '.join(repr(float(x)) for x in p)+'\n')
        for a,b,c in request['outer_faces']:
            f.write(f'f {a+1} {b+1} {c+1}\n')
    require(file_hash(model) == MODEL_SHA, 'FROZEN_NORMALIZED_OBJ_MISMATCH')
    spec = dict(schema='w2-one-cherchi-request-v3', source_sha=SOURCE_SHA,
                request=request, evaluator_settings=frozen['evaluator_settings'], native_c=native_c,
                expected_native_c_sha256=expected_c_sha, model_path=str(model.resolve()),
                expected_model_sha256=MODEL_SHA, expected_native_module_sha256=modules[MODULES[0]]['sha256'],
                reference_absolute_tolerances=[1e-10]*10, derivative_contract=None,
                rational_arithmetic_backend='NATIVE_GMP_SURFACE_V1',
                model_geometry_binding={'mode':'OBJ_IS_INPUT_NORMALIZED_OUTER_V1'},
                frame=dict(kind='INPUT_NORMALIZED_TARGET_V1', native_target_origin=request['target_origin'],
                           native_target_axes=request['target_axes']),
                geometry=dict(capture_run_id=out.name, historical_model_source_sha256=HEART_SOURCE_SHA['initial_top.obj'],
                              historical_design_sha256=HEART_SOURCE_SHA['design_domain.obj'],
                              historical_fixed_sha256=HEART_SOURCE_SHA['fixed_shell.obj'],
                              new_view_sha256=MODEL_SHA, cherchi_bridge_sha256=modules[MODULES[1]]['sha256'],
                              gmp_backend_sha256=modules[MODULES[2]]['sha256']))
    write_json(out/'spec.json',spec)
    return digest(request)


def capture(out, manifest, prepared_request, *, native_c=NATIVE_C, expected_c_sha=C_SHA,
            sample_schema="w2-m24-c0-sample-v1"):
    # The exact H8f signed moments require more decimal digits than Python's default.
    sys.set_int_max_str_digits(100000)
    out = out.resolve()
    out.mkdir(exist_ok=False)
    check_core()
    before = resource_snapshot()
    write_json(out/'resources-before.json', before)
    qualify(before, startup=True)
    # Preserve the independent build receipt before opening the evaluator.
    frozen_manifest = out/'module-manifest.json'
    shutil.copyfile(manifest, frozen_manifest)
    external_sha = file_hash(manifest)
    modules = validate_module_manifest(frozen_manifest)
    from make_it_stand import spin
    import _cherchi_core_boolean as bridge
    import _cherchi_rational_moments as gmp
    for name, module in zip(MODULES, (spin._native_module(), bridge, gmp)):
        require(Path(module.__file__).resolve() == Path(modules[name]['path']), 'WRONG_MODULE_PATH:'+name)
    require(Path(spin.__file__).resolve() == ROOT/'python/make_it_stand/spin.py', 'WRONG_PYTHON_SOURCE')
    request_sha = prepare(out, modules, prepared_request, native_c=native_c, expected_c_sha=expected_c_sha)
    result = run(out/'spec.json', out/'label.json', material_output=out/'material.json')
    write_json(out/'native-result.json', result)
    after = resource_snapshot()
    write_json(out/'resources-after.json', after)
    qualify(after, startup=False)
    require(all(after['memory_events'][e] == before['memory_events'][e]
                for e in ('oom', 'oom_kill', 'max')), 'RESOURCE_BUDGET_EXCEEDED')
    require(result.get('status') == 'LABEL_RECORDED' and result.get('native_cherchi') is True,
            'NATIVE_EVALUATION_FAILED')
    require(file_hash(frozen_manifest) == file_hash(manifest) == external_sha and
            validate_module_manifest(frozen_manifest) == modules, 'BUILD_DRIFT')
    label = strict_json((out/'label.json').read_bytes())['payload']
    proof = label['exact_proof']
    exact = proof['independent_ten_moment_audit']
    require(proof['native_stage_call_counts']['boolean'] == 1 and
            label['geometry']['capture_run_id'] == out.name and
            label['J_D'] is None and label['split'] == 'excluded' and
            exact['comparison']['within_tolerance'] is True, 'NUMERIC_DIAGNOSTIC_NOT_VERIFIED')
    material = strict_json((out/'material.json').read_bytes())
    hashes = {n: m['sha256'] for n,m in modules.items()}
    require(material['evaluation_map_identity'] == proof['evaluation_map_identity'] and
            material['boolean_backend_identity'].endswith(':'+hashes[MODULES[1]]) and
            material['rational_arithmetic_identity']['module_sha256'] == hashes[MODULES[2]], 'MATERIAL_RUNTIME_DRIFT')
    material_sha = digest(material)
    require(material_sha == proof['same_call_attachment']['serialized_material_sha256'], 'MATERIAL_HASH_DRIFT')
    files = {p.name:file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}
    record = dict(schema='w2-m23-fresh-heart-capture-v1',run_id=out.name,files=files,
                  source_checkout=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                  module_sha256=hashes, request_sha256=request_sha, native_cherchi_calls=1,
                  native_stage_call_counts=proof['native_stage_call_counts'], same_call_material_sha256=material_sha,
                  exact_boundary_sha256=exact['exact_boundary_sha256'],evaluation_map_identity=proof['evaluation_map_identity'],
                  fresh_evaluation_verified=True, training_admitted=False,J_D=None)
    if expected_c_sha != C_SHA:
        record.update(schema=sample_schema, native_c_sha256=expected_c_sha)
    write_json(out/'capture.json',record)
    return record


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepared-request',type=Path,required=True)
    parser.add_argument('--module-manifest',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(capture(args.output,args.module_manifest,args.prepared_request),indent=2))
