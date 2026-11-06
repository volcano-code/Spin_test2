"""Four same-build Heart captures for the literal native coefficient c[0]."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

from m16_native_acceptance import file_hash, require, strict_json, validate_module_manifest
from m23_fresh_point0 import HERE, NATIVE_C, ROOT, capture, write_json
from native_cherchi_teacher import digest

H = 3e-6
# Canonical coefficient digests from H8f material packets 001 through 004.
# The original five-point stencil visits -h, -h/2, the base, +h/2 and +h.
SAMPLES = (
    (-H,   'ad07c7d33d447d4eaf17dd81fdfca6194d13f149a3b246561dba5e22bc32c397'),
    (-H/2, '5db38b36ab462ae84dea6218a23c2f97a1ef38047a4f5d79f772379116bdbfae'),
    ( H/2, '05e19d9aa54ec2bae1be56dffe75f8ad0c72928930ab5093491b0778262704b8'),
    ( H,   '22f15c3f44f63706e8e7312674d2a94491ee4cab9da2a06a80658d0bb3c77967'),
)


def plan(group_id, modules, source_checkout, *, columns=None):
    baseline = strict_json((HERE/'M23_HEART_POINT000_FRESH.json').read_bytes())['runner_replay']
    require(modules == baseline['module_sha256'], 'M23_NATIVE_MODULE_DRIFT')
    registered = {0:SAMPLES} if columns is None else columns
    points = []
    for column, samples in registered.items():
        for offset, expected in samples:
            index = len(points) + 1
            c = list(NATIVE_C)
            c[column] += offset
            require(digest(c) == expected, 'H8F_STENCIL_COEFFICIENT_DRIFT')
            delta = Fraction(c[column]) - Fraction(NATIVE_C[column])
            point = dict(index=index, run_id=f'{group_id}-{index:03d}', nominal_offset=offset,
                         actual_offset_exact=str(delta), native_c_sha256=expected, native_c=c)
            if columns is not None:
                point['column'] = column
            points.append(point)
    result = dict(group_id=group_id, source_checkout=source_checkout,
                  base_run_id=baseline['run_id'], base_capture_sha256=baseline['capture_file_sha256'],
                  module_sha256=modules, request_sha256=baseline['request_sha256'], h=H, samples=points)
    if columns is None:
        result.update(schema='w2-m24-c0-plan-v1', column=0)
    else:
        result.update(schema='w2-m25-linear-plan-v1', columns=list(columns))
    return result


def campaign(out, manifest, prepared_request, *, columns=None):
    out = out.resolve()
    out.mkdir(exist_ok=False)
    modules = {name: m['sha256'] for name, m in validate_module_manifest(manifest).items()}
    source = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    stencil = plan(out.name, modules, source, columns=columns)
    plan_path = out/'stencil.json'
    write_json(plan_path, stencil)
    plan_sha = file_hash(plan_path)
    records = []
    for point in stencil['samples']:
        sample = out/point['run_id']
        record = capture(sample, manifest, prepared_request,
                         native_c=point['native_c'], expected_c_sha=point['native_c_sha256'],
                         sample_schema='w2-m25-linear-sample-v1' if columns is not None else 'w2-m24-c0-sample-v1')
        require(record['run_id'] == point['run_id'] and record['source_checkout'] == source
                and record['native_c_sha256'] == point['native_c_sha256']
                and record['request_sha256'] == stencil['request_sha256']
                and record['module_sha256'] == modules and record['native_cherchi_calls'] == 1
                and record['native_stage_call_counts']['boolean'] == 1, 'SAMPLE_IDENTITY_DRIFT')
        records.append(dict(index=point['index'], run_id=point['run_id'],
                            capture_sha256=file_hash(sample/'capture.json'),
                            same_call_material_sha256=record['same_call_material_sha256']))
        if columns is not None:
            records[-1]['column'] = point['column']
    require(file_hash(plan_path) == plan_sha, 'STENCIL_PLAN_CHANGED')
    result = dict(schema='w2-m25-linear-result-v1' if columns is not None else 'w2-m24-c0-result-v1',
                  group_id=out.name, stencil_sha256=plan_sha,
                  fresh_native_boolean_calls=len(records), samples=records, J_D=None,
                  training_admitted=False)
    write_json(out/'stencil-result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepared-request', type=Path, required=True)
    parser.add_argument('--module-manifest', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(campaign(args.output, args.module_manifest, args.prepared_request), indent=2))
