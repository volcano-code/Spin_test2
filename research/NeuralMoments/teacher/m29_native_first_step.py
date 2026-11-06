"""Record a bounded prefix of the unchanged strict Faster solver on the frozen Heart request."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from m16_native_acceptance import MODULES, REQUEST_SHA, file_hash, require, strict_json, validate_module_manifest
from m23_fresh_point0 import HERE, ROOT, check_core, qualify, resource_snapshot, write_json
from native_cherchi_teacher import digest

CAP = 20


def json_array(value):
    if isinstance(value, (np.generic, np.ndarray)):
        return value.tolist()
    raise TypeError(type(value).__name__)


def append(stream, row):
    stream.write(json.dumps(row, default=json_array, allow_nan=False, separators=(',', ':')) + '\n')
    stream.flush()


class PrefixComplete(Exception):
    """Controlled exit from the event observer, outside a geometry callback."""


class Recorder:
    def __init__(self, geometry, evaluations, events):
        self.geometry = geometry
        self.evaluations = evaluations
        self.events = events
        self.calls = 0
        self.accepted = None
        self.status = None
        self.fatal = False

    def evaluate(self, q):
        # Defensive bound even if a future native implementation omits decision events.
        if self.calls >= CAP or self.accepted is not None or self.fatal:
            raise RuntimeError('UNEXPECTED_NATIVE_PREFIX_CALLBACK')
        self.calls += 1
        row = dict(index=self.calls, native_c=list(q), call_counts_before=dict(self.geometry.call_counts))
        try:
            value = self.geometry(q)
        except Exception as exc:
            row.update(status='exception', error_type=type(exc).__name__, error=str(exc),
                       call_counts_after=dict(self.geometry.call_counts))
            append(self.evaluations, row)
            raise
        try:
            material = self.geometry.material(q)
            self.geometry.validate_same_call_material(material, q, self.geometry.request)
            require(all(np.array_equal(value[k], material[k]) for k in
                        ('raw_moments', 'variable_raw_moments', 'moment_jacobian')),
                    'CAPTURE_MAP_MATERIAL_DIVERGENCE')
            after = dict(self.geometry.call_counts)
            require(after['boolean'] - row['call_counts_before'].get('boolean', 0) == 1,
                    'CAPTURE_EXPECTED_ONE_BOOLEAN')
            row.update(status='returned', map=value, material=material, material_sha256=digest(material),
                       call_counts_after=after)
            append(self.evaluations, row)
        except Exception:
            self.fatal = True
            raise
        return value

    def observe(self, event):
        # Return normally from the accepted-trial event: C++ can commit candidate
        # and update its trace. Stop at its next event, before another geometry call.
        if self.fatal:
            raise RuntimeError('M29_CAPTURE_INTEGRITY_FAILED')
        append(self.events, event)
        if self.accepted is not None:
            self.status = 'FIRST_TRIAL_ACCEPTED'
            raise PrefixComplete
        if event['type'] == 'evaluation':
            if (event['iteration'] >= 1 and event['outcome'] == 'accepted' and
                    event['reason'] in ('residual_decreased', 'topology_transition_relinearized')):
                self.accepted = event['evaluation_index']
            elif self.calls >= CAP:
                self.status = 'BUDGET_EXHAUSTED'
                raise PrefixComplete


def run(prepared, manifest, out):
    from make_it_stand import spin
    from make_it_stand.verification.cherchi_inprocess import InProcessCherchiEvaluator
    import _cherchi_core_boolean as bridge
    import _cherchi_rational_moments as arithmetic

    sys.set_int_max_str_digits(100000)
    out = out.resolve()
    out.mkdir(exist_ok=False)
    check_core()
    require(Path(spin.__file__).resolve() == ROOT/'python/make_it_stand/spin.py', 'WRONG_FROZEN_SOURCE')
    before = resource_snapshot()
    qualify(before, startup=True)
    require(file_hash(prepared) == REQUEST_SHA, 'FROZEN_REQUEST_MISMATCH')
    shutil.copyfile(prepared, out/'request.json')
    shutil.copyfile(manifest, out/'module-manifest.json')
    require(file_hash(out/'request.json') == REQUEST_SHA, 'FROZEN_REQUEST_CHANGED')
    frozen = strict_json((out/'request.json').read_bytes())
    data = dict(frozen['request'])
    request_hash = digest(data)
    data['settings'] = spin.SpinSolverConfig(**data['settings'])
    req = spin.SpinItFasterRequest(**data)
    settings = spin.SpinEvaluatorConfig(**frozen['evaluator_settings'])
    pinned = strict_json((HERE/'M23_HEART_POINT000_FRESH.json').read_bytes())['runner_replay']
    entries = validate_module_manifest(out/'module-manifest.json')
    hashes = {name:info['sha256'] for name, info in entries.items()}
    require(hashes == pinned['module_sha256'] and request_hash == pinned['request_sha256'], 'M23_IDENTITY_DRIFT')
    for name, mod in zip(MODULES, (spin._native_module(), bridge, arithmetic)):
        require(Path(mod.__file__).resolve() == Path(entries[name]['path']), 'ELF_IMPORT_PATH_CHANGED')
    geometry = InProcessCherchiEvaluator(req, settings, rational_arithmetic_backend='NATIVE_GMP_SURFACE_V1')
    native_terminal = None
    with (out/'events.jsonl').open('x') as events, (out/'evaluations.jsonl').open('x') as evaluations:
        observer = Recorder(geometry, evaluations, events)
        try:
            # Empty initial list invokes the native deterministic sequence at its own seed=0.
            solved = spin.solve_spin_it_faster_with_external_moment_evaluator(req, observer.evaluate,
                                                                             event_sink=observer.observe)
            native_terminal = str(solved['status'])
        except PrefixComplete:
            pass
    require(not observer.fatal, 'CAPTURE_INTEGRITY_FAILED')
    require(observer.status is not None or native_terminal is not None, 'UNKNOWN_CAPTURE_END')
    after = resource_snapshot()
    qualify(after, startup=False)
    require(all(before['memory_events'][k] == after['memory_events'][k] for k in ('oom','oom_kill','max')),
            'NATIVE_SCOPE_EXCEEDED')
    require(validate_module_manifest(out/'module-manifest.json') == entries and
            file_hash(out/'module-manifest.json') == file_hash(manifest), 'ELF_POSTFLIGHT_MISMATCH')
    write_json(out/'resources.json', dict(before=before, after=after))
    status = observer.status or ('FIRST_TRIAL_ACCEPTED' if observer.accepted is not None else 'NATIVE_TERMINAL')
    result = dict(schema='w2-m29-first-newton-step-v1', run_id=out.name,
                  source_checkout=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                  status=status, native_terminal=native_terminal, evaluation_count=observer.calls,
                  native_boolean_calls=geometry.call_counts.get('boolean',0), native_stage_counts=geometry.call_counts,
                  accepted_evaluation_index=observer.accepted, explicit_seeds=0, maximum_capture_evaluations=CAP,
                  prepared_request_file_sha256=REQUEST_SHA, request_sha256=request_hash, module_sha256=hashes,
                  file_sha256={p.name:file_hash(p) for p in sorted(out.iterdir()) if p.is_file()},
                  independent_step_verified=False, training_admitted=False)
    write_json(out/'result.json', result)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared-request', type=Path, required=True)
    ap.add_argument('--module-manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    print(json.dumps(run(args.prepared_request,args.module_manifest,args.output), indent=2))
