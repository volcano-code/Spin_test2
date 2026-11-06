"""Replay the strict initial multiplier on the pinned M23 exploratory sphere."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np

from probe_h8_native_cherchi import initial_ellipsoid_q

BASE_ID = 'heart-point000-42177937306-1'
CAPTURE_SHA = '816fc58eda9b20a65efccff00b3b803b2870315d9422957d1d141dbe6f68da04'
REQUEST_SHA = '7084fecb6bbd4d17bca842bfe68299f9711ecead09eb01593364424a3e0c4a8a'


def pinned_json(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f'{path.name} digest mismatch')
    return json.loads(raw)


def audit(directory, probe):
    cap = pinned_json(directory/'capture.json', CAPTURE_SHA)
    if cap['run_id'] != BASE_ID or cap['request_sha256'] != REQUEST_SHA:
        raise ValueError('unexpected M23 run')
    spec, mat, label = (pinned_json(directory/name, cap['files'][name])
                        for name in ('spec.json','material.json','label.json'))
    req = spec['request']
    req_digest = sha256(json.dumps(req, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    if (req_digest != REQUEST_SHA or req['mode'] != 'SPINNING_TOP' or
            req['spinning_top_objective'] != 'INERTIA_AND_LOWERING' or req['center_of_mass_weight'] != 0
            or req['settings']['equation_profile'] != 'DUAL_MONOMIAL_RAW_V1'):
        raise ValueError('wrong frozen optimization problem')
    p = label['payload']
    q = np.array(mat['quadric_coefficients'], dtype=np.float64)
    if not (q.tobytes() == np.asarray(spec['native_c']).tobytes() == np.asarray(p['native_c']).tobytes()
            == initial_ellipsoid_q(req['target_origin']).tobytes()):
        raise ValueError('wrong exploratory sphere')
    moments = mat['raw_moments']
    if (mat['variable_raw_moments'] != p['r_D'] or p['frame']['kind'] != 'INPUT_NORMALIZED_TARGET_V1'
            or moments != [d+f for d, f in zip(p['r_D'],p['r_F'], strict=True)] or len(moments) != 10):
        raise ValueError('wrong target-frame total moments')
    data = ' '.join(map(str, [*q.tolist(), *moments]))
    output = subprocess.run([str(probe.resolve())], input=data, capture_output=True, text=True, check=True)
    values = list(map(float, output.stdout.split()))
    if len(values) != 7 or not np.isfinite(values).all():
        raise ValueError('invalid preflight result')
    mu, residual, objective, *constraints = values
    return dict(schema='w3-heart-point000-strict-init-preflight-v1', base_run_id=BASE_ID,
                capture_sha256=CAPTURE_SHA, request_sha256=REQUEST_SHA, capture_commit=cap['source_checkout'],
                objective_selector=req['spinning_top_objective'], center_of_mass_weight=req['center_of_mass_weight'],
                spin2014_objective_override=None, objective_value=objective, constraint_order=['x','y','I13','I23'],
                constraints=constraints, multiplier_mu=mu, raw_equation_residual=residual,
                strict_positive_initial_multiplier=mu > 0, expected_initial_reason=(
                    'positive_initial_multiplier' if mu > 0 else 'initial_multiplier_nonpositive'),
                input_type='exploratory_sphere_as_hypothetical_initial_guess',
                native_calls_in_preflight=0, newton_trial_evaluations=0, accepted_newton_steps=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('probe', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.base, args.probe), indent=2, allow_nan=False))
