"""Score both frozen M36 operators at every completed query of the three M37 traces."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from ..model import canonical_geometry
from .real_cgal_shadow import geometry, translation
from .refined_quadrature import cartesian
from .refinement_ablation import HERE, digest
from .residual_calibration import frozen_model

ARMS = ('exact', 'learned', 'unit')
MODES = ('learned', 'unit')
ANCHOR_SHA = '627898a26173f534bd418cc823daee21a9c4be777b962a29c9c4ce190d83f97f'
ANCHOR_MASS = {'learned': .3051402362065035, 'unit': .3431619622123074}
FIXED_SHA = '213e485af9604156e156426a1a302e8bceec4bf23bfd7cbdbc3af1844b2ca6ed'
ORIGINAL_SHA = '5912e9f1dc44378919f5c00690983d284189edadc857ea3b354f09c4b86a1986'


def read_izz(path, expected):
    if digest(path) != expected:
        raise ValueError('moment file differs from registered M32 source')
    # Use the same 190 fractional bits as W0. The tip translation is purely axial.
    result = 0
    for row in path.read_text().splitlines()[4:6]:
        a, b = (int(x, 0) for x in row.split('/'))
        result += (a << 190) // b
    return result / (1 << 190)


def summarize(rows, field='predicted_retention'):
    scores = {}
    for mode in MODES:
        confusion = Counter()
        for row in rows:
            trial_ok = row['exact_passes']
            pred_ok = row[field][mode] >= .9
            key = ('true_' if pred_ok == trial_ok else 'false_') + ('pass' if pred_ok else 'reject')
            confusion[key] += 1
        scores[mode] = {k: confusion[k] for k in ('true_pass', 'true_reject', 'false_pass', 'false_reject')}
    return scores


def anchor_adjust(anchor_actual, anchor_predictions, candidate_predictions):
    return {mode: anchor_actual + (candidate_predictions[mode] - anchor_predictions[mode]) for mode in MODES}


def run(traces, fixed_ten, original_ten):
    fixed, original = read_izz(fixed_ten, FIXED_SHA), read_izz(original_ten, ORIGINAL_SHA)
    model, _, fit = frozen_model(131)
    vox = json.loads((HERE / 'M35_REAL_INPUT.json').read_bytes())['voxels']
    X = geometry(vox)
    S, T = translation(vox['cube_center_in_tip'])
    with torch.inference_mode():
        gates = model.weight_head(model.encoder(canonical_geometry(X))).sigmoid().reshape(6, 6, 6).numpy()
    rows = {}
    reports = {}
    for arm in ARMS:
        path = traces / arm / 'report.json'
        trace = json.loads(path.read_bytes())
        if (trace['arm'] != arm or not trace['terminal'] or trace['protocol']['inertia_floor'] != .9
                or trace['protocol']['hashes']['design'] != vox['design_sha256']
                or trace['protocol']['hashes']['fixed_ten'] != FIXED_SHA
                or trace['protocol']['hashes']['original_ten'] != ORIGINAL_SHA):
            raise ValueError('unexpected trajectory protocol')
        reports[arm] = digest(path)
        for i, sample in enumerate(trace['evaluations'], 1):
            key = tuple(sample['q'])
            gate = sample['inertia_gate']
            passed = gate['passes']
            var = sample['variable_ten_tip']
            total = sample['retained_ten_tip']
            truth = (fixed + var[4] + var[5]) / original
            truth_saved = (total[4] + total[5]) / original
            if (abs(truth-truth_saved) > 2e-14 or not gate['lower']-2e-14 <= truth <= gate['upper']+2e-14
                    or passed != (gate['lower'] >= .9) or (not passed and gate['upper'] >= .9)
                    or i != sample['index'] or (i == 1) != (sample['result_sha256'] == ANCHOR_SHA)):
                raise ValueError('inconsistent query and fixed-axis certificate')
            if key in rows:
                row = rows[key]
                if (row['result_sha256'] != sample['result_sha256'] or row['exact_passes'] != passed
                        or not np.isclose(row['actual_retention'], truth, rtol=0., atol=2e-14)):
                    raise ValueError('inconsistent duplicate coefficient query')
                row['trajectory_calls'].append([arm, i])
                continue
            pred = {}
            for mode, weights in (('learned', gates), ('unit', np.ones_like(gates))):
                r, _ = cartesian(X, weights, T @ key, 24, .0075)
                variable = S @ r
                if i == 1 and not np.isclose(variable[0], ANCHOR_MASS[mode], rtol=1e-6, atol=1e-10):
                    raise ValueError('M36 anchor/coordinate adapter mismatch')
                pred[mode] = (fixed + variable[4] + variable[5]) / original
            rows[key] = dict(q=key, trajectory_calls=[[arm, i]], result_sha256=sample['result_sha256'],
                             actual_retention=truth, exact_passes=passed, predicted_retention=pred)
    anchor, = (row for row in rows.values() if row['result_sha256'] == ANCHOR_SHA)
    for row in rows.values():
        row['anchored_retention'] = anchor_adjust(
            anchor['actual_retention'], anchor['predicted_retention'], row['predicted_retention'])
    trials = [r for r in rows.values() if r['result_sha256'] != ANCHOR_SHA]
    return dict(seed=131, selected_step=fit['selected_step'], rule='fine_cool', threshold=.9, fixed_izz=fixed,
                original_izz=original, reports_sha256=reports, unique_trials=len(trials),
                confusion_unique_trials_excluding_anchor=summarize(trials),
                post_hoc_anchored_confusion=summarize(trials, 'anchored_retention'), unique_rows=list(rows.values()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('traces', 'fixed-ten', 'original-ten', 'output'):
        p.add_argument('--' + arg, type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(run(args.traces, args.fixed_ten, args.original_ten), indent=2, allow_nan=False) + '\n')
