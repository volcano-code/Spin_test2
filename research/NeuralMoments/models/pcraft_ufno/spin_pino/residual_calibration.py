"""Fit a single signed analytic correction to the frozen M36 positive quadrature."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..model import canonical_geometry
from .dataset import generate_splits
from .half_ellipsoid import panel
from .real_cgal_shadow import relative
from .refined_quadrature import localized
from .refinement_ablation import HERE, digest, real_cases
from .support_ablation import PINO
from .tournament import _build, _fit


def fit_coefficient(triples):
    """Minimize the equal-block sum of squared relative errors along unit -> learned."""
    numerator = denominator = 0.
    for unit, learned, target in triples:
        delta = learned - unit
        norm2 = np.vdot(target, target)
        if norm2 <= 0:
            raise ValueError('relative loss needs a nonzero target')
        numerator += np.vdot(delta, target - unit) / norm2
        denominator += np.vdot(delta, delta) / norm2
    return float(np.clip(numerator / denominator, -.5, 1.)) if denominator else 0.


def analytic(cases):
    eye = np.eye(10)
    return [('analytic', name, X, q, r, J, eye, eye, key) for name, X, q, r, J, key in cases]


def run():
    torch.set_num_threads(1)
    seed = 131
    data = generate_splits(seed=seed)
    torch.manual_seed(seed)
    model = _build(PINO)
    model.density_is_premultiplied = True
    fit = _fit(model, PINO, data['fit'], data['validation'], seed=seed, physics_weight=.05)
    reference = json.loads((HERE / 'M35_REAL_SHADOW.json').read_bytes())
    expected = next(row for row in reference['models'] if row['seed'] == seed)
    if fit['selected_step'] != expected['validation_selected_step']:
        raise ValueError('the frozen checkpoint selection changed')
    model.eval()

    def gates_for(X):
        with torch.inference_mode():
            return model.weight_head(model.encoder(canonical_geometry(X))).sigmoid().reshape(6, 6, 6).double().numpy()

    cal, test = analytic(panel(3701, 32)), analytic(panel())
    original = {row['geometry_id'] for part in ('fit', 'validation', 'test') for row in data[part][4]}
    cal_ids = {case[-1] for case in cal}
    test_ids = {case[-1] for case in test}
    if len(cal_ids) != len(cal) or cal_ids & test_ids or (cal_ids | test_ids) & original:
        raise ValueError('overlapping geometry identifiers')
    blocks = []
    for _, _, X, q, r, J, _, _, _ in cal:
        gates = gates_for(X)
        unit = localized(X, np.ones_like(gates), q)
        learned = localized(X, gates, q)
        blocks.extend(zip(unit[:2], learned[:2], (r, J)))
    coeff = fit_coefficient(blocks)

    files = {'Noodles': HERE / 'M35_REAL_INPUT.json', 'Bunny': HERE / 'M36_BUNNY_INPUT.json'}
    for name, path in files.items():
        test.extend(real_cases(path, name))
    records = []
    for group, name, X, q, r, J, S, T, key in test:
        gates = gates_for(X)
        scores = {}
        for arm, c in [('unit', 0.), ('frozen_m36', 1.), ('calibrated', coeff)]:
            pred, jac, rule = localized(X, 1. + c * (gates - 1.), T @ q)
            scores[arm] = [relative(S @ pred, r), relative(S @ jac @ T, J)]
        records.append(dict(group=group, name=name, source=key, rule=rule, relative_r_J=scores))
    means = {group: {arm: np.mean([row['relative_r_J'][arm] for row in records if row['group'] == group],
                                  axis=0).tolist() for arm in scores} for group in ('analytic', *files)}
    return dict(protocol_sha256=digest(HERE / 'M37_CALIBRATION_PROTOCOL.json'), pytorch=torch.__version__,
                seed=seed, selected_step=fit['selected_step'], coefficient=coeff, calibration_geometry_ids=sorted(cal_ids),
                real_input_sha256={k: digest(v) for k, v in files.items()}, real_training_queries=0,
                mean_relative_r_J=means, rows=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(), indent=2, allow_nan=False) + '\n')
