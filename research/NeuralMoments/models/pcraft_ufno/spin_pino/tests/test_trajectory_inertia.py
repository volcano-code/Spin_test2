import json

from research.NeuralMoments.models.pcraft_ufno.spin_pino.trajectory_inertia import HERE, anchor_adjust, summarize


def test_fixed_threshold_confusion_independent_per_mode():
    rows = [
        dict(exact_passes=True, predicted_retention=dict(learned=.90, unit=.89)),
        dict(exact_passes=False, predicted_retention=dict(learned=.91, unit=.89)),
    ]
    assert summarize(rows) == {
        'learned': dict(true_pass=1, true_reject=0, false_pass=1, false_reject=0),
        'unit': dict(true_pass=0, true_reject=1, false_pass=0, false_reject=1),
    }


def test_frozen_trajectory_accounting():
    saved = json.loads((HERE / 'M37_TRAJECTORY_INERTIA.json').read_bytes())
    rows = saved['unique_rows']
    indexed = [(arm, i) for row in rows for arm, i in row['trajectory_calls']]
    assert len(set(indexed)) == len(indexed) == 14
    assert len(set(tuple(row['q']) for row in rows)) == len(rows)
    anchor, *trials = rows
    assert anchor['trajectory_calls'] == [['exact', 1], ['learned', 1], ['unit', 1]]
    assert len(trials) == saved['unique_trials'] == 11
    assert saved['confusion_unique_trials_excluding_anchor'] == summarize(trials)
    assert saved['post_hoc_anchored_confusion'] == summarize(trials, 'anchored_retention')


def test_anchor_adjust_only_needs_one_exact_measurement():
    anchor_predictions = dict(learned=.6, unit=.7)
    assert anchor_adjust(.98, anchor_predictions, anchor_predictions) == dict(learned=.98, unit=.98)
    moved = anchor_adjust(.98, anchor_predictions, dict(learned=.5, unit=.8))
    assert abs(moved['learned']-.88) < 1e-12
    assert abs(moved['unit']-1.08) < 1e-12
