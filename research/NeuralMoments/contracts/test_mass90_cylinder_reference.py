from fractions import Fraction as Q

import numpy as np
import pytest

from .mass90_cylinder_reference import centered_case, references
from .moment_coordinates import RAW_R_ORDER, raw_to_tensors


@pytest.mark.parametrize('name, mass, ratio, multiplier', [
    ('active', Q(3, 4), Q(9, 10), Q(5, 2)),
    ('slack', Q(27, 32), Q(123, 128), Q(0)),
])
def test_primal_mass_matches_dual_and_inertia(name, mass, ratio, multiplier):
    case = references()[name]
    assert case['retained_mass_over_pi'] == case['retained_volume_over_pi'] == mass
    assert case['dual_mass_lower_over_pi'] == mass
    assert case['reference_inertia_over_pi'] == Q(1, 2)
    assert case['inertia_ratio'] == ratio
    assert case['multiplier'] == multiplier
    assert multiplier * case['inertia_slack_over_pi'] == 0
    raw = [float(case['total_raw_over_pi'][k]) for k in RAW_R_ORDER]
    _, first, second = raw_to_tensors(raw)
    np.testing.assert_array_equal(first[:2], [0, 0])
    np.testing.assert_array_equal(second[:2, 2], [0, 0])
    assert second[0, 0] + second[1, 1] == pytest.approx(float(ratio/2))


def test_off_axis_capacity_fails_balance_at_same_threshold():
    result = references()
    case = result['infeasible']
    assert result['inertia_fraction'] == Q(9, 10)
    assert case['reference_inertia_over_pi'] == Q(9, 16)
    assert case['first_x_about_axis_upper_over_pi'] == Q(-17, 128)
    assert case['balance_excluded']


@pytest.mark.parametrize('cut_squared', [Q(3, 10), Q(1, 2)])
def test_nonoptimal_or_insufficient_inertia_fails_certificate(cut_squared):
    with pytest.raises(ValueError, match='primal'):
        centered_case(Q(4, 5), cut_squared, Q(5, 2))
