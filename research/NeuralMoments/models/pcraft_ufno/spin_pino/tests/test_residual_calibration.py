import numpy as np
import pytest
from research.NeuralMoments.models.pcraft_ufno.spin_pino.residual_calibration import fit_coefficient


@pytest.mark.parametrize('target, expected', [(0.5, 1.), (1., 0.), (1.1, -.2), (2., -.5)])
def test_signed_bounded_solution(target, expected):
    block = np.array([1.]), np.array([.5]), np.array([target])
    assert fit_coefficient([block]) == pytest.approx(expected)


def test_equal_relative_weighting_and_neutral_tie():
    u, l = np.array([1.]), np.array([.5])
    one = (u, l, np.array([.8]))
    two = (u, l, np.array([1.1]))
    assert fit_coefficient([one, two]) == pytest.approx(fit_coefficient([one, tuple(500 * x for x in two)]))
    assert fit_coefficient([(u, u, l)]) == 0.
