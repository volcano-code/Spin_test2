import numpy as np
import pytest

from .m37_continuation import prefix
from .m37_reintegrate import flux_ten


def test_three_axis_flux_matches_analytic_tetrahedron():
    vertices = np.array(((0,0,0), (1,0,0), (0,1,0), (0,0,1)), dtype=float)
    faces = np.array(((1,2,3), (0,2,1), (0,1,3), (0,3,2)))
    expected = np.array([1/6, *[1/24]*3, *[1/60]*3, *[1/120]*3])
    for orientation in (1, -1):
        actual = flux_ten(vertices, faces[:,::orientation])
        np.testing.assert_allclose(actual, np.tile(orientation * expected, (3,1)), atol=1e-16)


@pytest.mark.parametrize("arm", ("exact", "learned", "unit"))
def test_prefix_rebuilds_registered_first_accepted_step(arm):
    samples, model = prefix(arm)
    assert samples[0][0] == "anchor"
    assert samples[-1][1]["inertia_gate"]["passes"]
    if arm == "exact":
        assert model is None and len(samples) == 2
    else:
        assert len(samples) == 3 and model["anchor_q"] == samples[0][1]["q"]
        assert not samples[1][1]["inertia_gate"]["passes"]
