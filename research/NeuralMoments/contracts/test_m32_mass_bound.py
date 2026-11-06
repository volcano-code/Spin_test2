from fractions import Fraction as Q

import pytest

from .m32_mass_bound import enclose, removed_volume_upper


def test_outward_rounding_both_signs():
    assert enclose("0x1/0x3",3) == (Q(333,1000),Q(334,1000))
    assert enclose("-0x1/0x3",3) == (-Q(334,1000),-Q(333,1000))
    assert enclose("0x1/0x4",3) == (Q(1,4),Q(1,4))


def test_cylindrical_extremizer_conservatively_enclosed():
    # Cylinder r=H=1. Using pi<22/7 both here and in the slab formula.
    assert removed_volume_upper(Q(1), Q(11,7), Q(1)) == Q(22,7)
    with pytest.raises(ValueError):
        removed_volume_upper(1, -1, Q(5))
