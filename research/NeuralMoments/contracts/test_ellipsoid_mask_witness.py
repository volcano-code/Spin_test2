from pathlib import Path

import pytest

from .ellipsoid_mask_witness import mesh_mask, verify


ROOT = Path(__file__).resolve().parents[3] / 'experiments/spin_it_reproduction'


def test_archived_ellipsoid_mask_witness():
    observed = verify(ROOT)
    assert observed['total_material_cells'] == 272
    assert observed['total_material_raw_moments'] == [
        '17/32','0','0','17/32','277/3072','223/3072','1759/3072','0','0','0']
    assert observed['exact_constraints'] == ['0'] * 4


def test_inward_cavity_mesh_cannot_be_read_as_outward_material():
    path = ROOT / 'viewer/spin2014/ellipsoid-top/repeat-1/cavity.obj'
    with pytest.raises(ValueError, match='orientation'):
        mesh_mask(path, 1)
