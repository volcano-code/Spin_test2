"""Rotated-only and direct-rasterized test panel invariants."""

import torch

from research.NeuralMoments.models.pcraft_ufno.spin_pino.dataset import generate_splits
from research.NeuralMoments.models.pcraft_ufno.spin_pino.rotation_sweep import rotation_panels


def test_common_panels_contain_only_the_assigned_geometry_and_rotation():
    test = generate_splits(seed=131)["test"]
    panels, occupancy_error = rotation_panels(test, seed=131)
    assert panels["original"] is test
    assert len({row["geometry_id"] for row in test[-1]}) == 8
    parent_ids = [row["geometry_id"] for row in test[-1]]
    for name in ("cube", "haar", "haar_rerasterized"):
        panel = panels[name]
        assert len(panel[0]) == len(test[0]) == 16
        assert [row["parent_geometry_id"] for row in panel[-1]] == parent_ids
        assert not set(row["geometry_id"] for row in panel[-1]).intersection(parent_ids)
    haar, direct = panels["haar"], panels["haar_rerasterized"]
    for interpolated_labels, direct_labels in zip(haar[1:4], direct[1:4]):
        assert torch.equal(interpolated_labels, direct_labels)
    assert torch.all((direct[0][:, 1] == 0) | (direct[0][:, 1] == 1))
    assert 0.0 < occupancy_error < 1.0
