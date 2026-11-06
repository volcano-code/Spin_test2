import pytest

from .heart_point0_solid import reconcile


def test_assignments_cover_two_operands_and_empty_parents():
    assert reconcile([[0, "cut", [1]], [1, "removed", []]], [[0, "retained", [0]]],
                     [[1, 0], [0, 0]], 2, 1) == 2


@pytest.mark.parametrize(("design", "cutter", "face_map", "reason"), [
    ([], [], [], "missing input parent"),
    ([[1, "removed", []]], [], [], "wrong input parent order"),
    ([[0, "removed", [0]]], [], [[0, 0]], "state conflicts"),
    ([[0, "retained", [0, 0]]], [], [[0, 0]], "duplicate output"),
    ([[0, "retained", [0]]], [], [[1, 0]], "disagree on parent"),
    ([[0, "retained", [0]]], [], [[0, 0], [0, 0]], "missing output"),
])
def test_rejects_incomplete_or_wrong_evidence(design, cutter, face_map, reason):
    with pytest.raises(ValueError, match=reason):
        reconcile(design, cutter, face_map, 1, 0)
