"""Deterministic, geometry-family-disjoint analytic P0 data for Spin-PINO.

Labels are exact continuum integrals for axis-aligned boxes, spheres and
ellipsoids cut by strictly interior vertical planes. They are synthetic
method-development labels, never W2/Cherchi teacher observations.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict

import numpy as np
import torch

from ..analytic_cut import TEACHER_KIND, assert_disjoint_family_splits, sample_dataset


def _subset(data, indices):
    return tuple(x[indices] for x in data[:4]) + ([data[4][i] for i in indices],)


def _digest(data) -> str:
    digest = hashlib.sha256()
    for tensor in data[:4]:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(data[4], sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _group_split(data, *, seed: int, validation_fraction: float):
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0,0.5)")
    family_ids = {}
    for row in data[4]:
        family_ids.setdefault(row["family"], set()).add(row["geometry_id"])
    if not family_ids or any(len(ids) < 2 for ids in family_ids.values()):
        raise ValueError("each geometry family needs at least two distinct instances")
    rng = np.random.default_rng(seed)
    validation_ids = set()
    fit_ids = set()
    for family in sorted(family_ids):
        geometry_ids = sorted(family_ids[family])
        shuffled = list(np.asarray(geometry_ids, dtype=object)[rng.permutation(len(geometry_ids))])
        n_validation = max(1, int(round(len(shuffled) * validation_fraction)))
        n_validation = min(n_validation, len(shuffled) - 1)
        validation_ids.update(shuffled[:n_validation])
        fit_ids.update(shuffled[n_validation:])
    fit_indices = [i for i, row in enumerate(data[4]) if row["geometry_id"] in fit_ids]
    validation_indices = [i for i, row in enumerate(data[4]) if row["geometry_id"] in validation_ids]
    fit = _subset(data, fit_indices)
    validation = _subset(data, validation_indices)
    if fit_ids & validation_ids:
        raise AssertionError("geometry ID leakage between fit and validation")
    return fit, validation


def generate_splits(
    *,
    seed: int,
    train_geometries_per_family: int = 16,
    test_ellipsoid_geometries: int = 8,
    queries_per_geometry: int = 2,
    grid_size: int = 8,
    validation_fraction: float = 0.2,
) -> Dict[str, object]:
    """Train on boxes+spheres; hold out entire ellipsoid geometry family."""
    if train_geometries_per_family < 4 or test_ellipsoid_geometries < 2:
        raise ValueError("too few geometry instances for a meaningful split")
    train_pool = sample_dataset(
        ("box", "sphere"),
        per_family=train_geometries_per_family,
        seed=seed,
        grid_size=grid_size,
        queries_per_geometry=queries_per_geometry,
    )
    fit, validation = _group_split(
        train_pool, seed=seed + 104729, validation_fraction=validation_fraction
    )
    test = sample_dataset(
        ("ellipsoid",),
        per_family=test_ellipsoid_geometries,
        seed=seed + 1_000_003,
        grid_size=grid_size,
        queries_per_geometry=queries_per_geometry,
    )
    assert_disjoint_family_splits(fit, test)
    fit_ids = {row["geometry_id"] for row in fit[4]}
    validation_ids = {row["geometry_id"] for row in validation[4]}
    test_ids = {row["geometry_id"] for row in test[4]}
    if fit_ids & validation_ids or fit_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("geometry identity leakage across fit/validation/test")
    for split in (fit, validation, test):
        if any(row["source"] != TEACHER_KIND for row in split[4]):
            raise AssertionError("unexpected label source in analytic P0 dataset")
    return {
        "fit": fit,
        "validation": validation,
        "test": test,
        "metadata": {
            "schema": "SPIN_PINO_ANALYTIC_P0_DATASET_V1",
            "seed": int(seed),
            "label_source": TEACHER_KIND,
            "fit_families": ["box", "sphere"],
            "validation_families": ["box", "sphere"],
            "held_out_test_family": ["ellipsoid"],
            "queries_per_geometry": int(queries_per_geometry),
            "grid_size": int(grid_size),
            "fit_samples": len(fit[0]),
            "validation_samples": len(validation[0]),
            "test_samples": len(test[0]),
            "fit_geometry_ids": sorted(fit_ids),
            "validation_geometry_ids": sorted(validation_ids),
            "test_geometry_ids": sorted(test_ids),
            "fit_sha256": _digest(fit),
            "validation_sha256": _digest(validation),
            "test_sha256": _digest(test),
            "w2_or_cherchi_labels_used": False,
            "derivative_scope": "analytic continuum Jacobian for strictly interior vertical plane cuts only",
        },
    }
