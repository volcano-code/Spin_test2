"""Paired orientation evaluation on separate, geometry-family-held-out test panels."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from ..analytic_cut import GeometrySpec, voxel_geometry
from .dataset import _digest, generate_splits
from .equivariance import augment_with_cube_rotations, augment_with_random_so3
from .tournament import _build
from .train import _evaluate, run_seed


def rotation_panels(test, seed):
    """Evaluate the same transformed queries for every model in a given seed."""
    count = len(test[0])
    rotation_seed = seed + 900001
    cube = tuple(col[count:] for col in augment_with_cube_rotations(test, seed=rotation_seed))
    haar = tuple(col[count:] for col in augment_with_random_so3(test, seed=rotation_seed))
    size = test[0].shape[-1]
    axis = np.linspace(-1, 1, size, dtype=np.float64)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)

    # The pinned analytic P0 source exports IDs but omits primitive parameters.
    # Replay its held-out geometry draws, verifying each ID and original field.
    rng = np.random.default_rng(seed + 1000003)
    specs = {}
    direct_fields = []
    for index, row in enumerate(test[-1]):
        identity = row["geometry_id"]
        if identity not in specs:
            center = tuple(map(float, rng.uniform(-0.1, 0.1, 3)))
            a, b, c = rng.uniform(0.46, 0.73, 3)
            spec = GeometrySpec("ellipsoid", center, (float(a), float(b)*0.8, float(c)*1.1))
            if spec.geometry_id != identity or not torch.equal(voxel_geometry(spec, size), test[0][index]):
                raise ValueError("held-out ellipsoid does not match the registered analytic source")
            specs[identity] = spec
        rng.uniform(-0.65, 0.65)  # original query height
        spec = specs[identity]
        rotation = np.asarray(haar[-1][index]["rotation_matrix"])
        local = (grid @ rotation - spec.center) / spec.axes
        occupied = np.square(local).sum(axis=-1) <= 1
        empty = np.zeros_like(occupied)
        direct_fields.append(np.stack((spec.density*occupied, occupied, empty, empty)))
    direct = torch.from_numpy(np.asarray(direct_fields)).to(test[0])
    panels = {"original": test, "cube": cube, "haar": haar,
              "haar_rerasterized": (direct, *haar[1:])}
    occupancy_error = (haar[0][:, 1] - direct[:, 1]).norm() / direct[:, 1].norm()
    return panels, float(occupancy_error)


def run(*, seeds=(131, 137, 139), steps=120, batch_size=8):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected distinct, nonempty seeds")
    per_seed = {}
    for seed in seeds:
        splits = generate_splits(seed=seed)
        panels, interpolation_error = rotation_panels(splits["test"], seed)
        results = {}
        for condition, augmentation, mode in (("none", False, "cube"), ("cube", True, "cube"),
                                              ("haar", True, "so3")):
            trained = run_seed(
                seed=seed, steps=steps, batch_size=batch_size, grid_size=8, points_per_axis=6,
                smoothing=0.02, temperature=0.12, moment_parameterization="logistic_atom",
                rotation_augmentation=augmentation, rotation_augmentation_mode=mode,
            )
            if trained["dataset"] != splits["metadata"]:
                raise RuntimeError("training and test panel dataset identity mismatch")
            model = _build("spin_pino_6node_logistic")
            model.load_state_dict(trained["checkpoint_state"])
            results[condition] = {name: _evaluate(model, data) for name, data in panels.items()}
        per_seed[seed] = {
            "paired_sha256": {name: _digest(data) for name, data in panels.items()},
            "samples_per_panel": len(panels["original"][0]), "held_out_geometries": len(splits["metadata"]["test_geometry_ids"]),
            "interpolated_occupancy_relative_l2": interpolation_error, "results": results,
        }
    means = {
        cond: {panel: {metric: statistics.mean(s["results"][cond][panel][metric]
                                              for s in per_seed.values())
                       for metric in ("r_relative_l2", "J_relative_l2")}
               for panel in panels}
        for cond in results
    }
    return {
        "schema": "SPIN_PINO_PAIRED_ROTATIONS_V2", "label_source": splits["metadata"]["label_source"],
        "steps": steps, "batch_size": batch_size, "torch_version": str(torch.__version__),
        "w2_or_cherchi_labels_used": False, "newton_admitted": False, "means": means,
        "per_seed": per_seed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="131,137,139")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(seeds=tuple(map(int, args.seeds.split(","))), steps=args.steps, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result["means"], indent=2))
