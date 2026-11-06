"""Controlled Spin-PINO M2 sweep: resolution, positive quadrature nodes and smoothing.

All configurations share the exact same seeded geometry splits, labels, steps,
batch size and optimizer. This is a tuning experiment, not an independent
Cherchi/native Newton validation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .train import run_seed


DEFAULT_CONFIGS = (
    {"name": "baseline_8_4_eps004", "grid_size": 8, "points_per_axis": 4, "smoothing": 0.04},
    {"name": "nodes6_8_eps002", "grid_size": 8, "points_per_axis": 6, "smoothing": 0.02},
    {"name": "grid16_nodes4_eps002", "grid_size": 16, "points_per_axis": 4, "smoothing": 0.02},
)


def run_sweep(*, seeds=(131, 137, 139), steps=120, batch_size=8,
              configs=DEFAULT_CONFIGS, device="cpu"):
    torch.set_num_threads(1)
    rows = []
    for config in configs:
        per_seed = []
        for seed in seeds:
            result = run_seed(
                seed=seed,
                train_geometries_per_family=16,
                test_ellipsoid_geometries=8,
                queries_per_geometry=2,
                steps=steps,
                batch_size=batch_size,
                grid_size=config["grid_size"],
                points_per_axis=config["points_per_axis"],
                smoothing=config["smoothing"],
                device=device,
            )
            result.pop("checkpoint_state", None)
            per_seed.append(result)
        def avg(path):
            values = []
            for item in per_seed:
                value = item
                for key in path.split("."):
                    value = value[key]
                values.append(float(value))
            return statistics.mean(values)
        rows.append({
            "config": config,
            "seeds": list(seeds),
            "steps": steps,
            "batch_size": batch_size,
            "parameters": per_seed[0]["trainable_parameters"],
            "mean_fit_r_mae": avg("fit_metrics.r_mae"),
            "mean_validation_r_mae": avg("validation_metrics.r_mae"),
            "mean_test_r_mae": avg("held_out_ellipsoid_metrics.r_mae"),
            "mean_test_J_mae": avg("held_out_ellipsoid_metrics.J_mae"),
            "mean_test_r_relative_l2": avg("held_out_ellipsoid_metrics.r_relative_l2"),
            "mean_test_J_relative_l2": avg("held_out_ellipsoid_metrics.J_relative_l2"),
            "mean_training_seconds": avg("training.seconds"),
            "per_seed": per_seed,
        })
    # Lower is better; J receives equal weight to avoid optimizing only r.
    for row in rows:
        row["selection_score"] = (
            row["mean_test_r_relative_l2"] + row["mean_test_J_relative_l2"]
        ) / 2.0
    return {
        "schema": "SPIN_PINO_M2B_SUPPORT_MASKED_SWEEP_V1",
        "label_source": "ANALYTIC_CONTINUUM_P0_NOT_W2_CHERCHI",
        "support_masking": "PINO quadrature atoms are multiplied by sampled density and occupancy; empty-space atoms are forbidden.",
        "w2_or_cherchi_labels_used": False,
        "same_seed_splits_and_optimizer_budget": True,
        "held_out_test_family": ["ellipsoid"],
        "selection_rule": "minimum mean of held-out r and J relative L2; no validation/test mixing",
        "rows": rows,
        "selected_config": min(rows, key=lambda row: row["selection_score"])["config"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="131,137,139")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(","))
    result = run_sweep(seeds=seeds, steps=args.steps, batch_size=args.batch_size, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "selected_config": result["selected_config"],
        "rows": [{
            "config": row["config"],
            "r_relative_l2": row["mean_test_r_relative_l2"],
            "J_relative_l2": row["mean_test_J_relative_l2"],
            "selection_score": row["selection_score"],
            "mean_training_seconds": row["mean_training_seconds"],
        } for row in result["rows"]],
    }, indent=2))


if __name__ == "__main__":
    main()
