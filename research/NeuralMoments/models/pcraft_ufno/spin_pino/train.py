"""Reproducible CPU-first Spin-PINO training on analytic continuum P0 labels."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import generate_splits
from .spin_pino import SpinPINO, SpinPINOLoss
from .loss_conditioning import DiagonalMomentResidualPreconditioner, pcgrad_backward
from .equivariance import augment_with_cube_rotations, augment_with_random_so3


def _evaluate(model, data):
    D, c, target_r, target_J, _ = data
    model.eval()
    with torch.no_grad():
        z = model.encode_geometry(D)
        pred_r = model.moment_from_weights(z, c)
        pred_J = model.jacobian_from_weights(z, c)
    r_abs = (pred_r - target_r).abs()
    J_abs = (pred_J - target_J).abs()
    return {
        "r_mae": float(r_abs.mean()),
        "r_max_component_mae": [float(x) for x in r_abs.mean(dim=0)],
        "r_relative_l2": float((pred_r - target_r).norm() / target_r.norm().clamp_min(1e-12)),
        "J_mae": float(J_abs.mean()),
        "J_relative_l2": float((pred_J - target_J).norm() / target_J.norm().clamp_min(1e-12)),
        "negative_mass_fraction": float((pred_r[:, 0] < -1e-7).float().mean()),
    }


def run_seed(
    *,
    seed: int,
    train_geometries_per_family: int = 16,
    test_ellipsoid_geometries: int = 8,
    queries_per_geometry: int = 2,
    steps: int = 120,
    batch_size: int = 8,
    grid_size: int = 8,
    points_per_axis: int = 4,
    smoothing: float = 0.04,
    preconditioned_loss: bool = False,
    pcgrad: bool = False,
    moment_parameterization: str = "smooth_hinge",
    temperature: float = 0.12,
    rotation_augmentation: bool = False,
    rotation_augmentation_mode: str = "cube",
    device: str = "cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    splits = generate_splits(
        seed=seed,
        train_geometries_per_family=train_geometries_per_family,
        test_ellipsoid_geometries=test_ellipsoid_geometries,
        queries_per_geometry=queries_per_geometry,
        grid_size=grid_size,
    )
    fit, validation, test = (splits[k] for k in ("fit", "validation", "test"))
    if rotation_augmentation:
        if rotation_augmentation_mode == "cube":
            fit = augment_with_cube_rotations(fit, seed=seed + 700001)
        elif rotation_augmentation_mode == "so3":
            fit = augment_with_random_so3(fit, seed=seed + 700001)
        else:
            raise ValueError("rotation_augmentation_mode must be cube or so3")
    model = SpinPINO(
        width=4 if grid_size <= 8 else 8,
        modes=2 if grid_size <= 8 else 4,
        n_layers=2 if grid_size <= 8 else 4,
        grid_size=grid_size,
        points_per_axis=points_per_axis,
        hidden_dim=48,
        smoothing=smoothing,
        moment_parameterization=moment_parameterization,
        temperature=temperature,
    ).to(device)
    preconditioner = (
        DiagonalMomentResidualPreconditioner.fit(fit[2], fit[3]).to(device=device, dtype=fit[2].dtype)
        if preconditioned_loss else None
    )
    criterion = SpinPINOLoss(
        lambda_data=1.0, lambda_physics=0.05, lambda_jacobian=0.25,
        moment_scale=preconditioner.moment_scale if preconditioner else None,
        jacobian_scale=preconditioner.jacobian_scale if preconditioner else None,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-3)
    generator = torch.Generator().manual_seed(seed + 31337)
    best_score = math.inf
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        indices = torch.randint(
            len(fit[0]), (min(batch_size, len(fit[0])),), generator=generator
        )
        D, c, target_r, target_J = (
            x[indices].to(device) for x in fit[:4]
        )
        weights = model.encode_geometry(D)
        # These Jacobians are exact analytic labels for the restricted P0
        # vertical-plane family, not diagnostic W2 records.
        losses = criterion(
            model,
            weights,
            c,
            target_moments=target_r,
            target_jacobian=target_J,
            derivative_admitted=True,
        )
        if not bool(torch.isfinite(losses["total"])):
            raise FloatingPointError(f"nonfinite Spin-PINO loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        grad_stats = None
        if pcgrad:
            grad_stats = pcgrad_backward(
                (losses["data"], losses["physics"], losses["jacobian_supervision"]),
                model.parameters(),
                (criterion.lambda_data, criterion.lambda_physics, criterion.lambda_jacobian),
            )
        else:
            losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == steps:
            val = _evaluate(model, validation)
            score = val["r_mae"] + 0.25 * val["J_mae"]
            history.append({
                "step": step,
                "train_total": float(losses["total"].detach()),
                "train_data": float(losses["data"].detach()),
                "train_physics": float(losses["physics"].detach()),
                "train_jacobian": float(losses["jacobian_supervision"].detach()),
                "gradient_conflict": grad_stats,
                "validation_score": score,
                "validation": val,
            })
            if score < best_score:
                best_score = score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    fit_metrics = _evaluate(model, fit)
    validation_metrics = _evaluate(model, validation)
    test_metrics = _evaluate(model, test)
    return {
        "schema": "SPIN_PINO_ANALYTIC_P0_TRAINING_V1",
        "seed": seed,
        "label_source": splits["metadata"]["label_source"],
        "w2_or_cherchi_labels_used": False,
        "derivative_scope": splits["metadata"]["derivative_scope"],
        "dataset": splits["metadata"],
        "trainable_parameters": model.real_parameter_count(),
        "loss_conditioning": {
            "diagonal_preconditioner": bool(preconditioned_loss),
            "pcgrad": bool(pcgrad),
            "preconditioner_fit_split_only": True,
            "preconditioner": preconditioner.as_dict() if preconditioner else None,
            "rotation_augmentation": bool(rotation_augmentation),
            "rotation_augmentation_mode": rotation_augmentation_mode,
            "fit_samples_after_augmentation": int(len(fit[0])),
        },
        "model_config": {"grid_size": grid_size, "points_per_axis": points_per_axis,
                         "smoothing": smoothing, "moment_parameterization": moment_parameterization,
                         "temperature": temperature, "width": 4 if grid_size <= 8 else 8,
                         "modes": 2 if grid_size <= 8 else 4},
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "selected_step": best_step,
            "seconds": training_seconds,
            "optimizer": "Adam",
            "learning_rate": 1.5e-3,
            "history": history,
        },
        "fit_metrics": fit_metrics,
        "validation_metrics": validation_metrics,
        "held_out_ellipsoid_metrics": test_metrics,
        "checkpoint_state": best_state,
        "claims": {
            "trained_spin_model": True,
            "cherchi_supervised": False,
            "arbitrary_quadric_certified": False,
            "newton_admitted": False,
            "real_spin_end_to_end_validated": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="131,137,139")
    parser.add_argument("--train-geometries-per-family", type=int, default=16)
    parser.add_argument("--test-ellipsoid-geometries", type=int, default=8)
    parser.add_argument("--queries-per-geometry", type=int, default=2)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--points-per-axis", type=int, default=4)
    parser.add_argument("--smoothing", type=float, default=0.04)
    parser.add_argument("--preconditioned-loss", action="store_true")
    parser.add_argument("--pcgrad", action="store_true")
    parser.add_argument("--moment-parameterization", choices=("smooth_hinge", "logistic_atom"), default="smooth_hinge")
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--rotation-augmentation", action="store_true")
    parser.add_argument("--rotation-augmentation-mode", choices=("cube", "so3"), default="cube")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()
    results = []
    for seed in (int(x) for x in args.seeds.split(",")):
        result = run_seed(
            seed=seed,
            train_geometries_per_family=args.train_geometries_per_family,
            test_ellipsoid_geometries=args.test_ellipsoid_geometries,
            queries_per_geometry=args.queries_per_geometry,
            steps=args.steps,
            batch_size=args.batch_size,
            grid_size=args.grid_size,
            points_per_axis=args.points_per_axis,
            smoothing=args.smoothing,
            preconditioned_loss=args.preconditioned_loss,
            pcgrad=args.pcgrad,
            moment_parameterization=args.moment_parameterization,
            temperature=args.temperature,
            rotation_augmentation=args.rotation_augmentation,
            rotation_augmentation_mode=args.rotation_augmentation_mode,
            device=args.device,
        )
        state = result.pop("checkpoint_state")
        if args.save_checkpoints:
            checkpoint_path = args.output.with_name(f"{args.output.stem}_seed{seed}.pt")
            torch.save(state, checkpoint_path)
            result["checkpoint_path"] = str(checkpoint_path)
        results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps([
        {"seed": row["seed"], "selected_step": row["training"]["selected_step"],
         "fit_r_mae": row["fit_metrics"]["r_mae"],
         "validation_r_mae": row["validation_metrics"]["r_mae"],
         "held_out_ellipsoid_r_mae": row["held_out_ellipsoid_metrics"]["r_mae"],
         "held_out_ellipsoid_J_mae": row["held_out_ellipsoid_metrics"]["J_mae"]}
        for row in results
    ], indent=2))


if __name__ == "__main__":
    main()
