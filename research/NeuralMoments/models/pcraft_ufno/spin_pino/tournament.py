"""M3 same-data tournament: PINO vs W6 M7 U-FNO vs W5 voxel DeepONet.

All models see exactly the same four-channel geometry tensors, coefficient
vectors, analytic r/J labels, geometry-family split, seeds and optimizer-step
budget. The label source is the restricted analytic vertical-plane P0 task,
not Cherchi. This isolates architecture effects from the mesh-source M13 study.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .dataset import generate_splits
from .spin_pino import SpinPINO, SpinPINOLoss
from ..m7_realizable import RealizableUFNOMoment
from ..m10_w5_readonly import W5VoxelDeepONet


def _build(kind: str):
    if kind == "spin_pino_4node":
        return SpinPINO(width=4, modes=2, n_layers=2, grid_size=8,
                        points_per_axis=4, hidden_dim=48, smoothing=0.04)
    if kind == "spin_pino_6node":
        return SpinPINO(width=4, modes=2, n_layers=2, grid_size=8,
                        points_per_axis=6, hidden_dim=48, smoothing=0.02)
    if kind == "spin_pino_6node_gauss":
        return SpinPINO(width=4, modes=2, n_layers=2, grid_size=8,
                        points_per_axis=6, hidden_dim=48, smoothing=0.02,
                        quadrature_rule="gauss")
    if kind == "spin_pino_6node_logistic":
        return SpinPINO(width=4, modes=2, n_layers=2, grid_size=8,
                        points_per_axis=6, hidden_dim=48, smoothing=0.02,
                        moment_parameterization="logistic_atom", temperature=0.12)
    if kind == "w6_m7_realizable_ufno":
        return RealizableUFNOMoment(
            rule="gauss", points_per_axis=6, grid_size=8,
            hidden=32, temperature=0.12,
        )
    if kind == "w5_voxel_deeponet":
        return W5VoxelDeepONet()
    raise ValueError(f"unknown model kind {kind}")


def _predict(model, kind, D, c):
    if kind.startswith("spin_pino"):
        weights = model.encode_geometry(D)
        return (
            model.moment_from_weights(weights, c),
            model.jacobian_from_weights(weights, c),
            weights,
        )
    z = model.encode_geometry(D)
    return model.moment(z, c), model.jacobian(z, c), z


def _metrics(model, kind, data):
    D, c, target_r, target_J, _ = data
    model.eval()
    with torch.no_grad():
        pred_r, pred_J, _ = _predict(model, kind, D, c)
    r_abs = (pred_r - target_r).abs()
    j_abs = (pred_J - target_J).abs()
    return {
        "r_mae": float(r_abs.mean()),
        "r_relative_l2": float((pred_r-target_r).norm()/target_r.norm().clamp_min(1e-12)),
        "r_component_mae": [float(x) for x in r_abs.mean(0)],
        "J_mae": float(j_abs.mean()),
        "J_relative_l2": float((pred_J-target_J).norm()/target_J.norm().clamp_min(1e-12)),
        "J_component_mae": [float(x) for x in j_abs.mean((0,1))],
        "negative_mass_fraction": float((pred_r[:,0] < -1e-7).float().mean()),
    }


def _fit(model, kind, fit, validation, *, seed, steps=120, batch_size=8):
    generator = torch.Generator().manual_seed(seed + 31337)
    schedule = torch.randint(
        len(fit[0]), (steps, min(batch_size, len(fit[0]))), generator=generator
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-3)
    pino_loss = SpinPINOLoss(lambda_data=1.0, lambda_physics=0.05, lambda_jacobian=0.25)
    best_score = math.inf
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    started = time.perf_counter()
    for step, indices in enumerate(schedule, 1):
        model.train()
        D, c, target_r, target_J = (x[indices] for x in fit[:4])
        pred_r, pred_J, cache = _predict(model, kind, D, c)
        if kind.startswith("spin_pino"):
            losses = pino_loss(
                model, cache, c, target_moments=target_r,
                target_jacobian=target_J, derivative_admitted=True,
            )
            loss = losses["total"]
            physics_value = float(losses["physics"].detach())
        else:
            loss = F.mse_loss(pred_r, target_r) + 0.25 * F.mse_loss(pred_J, target_J)
            physics_value = 0.0
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"{kind}: nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == steps:
            val = _metrics(model, kind, validation)
            score = val["r_mae"] + 0.25 * val["J_mae"]
            history.append({
                "step": step,
                "train_loss": float(loss.detach()),
                "train_physics": physics_value,
                "validation_score": score,
                "validation": val,
            })
            if score < best_score:
                best_score = score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    return {
        "steps": steps, "batch_size": batch_size,
        "selected_step": best_step, "training_seconds": seconds,
        "best_validation_score": best_score, "history": history,
    }


def run(*, seeds=(131,137,139), steps=120, batch_size=8,
        train_geometries_per_family=16, test_ellipsoid_geometries=8):
    torch.set_num_threads(1)
    kinds = (
        "spin_pino_4node",
        "spin_pino_6node",
        "spin_pino_6node_gauss",
        "spin_pino_6node_logistic",
        "w6_m7_realizable_ufno",
        "w5_voxel_deeponet",
    )
    all_rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        splits = generate_splits(
            seed=seed,
            train_geometries_per_family=train_geometries_per_family,
            test_ellipsoid_geometries=test_ellipsoid_geometries,
            queries_per_geometry=2,
            grid_size=8,
        )
        fit, validation, test = (splits[k] for k in ("fit","validation","test"))
        model_rows = {}
        for kind in kinds:
            torch.manual_seed(seed)
            model = _build(kind)
            params = sum(p.numel() * (2 if p.is_complex() else 1)
                         for p in model.parameters() if p.requires_grad)
            training = _fit(
                model, kind, fit, validation, seed=seed,
                steps=steps, batch_size=batch_size,
            )
            model_rows[kind] = {
                "parameters": params,
                "training": training,
                "fit": _metrics(model,kind,fit),
                "validation": _metrics(model,kind,validation),
                "held_out_ellipsoid": _metrics(model,kind,test),
            }
        all_rows.append({
            "seed": seed,
            "dataset": splits["metadata"],
            "models": model_rows,
        })
    aggregate = {}
    for kind in kinds:
        aggregate[kind] = {
            "parameters_mean": statistics.mean(row["models"][kind]["parameters"] for row in all_rows),
            "r_mae": statistics.mean(row["models"][kind]["held_out_ellipsoid"]["r_mae"] for row in all_rows),
            "r_relative_l2": statistics.mean(row["models"][kind]["held_out_ellipsoid"]["r_relative_l2"] for row in all_rows),
            "J_mae": statistics.mean(row["models"][kind]["held_out_ellipsoid"]["J_mae"] for row in all_rows),
            "J_relative_l2": statistics.mean(row["models"][kind]["held_out_ellipsoid"]["J_relative_l2"] for row in all_rows),
            "training_seconds": statistics.mean(row["models"][kind]["training"]["training_seconds"] for row in all_rows),
        }
        aggregate[kind]["combined_score"] = (
            aggregate[kind]["r_relative_l2"] + aggregate[kind]["J_relative_l2"]
        ) / 2
    return {
        "schema": "SPIN_PINO_M3_SAME_ANALYTIC_DATA_TOURNAMENT_V1",
        "label_source": "ANALYTIC_CONTINUUM_P0_NOT_W2_CHERCHI",
        "w2_or_cherchi_labels_used": False,
        "geometry_split": "fit/validation box+sphere; held-out ellipsoid family",
        "same_inputs_labels_splits_steps_optimizer": True,
        "seeds": list(seeds),
        "steps": steps,
        "models": kinds,
        "aggregate": aggregate,
        "per_seed": all_rows,
        "limits": [
            "Analytic vertical-plane cuts only; not arbitrary 10D quadrics.",
            "This is a same-data architecture comparison, not native Spin/Cherchi validation.",
            "Parameter counts are reported and not forcibly equal; interpret accuracy alongside capacity and runtime.",
            "No model Jacobian is admitted to Newton."
        ],
        "selected_by_combined_relative_l2": min(
            kinds, key=lambda k: aggregate[k]["combined_score"]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="131,137,139")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        seeds=tuple(int(x) for x in args.seeds.split(",")),
        steps=args.steps, batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"aggregate":result["aggregate"],
                      "selected":result["selected_by_combined_relative_l2"]},indent=2))


if __name__ == "__main__":
    main()
