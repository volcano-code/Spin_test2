"""M4 ablation of PINO loss conditioning and gradient conflict mitigation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .train import run_seed


CONFIGS = (
    {"name": "baseline_m2b", "preconditioned_loss": False, "pcgrad": False},
    {"name": "diagonal_preconditioned", "preconditioned_loss": True, "pcgrad": False},
    {"name": "diagonal_preconditioned_pcgrad", "preconditioned_loss": True, "pcgrad": True},
)


def run(*, seeds=(131,137,139), steps=120, batch_size=8):
    rows = []
    for cfg in CONFIGS:
        per_seed = []
        for seed in seeds:
            result = run_seed(
                seed=seed,
                train_geometries_per_family=16,
                test_ellipsoid_geometries=8,
                queries_per_geometry=2,
                steps=steps,
                batch_size=batch_size,
                grid_size=8,
                points_per_axis=6,
                smoothing=0.02,
                preconditioned_loss=cfg["preconditioned_loss"],
                pcgrad=cfg["pcgrad"],
                device="cpu",
            )
            result.pop("checkpoint_state", None)
            per_seed.append(result)
        def avg(section, metric):
            return statistics.mean(float(x[section][metric]) for x in per_seed)
        rows.append({
            "config": cfg,
            "seeds": list(seeds),
            "steps": steps,
            "batch_size": batch_size,
            "parameters": per_seed[0]["trainable_parameters"],
            "mean_heldout_r_mae": avg("held_out_ellipsoid_metrics","r_mae"),
            "mean_heldout_J_mae": avg("held_out_ellipsoid_metrics","J_mae"),
            "mean_heldout_r_relative_l2": avg("held_out_ellipsoid_metrics","r_relative_l2"),
            "mean_heldout_J_relative_l2": avg("held_out_ellipsoid_metrics","J_relative_l2"),
            "mean_training_seconds": statistics.mean(x["training"]["seconds"] for x in per_seed),
            "preconditioner": per_seed[0]["loss_conditioning"]["preconditioner"],
            "per_seed": per_seed,
        })
    for row in rows:
        row["combined_score"] = (
            row["mean_heldout_r_relative_l2"] + row["mean_heldout_J_relative_l2"]
        ) / 2
    return {
        "schema": "SPIN_PINO_M4_LOSS_CONDITIONING_V1",
        "label_source": "ANALYTIC_CONTINUUM_P0_NOT_W2_CHERCHI",
        "w2_or_cherchi_labels_used": False,
        "same_data_splits_seeds_optimizer_steps": True,
        "loss_preconditioner_fit_on_training_split_only": True,
        "models": rows,
        "selected_by_combined_relative_l2": min(rows,key=lambda x:x["combined_score"])["config"],
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",default="131,137,139")
    parser.add_argument("--steps",type=int,default=120)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run(seeds=tuple(int(x) for x in args.seeds.split(",")),
               steps=args.steps,batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "selected":result["selected_by_combined_relative_l2"],
        "results":[{k:x[k] for k in ("config","mean_heldout_r_relative_l2","mean_heldout_J_relative_l2","combined_score","mean_training_seconds")} for x in result["models"]]
    },indent=2))


if __name__=="__main__":
    main()
