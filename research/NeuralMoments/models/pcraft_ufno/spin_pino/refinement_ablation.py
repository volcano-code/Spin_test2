"""Factorial frozen-PINO and numerical quadrature comparison on individual cuts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..model import canonical_geometry
from .dataset import generate_splits
from .half_ellipsoid import panel
from .real_cgal_shadow import geometry, translation, relative
from .refined_quadrature import cartesian, localized
from .support_ablation import PINO, SEEDS
from .tournament import _build, _fit

HERE = Path(__file__).resolve().parent
ARMS = {"coarse": (6, .12), "cool": (6, .0075), "fine": (24, .12),
        "fine_cool": (24, .0075), "chart": None}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def real_cases(path, group):
    d = json.loads(path.read_bytes())
    X = geometry(d["voxels"])
    S, T = translation(d["voxels"]["cube_center_in_tip"])
    return [(group, row["name"], X, np.array(row["q"]), np.array(row["variable_ten_tip"]),
             np.array(row["J_tip"]), S, T, row["result_sha256"]) for row in d["samples"]]


def run(real_input, extra, anchor, query):
    torch.set_num_threads(1)
    holdout = panel()
    cases = real_cases(real_input, "Noodles")
    cases += [("analytic", name, v, q, r, j, np.eye(10), np.eye(10), key)
              for name, v, q, r, j, key in holdout]
    if extra:
        cases.extend(real_cases(extra, "heldout_real"))
    historic = json.loads((HERE / "M35_REAL_SHADOW.json").read_bytes())
    trials = []
    anchors = []
    numerical = {}
    for seed in SEEDS:
        data = generate_splits(seed=seed)
        reference = next(row for row in historic["models"] if row["seed"] == seed)
        assert not {row[-1] for row in holdout}.intersection(
            row["geometry_id"] for part in ("fit", "validation", "test") for row in data[part][4])
        torch.manual_seed(seed)
        model = _build(PINO)
        model.density_is_premultiplied = True
        fit = _fit(model, PINO, data["fit"], data["validation"], seed=seed, physics_weight=.05)
        if fit["selected_step"] != reference["validation_selected_step"]:
            raise ValueError("analytic checkpoint replay changed")
        model.eval()
        if query and anchor:
            # The exact Newton state and target labels are not used in proposal generation.
            request=json.loads(query.read_bytes())
            _, _, X, _, _, _, S, T, _ = cases[0]
            q=np.array(request["q"])
            with torch.inference_mode():
                gates=model.weight_head(model.encoder(canonical_geometry(X))).sigmoid().reshape(6,6,6).numpy()
            for mode, gate in (("learned",gates),("unit",np.ones_like(gates))):
                for arm in ("coarse","fine_cool","chart"):
                    if ARMS[arm]:
                        r,J=cartesian(X,gate,T@q,*ARMS[arm])
                    else:
                        r,J,_=localized(X,gate,T@q)
                    anchors.append(dict(name="requested",source_sha256=digest(query),seed=seed,mode=mode,arm=arm,
                                        q=q.tolist(),J_tip=(S@J@T).tolist(),r_tip=(S@r).tolist()))
        measured = []
        for group, name, X, q, r, J, S, T, key in cases:
            with torch.inference_mode():
                f = model.encoder(canonical_geometry(X))
                learned = model.weight_head(f).sigmoid().reshape(6, 6, 6).numpy()
            for mode, gates in (("learned", learned), ("unit", np.ones_like(learned))):
                if mode == "unit" and seed != SEEDS[0]:
                    continue
                for arm, rule in ARMS.items():
                    if rule is None:
                        a, b, method = localized(X, gates, T @ q)
                    else:
                        a, b = cartesian(X, gates, T @ q, *rule)
                        method = "midpoint_logistic"
                    a, b = S @ a, S @ b @ T
                    row = dict(group=group, case=name, arm=arm, mode=mode, method=method,
                               r=relative(a, r), J=relative(b, J), mass=float(a[0]),
                               relative_mass=float(abs(a[0]/r[0] - 1)))
                    if name == "ellipsoid-1" and group == "Noodles":
                        anchors.append(dict(name="ellipsoid-1", seed=seed, mode=mode, arm=arm, J_tip=b.tolist(),
                                            r_tip=a.tolist(), q=q.tolist(), result_sha256=key))
                    if arm == "coarse" and mode == "learned" and group == "Noodles":
                        prev = next(x for x in reference["cases"] if x["name"] == name)
                        np.testing.assert_allclose([row["r"], row["J"]], [prev["r_relative_l2"],
                                                   prev["J_relative_l2"]], rtol=1e-5, atol=1e-6)
                    if mode=="learned":
                        measured.append(row)
                    else:
                        numerical[group, name, arm] = row
        trials.append(dict(seed=seed, validation_selected_step=fit["selected_step"], scores=measured))
        if anchor:
            tmp = anchor.with_suffix(".pending")
            tmp.write_text(json.dumps(dict(schema="W6_M36_CAVITY_ANCHOR",
                positive_mass=True, coefficient_derivative="tip raw ambient, ten native c incl doubled cross terms",
                samples=anchors), separators=(",", ":"), allow_nan=False)+"\n")
            tmp.replace(anchor)
    values = ("r", "J", "mass", "relative_mass")
    mean = []
    for i, first in enumerate(trials[0]["scores"]):
        items = [t["scores"][i] for t in trials]
        mean.append({**first, **{v:float(np.mean([x[v] for x in items])) for v in values}})
    inputs = [dict(group=group, case=name, source=key, q=q.tolist(), true_mass=float(r[0]))
              for group, name, _, q, r, _, _, _, key in cases]
    return dict(schema="W6_M36_FROZEN_REFINEMENT_V1", protocol_sha256=digest(HERE/"M36_PROTOCOL.json"),
                M35_input_sha256=digest(real_input), heldout_real_sha256=digest(extra) if extra else None,
                inputs=inputs, learned_seed_means=mean, learned_seeds=trials,
                numerical=list(numerical.values()), pytorch=torch.__version__, real_train_queries=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=HERE/"M35_REAL_INPUT.json")
    p.add_argument("--extra", type=Path)
    p.add_argument("--query", type=Path)
    p.add_argument("--anchor", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a.input, a.extra, a.anchor, a.query), separators=(",", ":"),
                                   allow_nan=False)+"\n")
