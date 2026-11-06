"""Geometry-held-out test of support counting for effective density in the W6 logistic head."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from ..analytic_cut import GeometrySpec, vertical_cut_truth
from .dataset import generate_splits
from .equivariance import random_so3_rotation, transform_native_coefficients, transform_native_jacobian, transform_raw_moments
from .tournament import _build, _fit, _metrics, _predict

SEEDS = (131, 137, 139)
PINO = "spin_pino_6node_logistic"
DEEPONET = "w5_voxel_deeponet"
ARMS = (("w5", DEEPONET, False, 0.0),
        ("double_physics", PINO, False, 0.05), ("double_data", PINO, False, 0.0),
        ("single_physics", PINO, True, 0.05), ("single_data", PINO, True, 0.0))


def two_component_panel(seed, count=8):
    """Two separated ellipsoids; common interior cut. Directly sample the rotated analytic solid."""
    rng = np.random.default_rng(seed + 7000401)
    xyz = np.linspace(-1, 1, 8)
    grid = np.stack(np.meshgrid(xyz, xyz, xyz, indexing="ij"), axis=-1)
    results = [[], [], [], [], []]
    for i in range(count):
        components = [GeometrySpec("ellipsoid", (sign*rng.uniform(.44, .50), rng.uniform(-.06,.06), 0.),
                                   tuple(rng.uniform((.24,.40,.40), (.37,.65,.65)))) for sign in (-1,1)]
        a, b = components
        assert a.center[0]+a.axes[0] < b.center[0]-b.axes[0]
        parent = hashlib.sha256((a.geometry_id + b.geometry_id).encode()).hexdigest()
        rot = random_so3_rotation(seed + 1700001 + i, dtype=torch.float64)
        unrotated_points = grid @ rot.numpy()
        inside = np.zeros(grid.shape[:-1], dtype=bool)
        for part in components:
            inside |= (((unrotated_points - part.center)/part.axes)**2).sum(-1) <= 1
        effective = torch.tensor(inside, dtype=torch.float32)
        voxels = torch.stack((effective, effective, effective*0, effective*0))
        for height in (-.125, .125):
            labels = [vertical_cut_truth(p, height/p.axes[2]) for p in components]
            np.testing.assert_allclose(labels[0][0], labels[1][0], atol=1e-15, rtol=0)
            c, m1, j1 = labels[0]
            _, m2, j2 = labels[1]
            raw = transform_raw_moments(torch.from_numpy(m1+m2)[None],rot)[0]
            jac = transform_native_jacobian(torch.from_numpy(j1+j2)[None],rot)[0]
            native = transform_native_coefficients(torch.from_numpy(c)[None],rot)[0]
            for col,value in zip(results,(voxels,native.float(),raw.float(),jac.float(),parent)):
                col.append(value)
    return (*[torch.stack(col) for col in results[:4]], results[4])


def score(model, kind, data):
    # Three predetermined ambient native coefficient directions; units and convention match training labels.
    D, c, truth_r, truth_j, _ = data
    probes = torch.eye(10, dtype=c.dtype)[:, [0, 1, 5]]
    model.eval()
    with torch.no_grad():
        r, j, _ = _predict(model,kind,D,c)
    def relative(actual, expected):
        return float((actual-expected).norm()/expected.norm())
    return dict(r=relative(r,truth_r), J=relative(j,truth_j),
                Jv=relative(j@probes,truth_j@probes))


def run():
    torch.set_num_threads(1)
    records = []
    for seed in SEEDS:
        splits = generate_splits(seed=seed)
        panel = two_component_panel(seed)
        parents = sorted(set(panel[4]))
        assert not set(parents).intersection(x["geometry_id"] for x in splits["fit"][4])
        digest = hashlib.sha256()
        for column in panel[:4]:
            digest.update(column.contiguous().numpy().tobytes())
        out = {}
        for arm, kind, single, weight in ARMS:
            torch.manual_seed(seed)
            model = _build(kind)
            if kind == PINO:
                model.density_is_premultiplied = single
            fit = _fit(model,kind,splits["fit"],splits["validation"],seed=seed,physics_weight=weight)
            ref = _metrics(model,kind,splits["test"])
            out[arm] = dict(two_component=score(model,kind,panel),validation_selected_step=fit["selected_step"],
                            original_ellipsoid=dict(r=ref["r_relative_l2"],J=ref["J_relative_l2"]))
        records.append(dict(seed=seed,parents=parents,test_tensor_sha256=digest.hexdigest(),arms=out))
    summary = {arm:{k:mean(x["arms"][arm]["two_component"][k] for x in records) for k in ("r","J","Jv")}
               for arm,*_ in ARMS}
    return dict(schema="spin-pino-effective-density-analytic-holdout-v1",torch_version=torch.__version__,
                settings=dict(grid=8,queries_per_parent=2,parents_per_seed=8,training_steps=120,batch_size=8,
                              Jv_columns=[0,1,5],adaptation_steps=0),
                raw=records,means=summary,real_cherchi_samples=0,newton_runs=0)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(run(),indent=2,allow_nan=False)+"\n")
