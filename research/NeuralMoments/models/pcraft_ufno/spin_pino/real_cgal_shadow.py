"""Frozen analytic-trained W6 evaluation on three discrete real-geometry teacher calls."""
import argparse
import hashlib
import json
from pathlib import Path
from fractions import Fraction

import torch.nn.functional as F

import numpy as np
import torch

from .dataset import generate_splits
from .spin_pino import NATIVE_B
from ..model import canonical_geometry
from .support_ablation import PINO, SEEDS
from .tournament import _build, _fit, _metrics, _predict

HERE = Path(__file__).resolve().parent


def translation(center):
    """S m(y) = m(x) where x=y+center. T=B^-1 S^T B gives q_y=T q_x."""
    center = np.asarray(center, dtype=float)
    S = np.eye(10)
    S[1:4, 0] = center
    for row, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)), 4):
        S[row, 0] = center[i] * center[j]
        S[row, 1+i] += center[j]
        S[row, 1+j] += center[i]
    B = np.asarray(NATIVE_B)
    return S, (S.T * B[None, :]) / B[:, None]


def geometry(v):
    a = np.asarray(v['occupancy'])
    if (v['indexing'] != 'ij' or v['cube_halfwidth'] != 1 or a.shape != (8,8,8)
        or not np.isin(a, (0,1)).all() or int(a.sum()) != v['occupied']
        or not np.allclose(v['axis'], np.linspace(-1,1,8), atol=1e-15, rtol=0)):
        raise ValueError('invalid unit-density point occupancy')
    rho = torch.as_tensor(a, dtype=torch.float32)
    return torch.stack((rho, rho, rho*0, rho*0)).unsqueeze(0)


def relative(a, b):
    return float(np.linalg.norm(a-b) / np.linalg.norm(b))


def support_mass_ratios(model, X, c, true_mass):
    """Exact bounds for positive sums of the captured binary32 logistic activations."""
    geometry_channels = canonical_geometry(X)
    density = F.grid_sample(geometry_channels[:,0:1], model.voxel_sampling_grid,
                            align_corners=True, mode='bilinear', padding_mode='zeros').flatten()
    occupancy = F.grid_sample(geometry_channels[:,1:2], model.voxel_sampling_grid,
                              align_corners=True, mode='bilinear', padding_mode='zeros').flatten()
    support = (density>0) & (occupancy>0)
    n = model.monomials.shape[0]
    _,_,_,_,_,alpha,_ = model._logistic_atom_terms(c.new_ones(len(c), n),c)
    reference = 2
    rows=[]
    for i in (0,1):
        # For shared nonnegative weights w, m_i/m_ref is an average of
        # alpha_i/alpha_ref with nonnegative weights w * alpha_ref.
        lower = min(Fraction(float(a))/Fraction(float(b)) for a,b in
                    zip(alpha[i,support],alpha[reference,support]))
        target = Fraction(float(true_mass[i]))/Fraction(float(true_mass[reference]))
        # m_i <= true_i*(1+e), m_ref >= true_ref*(1-e) for e<1.
        floor = max(Fraction(0),(lower-target)/(lower+target))
        rows.append(dict(query=i,reference=reference,truth_ratio=float(target),
                         minimum_supported_activation_ratio=str(lower),activation_ratio_lower_approx=float(lower),
                         pairwise_max_relative_mass_error_at_least=str(floor.numerator*10**8//floor.denominator)+'/100000000'))
    return dict(support_atoms=int(support.sum()),all_atoms=n,pairs=rows,
                scope='Positive real sums with this binary32 geometry support, coefficient adapter and logistic activations before reduction rounding.')


def run(path):
    data = path.read_bytes()
    packet = json.loads(data)
    voxels = packet['voxels']
    X = geometry(voxels)
    S, T = translation(voxels['cube_center_in_tip'])
    rows = packet['samples']
    if [r['name'] for r in rows] != ['ellipsoid-1','ellipsoid-6','hyperboloid-1']:
        raise ValueError('unexpected preselected query list')
    raw = np.array([r['q'] for r in rows])
    c = torch.as_tensor(raw @ T.T, dtype=torch.float32)
    refs = json.loads((HERE/'SUPPORT_HOLDOUT.json').read_bytes())
    results = []
    capacity = None
    torch.set_num_threads(1)
    for seed in SEEDS:
        splits = generate_splits(seed=seed)
        torch.manual_seed(seed)
        model = _build(PINO)
        model.density_is_premultiplied = True
        fit = _fit(model, PINO, splits['fit'], splits['validation'], seed=seed, physics_weight=.05)
        ref = next(r['arms']['single_physics'] for r in refs['raw'] if r['seed']==seed)
        analytic = _metrics(model,PINO,splits['test'])
        if fit['selected_step'] != ref['validation_selected_step'] or not np.allclose(
                [analytic['r_relative_l2'],analytic['J_relative_l2']],[ref['original_ellipsoid']['r'],ref['original_ellipsoid']['J']],
                atol=1e-5,rtol=0):
            raise ValueError('analytic replay differs from the frozen arm')
        model.eval()
        with torch.inference_mode():
            m, J, _ = _predict(model, PINO, X.expand(len(rows),-1,-1,-1,-1), c)
            check = support_mass_ratios(model,X,c,[r['variable_ten_tip'][0] for r in rows])
            if capacity is not None and check!=capacity:
                raise ValueError('supposedly weight-independent mass-ratio bound changed')
            capacity=check
        predicted_r = m.double().numpy() @ S.T
        predicted_J = S @ J.double().numpy() @ T
        truth_r = np.array([r['variable_ten_tip'] for r in rows])
        truth_J = np.array([r['J_tip'] for r in rows])
        scores = []
        for i,r in enumerate(rows):
            scores.append(dict(name=r['name'],true_mass=truth_r[i,0],predicted_mass=predicted_r[i,0],
                               r_relative_l2=relative(predicted_r[i],truth_r[i]),
                               J_relative_l2=relative(predicted_J[i],truth_J[i]),
                               max_absolute_J=float(abs(predicted_J[i]-truth_J[i]).max()),
                               predicted_ten_tip=predicted_r[i].tolist(),predicted_J_tip=predicted_J[i].tolist()))
        results.append(dict(seed=seed,validation_selected_step=fit['selected_step'],
                            analytic_r=analytic['r_relative_l2'],analytic_J=analytic['J_relative_l2'],
                            cases=scores,pooled_r=relative(predicted_r,truth_r),pooled_J=relative(predicted_J,truth_J)))
    means = [dict(name=r['name'],mean_r=float(np.mean([s['cases'][i]['r_relative_l2'] for s in results])),
                  mean_J=float(np.mean([s['cases'][i]['J_relative_l2'] for s in results]))) for i,r in enumerate(rows)]
    return dict(schema='w6-frozen-analytic-to-discrete-cgal-real-shadow-v1',input_sha256=hashlib.sha256(data).hexdigest(),
                pretrained_protocol_sha256=hashlib.sha256((HERE/'SUPPORT_HOLDOUT.json').read_bytes()).hexdigest(),
                pytorch=torch.__version__,voxel_occupied=voxels['occupied'],models=results,seed_means=means,
                logistic_head_mass_ratio=capacity,
                averages_of_pooled_errors={k:float(np.mean([row[k] for row in results])) for k in ('pooled_r','pooled_J')},
                label_scope='variable Dprime retained positive side, tip moments; no fixed shell',
                real_geometry_count=1,real_supervised_samples=0,real_adaptation_steps=0,newton_calls_by_network=0)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=HERE/'M35_REAL_INPUT.json')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.write_text(json.dumps(run(args.input),indent=2,allow_nan=False)+'\n')
