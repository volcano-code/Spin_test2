"""Learn positive subcell quadrature weights. NumPy CPU, one fixed mesh, ten archived cuts."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

TRACE = '0872197d53d0744fee9968dbf0d80c8bdffab65b942620313d2c70aaf23061aa'
TRUTH = '5ef44d53354758034f71bdc171aa156a2e69f633cd216a58ae9782861721504d'
PANEL = 'ef3ec5619e62ebce9de998c5eb8a69032abb6a4aafc433e28941677161ea622f'
RIDGE = 0.1
TRAIN = np.arange(5)
TEST = np.arange(5, 10)


def checked(path, digest):
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError(f'input SHA-256 mismatch: {path}')
    return data


def tip(ten):
    m = np.array(ten, dtype=float)
    m[6] += 0.32*m[3] + 0.16**2*m[0]
    m[7] += 0.16*m[2]
    m[8] += 0.16*m[1]
    m[3] += 0.16*m[0]
    return m


def phase_moments(mask, n):
    """(z slabs, phase in 2x2x2 supercell, ten raw moments), with exact cube basis."""
    from vtkmodules.vtkCommonCore import reference

    extents = []
    lo, hi, it = reference(0), reference(0), reference(0)
    for k in range(n):
        for j in range(n):
            it.set(0)
            while mask.GetNextExtent(lo, hi, 0, n-1, j, k, it):
                extents.append((int(lo), int(hi)+1, j, k))
    starts, stops, j, k = np.asarray(extents, dtype=int).T
    h = 2.0/n
    i = np.arange(n)
    xy = -1 + (i + 0.5)*h
    z = -0.16 + (i + 0.5)*h
    yy, zz = xy[j], z[k]
    corr = h*h/12
    result = np.zeros((8, 8, 10))
    for px in range(2):
        prefix = np.c_[np.ones(n), xy, xy*xy+corr] * (i%2 == px)[:, None]
        prefix = np.vstack([np.zeros(3), np.cumsum(prefix,axis=0)])
        v, x, xx = (prefix[stops]-prefix[starts]).T
        moments = (v,x,v*yy,v*zz,xx,v*(yy*yy+corr),v*(zz*zz+corr),v*yy*zz,x*zz,x*yy)
        classes = 8*(k*8//n) + (px+2*(j%2)+4*(k%2))
        for c, values in enumerate(moments):
            result[:, :, c] += np.bincount(classes, weights=values, minlength=64).reshape(8,8)
    return result * h**3


def extract(args):
    sys.path.insert(0, str(args.geometry_code.resolve()))
    from noodles_resolution import DESIGN_SHA, FULL_SHA, FIXED_SHA, parse_mesh, stencil, ref_moments
    from paired_cuts import retained_stencil
    from vtkmodules.vtkImagingCore import vtkImageStencilData
    from vtkmodules.vtkCommonCore import vtkVersion

    root = args.handoff
    native = json.loads(checked(root/'online/exact/report.json', TRACE))['evaluations']
    panel = json.loads(checked(root/'panel/truth.json', TRUTH))['labels']
    file_queries = json.loads(checked(root/'panel/queries.json', PANEL))['queries']
    dataset = [(root/'online/exact'/f"call-{k['index']:03d}", k) for k in native[1:]]
    dataset += [(root/'panel/exact'/k['id'], k) for k in panel]
    original = ref_moments(root/'inputs/A.ten', FULL_SHA)
    fixed = ref_moments(root/'inputs/Fprime.ten', FIXED_SHA)
    design, _ = parse_mesh(root/'inputs/Dprime.txt')
    truth, qs, covers, info = [], [], [], []
    for folder, row in dataset:
        receipt = json.loads((folder/'receipt.json').read_bytes())
        assert receipt['design_sha256'] == DESIGN_SHA
        source = json.loads(checked(folder/'source.json', receipt['source_sha256']))
        raw = (folder/'moments.json').read_bytes()
        moments = json.loads(raw)
        assert row['q'] == source['raw_q']
        assert moments['source_sha256'] == receipt['result_sha256']
        assert moments['unit_density'] and moments['frame'] == 'model origin, retained variable region'
        assert np.allclose(tip(moments['ten']), row['variable_ten_tip'], rtol=1e-12, atol=1e-13)
        assert receipt['operation'] in ('intersection','difference')
        if 'id' in row:
            assert receipt == row['boolean'] and sha256(raw).hexdigest() == row['moment_packet_sha256']
            assert row['q'] == next(q['q'] for q in file_queries if q['id']==row['id'])
        else:
            assert row['result_sha256'] == receipt['result_sha256'] and row['operation'] == receipt['operation']
        checked(folder/'result.json', receipt['result_sha256'])
        cutter, _ = parse_mesh(folder/'cutter.txt', receipt['cutter_sha256'])
        covers.append((receipt['operation'], cutter))
        qs.append(row['q'])
        truth.append(moments['ten'])
        info.append(dict(id=folder.name, operation=receipt['operation'], cutter_sha256=receipt['cutter_sha256'],
                         moment_sha256=sha256(raw).hexdigest(), source_sha256=receipt['source_sha256'],
                         result_sha256=receipt['result_sha256']))
    qs = np.array(qs)
    assert len(qs) == len(np.unique(qs, axis=0)) == 10
    arrays = dict(truth=truth, q=qs, info=json.dumps(info), sides=args.sizes, original_izz=original[1],
                  fixed_izz=fixed[1], vtk=vtkVersion.GetVTKVersion())
    for n in args.sizes:
        if n%8 or n<16: raise ValueError('sides must be >=16 and divisible by 8')
        d = stencil(design, n)
        cases = []
        for op,cutter in covers:
            s = stencil(cutter, n)
            if op == 'intersection':
                kept = retained_stencil(d,s)
            else:
                kept = vtkImageStencilData()
                kept.DeepCopy(d)
                kept.Subtract(s)
            cases.append(phase_moments(kept,n))
        arrays[f'phases{n}'] = np.stack(cases)
        print('extracted', n, flush=True)
    np.savez_compressed(args.features, **arrays)


def gains(logits):
    p = np.exp(logits - np.max(logits.real))
    return 8*p/p.sum()


def fit(parts, targets):
    """Fixed objective: per-query norm, unit training error scale, 0.1 weight shrinkage."""
    denom = np.linalg.norm(targets,axis=-1)[:,None]
    whole = parts.sum(axis=1)
    unit_res = (whole.sum(axis=1)-targets)/denom
    scale = np.mean(np.sum(unit_res**2,axis=1))
    if scale <= 0: raise ValueError('zero uniform training residual')

    def value(logits):
        w = gains(logits)
        r = (np.einsum('npc,p->nc', whole, w)-targets)/denom
        return np.mean(np.sum(r*r,axis=-1))/scale + RIDGE*np.mean((w-1)**2)

    def streaming(logits):
        w = gains(logits)
        m = np.zeros_like(targets)
        for slab in parts.transpose(1,0,2,3):
            m += np.einsum('npc,p->nc', slab, w)
        adjoint = 2*(m-targets)/(len(parts)*scale*denom**2)
        grad_w = 2*RIDGE*(w-1)/len(w)
        for slab in parts.transpose(1,0,2,3):
            grad_w += np.einsum('nc,npc->p',adjoint,slab)
        return w*(grad_w - np.dot(grad_w,w)/len(w))

    # Independent complex-step derivative of the one-pass objective on all dimensions.
    theta = np.linspace(-.15,.12,8)
    numeric = np.array([value(theta + 1j*1e-22*e).imag/1e-22 for e in np.eye(8)])
    analytic = streaming(theta)
    rel = float(np.linalg.norm(numeric-analytic) / np.linalg.norm(numeric))
    if rel>1e-7: raise ValueError(f'global loss gradient mismatch: {rel}')
    out = minimize(value, np.zeros(8), jac=streaming, method='BFGS',
                   options=dict(gtol=1e-6,maxiter=150))
    if not out.success: raise RuntimeError(out.message)
    return gains(out.x), dict(initial_loss=float(value(np.zeros(8))), final_loss=float(out.fun),
                              iterations=int(out.nit), gradient_verification_relative_l2=rel)


def report(args):
    with np.load(args.features, allow_pickle=False) as d:
        truth, qs = d['truth'], d['q']
        sources = json.loads(str(d['info']))
        izz = lambda m:(d['fixed_izz']+m[...,4]+m[...,5])/d['original_izz']
        distance = np.linalg.norm(qs[TRAIN,None,:]-qs[None,TEST,:],axis=-1)
        result = dict(schema='m38-cpu-parity-readout-v1', settings=dict(phase_count=8, sum_gains=8, ridge=RIDGE,
            train='call-002..006', evaluation='existing p1..p5', moments='raw ten in model frame, variable region only',
            parametrization='8*softmax(logits), shared across cases, trained separately at each side',
            gradient_check='two-pass 8 z slabs vs complex step of full ten-moment objective'),
            feature_sha256=sha256(args.features.read_bytes()).hexdigest(), vtk=str(d['vtk']), q_train_test_min_l2=float(distance.min()),
            samples=[dict(**src, q=q.tolist()) for src,q in zip(sources,qs)], resolutions=[])
        for n in d['sides']:
            parts = d[f'phases{n}']
            w, diagnostics = fit(parts[TRAIN], truth[TRAIN])
            e = dict(side=int(n), gains=w.tolist(), diagnostics=diagnostics, metrics={})
            for name, ww in [('uniform',np.ones(8)),('fitted',w)]:
                pred = np.einsum('nspc,p->nc',parts,ww)
                rel = np.linalg.norm(pred-truth,axis=1)/np.linalg.norm(truth,axis=1)
                pp = (izz(pred)-izz(truth))*100.0
                e['metrics'][name] = {subset: dict(median_ten_relative_l2_pct=float(np.median(rel[idx])*100),
                    worst_izz_pp=float(np.max(abs(pp[idx]))), mean_izz_abs_pp=float(np.mean(abs(pp[idx]))),
                    gate_mismatches=int(np.sum((izz(pred[idx])>=.9)!=(izz(truth[idx])>=.9))),
                    ten_relative_l2_pct=(rel[idx]*100).tolist(), izz_error_pp=pp[idx].tolist())
                    for subset, idx in [('train',TRAIN),('eval',TEST)]}
            result['resolutions'].append(e)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        for e in result['resolutions']:
            print(e['side'],e['diagnostics'],[(name,e['metrics'][name]['eval']['median_ten_relative_l2_pct'],
                 e['metrics'][name]['eval']['worst_izz_pp']) for name in e['metrics']])


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--handoff',type=Path)
    p.add_argument('--geometry-code',type=Path)
    p.add_argument('--sizes',type=int,nargs='+',default=[32,64,128,256])
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.handoff:
        if not args.geometry_code: p.error('--handoff requires --geometry-code')
        extract(args)
    report(args)
