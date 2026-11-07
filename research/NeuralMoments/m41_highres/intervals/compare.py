"""Compare the smooth positive quadric with registered finite native cuts of Noodles."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from line_operator import evaluate
from ray_bank import load_design, material_rays, DESIGN_SHA


def relative(value, ref):
    return float(np.linalg.norm(value-ref)/np.linalg.norm(ref))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit', type=Path, required=True, help='extracted Spin_5090_HighRes_Kit')
    p.add_argument('--sizes', nargs='+', type=int, default=[32,64,128,256,512])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    vertices, faces = load_design(args.kit/'inputs/Dprime.txt')
    path = args.kit/'native/exact/report.json'
    report = json.loads(path.read_bytes())
    if sha256(path.read_bytes()).hexdigest() != '0872197d53d0744fee9968dbf0d80c8bdffab65b942620313d2c70aaf23061aa':
        raise ValueError('unexpected M38 trace')
    states = [r for r in report['evaluations'] if 'J_tip' in r]
    if [r['index'] for r in states] != [1,6]:
        raise ValueError("unexpected native state IDs")
    qs = [np.array(r['q']) for r in states]
    result = dict(reference='cached finite native moving-face J_tip in tip coordinates',
                  grid='axis families, side x side transverse rays per family; positive side; unique transverse phase fixed per side',
                  design_sha256=DESIGN_SHA, trace_sha256=sha256(path.read_bytes()).hexdigest(), rows=[])
    for n in args.sizes:
        banks = [material_rays(vertices, faces, axis, n) for axis in range(3)]
        for state,q in zip(states,qs):
            delta = qs[1]-qs[0]
            v = delta-q*(q@delta)/(q@q)
            v /= np.linalg.norm(v)
            ref = np.array(state['J_tip'])
            refm = np.array(state['variable_ten_tip'])
            ms, js, axis_rows = [], [], []
            for axis, bank in enumerate(banks):
                m, j, diag = evaluate(bank, axis, q)
                ms.append(m)
                js.append(j)
                axis_rows.append(dict(axis='xyz'[axis], intervals=len(bank[1]), **diag,
                                      native_Jv_relative=relative(j@v,ref@v)))
            m, j = np.mean(ms,axis=0), np.mean(js,axis=0)
            result['rows'].append(dict(side=n, state=state['index'], axes=axis_rows,
                direction='forward trajectory tangent on the unit q-sphere between accepted states 1 and 6',
                native_moments_relative=relative(m,refm), native_J_relative=relative(j,ref),
                native_Jv_relative=relative(j@v,ref@v),
                native_izz_gradient_relative=relative(j[4]+j[5],ref[4]+ref[5]),
                native_Jv=(ref@v).tolist(), estimated_Jv=(j@v).tolist(),
                radial_defect=float(np.linalg.norm(j@q)/np.linalg.norm(j))))
            print(n, state['index'], result['rows'][-1]['native_moments_relative'],
                  result['rows'][-1]['native_Jv_relative'], axis_rows, flush=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
