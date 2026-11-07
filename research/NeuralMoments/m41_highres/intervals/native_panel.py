"""Recompute the registered five-state M37 finite-cutter check from an archived M37 run tree."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from native_flux import flux, keys
from ray_bank import crossings, DESIGN_SHA, load_design


STATES=[('exact',1),('exact',2),('exact',4),('unit',3),('learned',3)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--design',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sizes',type=int,nargs='+',default=[256,512])
    args=p.parse_args()
    vertices,faces=load_design(args.design)
    result={'scope':'five distinct M37 archived native states, same Noodles, same tip frame; reference matrices for scoring only','rows':[]}
    for n in args.sizes:
        tick=perf_counter()
        design=[keys(*crossings(vertices,faces,k,n)[:2]) for k in range(3)]
        seconds=perf_counter()-tick
        for arm,i in STATES:
            run=args.runs/arm
            row=json.loads((run/'report.json').read_bytes())['evaluations'][i-1]
            call=run/f'call-{i:03}'
            receipt=json.loads((call/'receipt.json').read_bytes())
            raw=(call/'source.json').read_bytes()
            source=json.loads(raw)
            if (sha256(raw).hexdigest()!=receipt['source_sha256'] or DESIGN_SHA!=receipt['design_sha256'] or
                    row['result_sha256']!=receipt['result_sha256'] or row['q']!=source['raw_q'] or
                    receipt['operation']!=('intersection' if source['interior_is_positive'] else 'difference')):
                raise ValueError('native binding mismatch')
            qsign=1. if receipt['operation']=='intersection' else -1.
            tick=perf_counter()
            pieces=[flux(k,n,design[k],source) for k in range(3)]
            j=qsign*sum(a for a,_ in pieces)
            query_s=perf_counter()-tick
            ref=np.array(row['J_tip'])
            result['rows'].append(dict(side=n,arm=arm,state=i,surface=source['surface_type'],
                operation=receipt['operation'],cap_output_faces=receipt['cap_output_faces'],
                native_source_sha256=receipt['source_sha256'], result_sha256=row['result_sha256'],
                native_J_relative=float(np.linalg.norm(j-ref)/np.linalg.norm(ref)),
                native_izz_gradient_relative=float(np.linalg.norm((j-ref)[4]+(j-ref)[5])/np.linalg.norm(ref[4]+ref[5])),
                material_precompute_seconds=seconds,projected_flux_seconds=query_s,
                hits_by_axis=[hits for _,hits in pieces]))
            print(n,arm,i,result['rows'][-1]['native_J_relative'],flush=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
