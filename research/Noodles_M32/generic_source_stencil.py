"""Independently query every native general-source velocity column without further Boolean calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np


def audit(sample, bounds, executable, workers=1):
    src=json.loads((sample/'source.json').read_bytes())
    r=json.loads((sample/'receipt.json').read_bytes())
    if (sha256((sample/'source.json').read_bytes()).hexdigest()!=r['source_sha256'] or
        sha256(executable.read_bytes()).hexdigest()!=r['source_executable'] or
        sha256(bounds.read_bytes()).hexdigest()!=src['bound_reference_sha256']):
        raise ValueError('wrong source, binary or fixed truncation input')
    q=np.asarray(src['raw_q']); ref=np.asarray(src['native_jacobian'])/np.linalg.norm(q)
    n=len(ref);faces=np.column_stack((src['faces'],src['face_is_cap']))
    def evaluate(x):
        lines=subprocess.run([str(executable),str(bounds)],input=' '.join(map(repr,map(float,x))),
                             capture_output=True,check=True,text=True).stdout.splitlines()
        h=lines[0].split()
        if (int(h[0])!=n or int(h[1])!=len(faces) or bool(int(h[2]))!=src['interior_is_positive'] or
            h[4]!=src['surface_type'] or not np.array_equal(np.loadtxt(lines[1+n:],dtype=int),faces)):
            raise ValueError('source incidence or retained side changed')
        return np.loadtxt(lines[1:1+n])[:,:3]
    if not np.array_equal(evaluate(q),np.array(src['vertices_tip'])):
        raise ValueError('baseline source replay differs')
    # Freeze bytes across the run; the path can otherwise be relinked mid-stencil.
    def pair(spec):
        exp,j=spec;d=np.eye(10)[j]*2**(-exp);l=q-d;u=q+d
        fd=(evaluate(u)-evaluate(l))/(u[j]-l[j])
        err=np.abs(fd-ref[:,:,j])
        return dict(column=j,maximum_absolute_error=float(err.max()),
                    maximum_scaled_error=float(np.max(err/np.maximum(1,np.abs(ref[:,:,j])))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        columns=list(pool.map(pair,[(e,j) for e in (19,20) for j in range(10)]))
    if sha256(executable.read_bytes()).hexdigest()!=r['source_executable']:
        raise ValueError('source binary changed during replay')
    vals=[]
    for exp in (19,20):
        per_col=columns[(exp-19)*10:(exp-18)*10]
        vals.append(dict(step=f'2^-{exp}',columns=per_col,max_scaled_error=max(x['maximum_scaled_error'] for x in per_col)))
    return dict(source_sha256=r['source_sha256'],source_executable=r['source_executable'],source_type=src['surface_type'],vertices=n,
                native_calls_excluding_baseline=40,two_sided_source_stencils=vals,
                total_source_vertex_component_column_count=n*3*10,
                scope='Native source motion only, including cap/truncation. No fresh Boolean or moment solve.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sample','bounds','executable'):p.add_argument(name,type=Path)
    p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();print(json.dumps(audit(a.sample,a.bounds,a.executable,a.workers),indent=2))
