"""Rational CGAL retained intersection, with the original level-2 ellipsoid inverse-Gauss source."""
from argparse import ArgumentParser
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time

import numpy as np

D_SHA = "c3633c74bc74c2be488eb22de108454276f28941f0477cfee5837f57db99d622"
TIP = np.array([0.,0.,-.16])
COLUMNS = (0,1,8)
EXPONENTS = (19,20)


def digest(data):
    return sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value,allow_nan=False,indent=2,sort_keys=True)+'\n')


def select_probe(design):
    lines=design.decode("ascii").splitlines()
    nv,nf=map(int,lines[0].split())
    verts=np.array([[float(Q(x)) for x in r.split()] for r in lines[1:1+nv]])
    faces=np.array([list(map(int,r.split())) for r in lines[nv+1:]])
    assert len(faces)==nf
    tris=verts[faces]
    idx=int(np.argmax(np.linalg.norm(np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]),axis=1)))
    center=tris[idx].mean(axis=0)-TIP
    axes=np.array([.11,.079,.054])
    # A small, fixed shear breaks axis, orientation and source-normal symmetry.
    basis=np.diag(axes)
    basis[0,1]=.013
    basis[1,2]=-.009
    a=-np.linalg.inv(basis@basis.T)
    q=np.r_[1+center@a@center, -2*a@center, a.diagonal(),a[1,2],a[0,2],a[0,1]]
    return q/np.linalg.norm(q),idx


def make_source(raw, executable, folder):
    raw=np.array(raw,dtype=float)
    if raw.shape!=(10,) or not np.isfinite(raw).all():
        raise ValueError('a finite ten-vector is required')
    a=[[raw[4],raw[9],raw[8]],[raw[9],raw[5],raw[7]],[raw[8],raw[7],raw[6]]]
    if max(np.linalg.eigvalsh(a))>=0:
        raise ValueError('bounded negative-definite ellipsoid required')
    source=subprocess.check_output([str(executable)],input=' '.join(map(repr,map(float,raw))),text=True).splitlines()
    nv,nf=map(int,source[0].split())
    table=np.array([list(map(float,r.split())) for r in source[1:1+nv]])
    faces=[list(map(int,r.split())) for r in source[1+nv:]]
    if len(faces)!=nf or table.shape!=(nv,36) or not np.isfinite(table).all():
        raise ValueError('invalid source packet')
    tip_xyz=table[:,3:6]
    xyz=tip_xyz+TIP
    def frac(x):
        n,d=float(x).as_integer_ratio()
        return f"{n}/{d}"
    payload=('\n'.join([f"{nv} {nf}",*(' '.join(map(frac,v)) for v in xyz),
                        *(' '.join(map(str,f)) for f in faces)])+'\n').encode('ascii')
    (folder/'cutter.txt').write_bytes(payload)
    j=table[:,6:].reshape(nv,3,10)
    dump(folder/'source.json',dict(raw_q=raw.tolist(),q=(raw/np.linalg.norm(raw)).tolist(),
         normals=table[:,:3].tolist(),vertices_tip=tip_xyz.tolist(),vertices=xyz.tolist(),
         native_jacobian=j.tolist(),faces=faces))
    return payload


def evaluate(design, q, folder, executable, source):
    if digest(design)!=D_SHA:raise ValueError('M32 exact design operand hash differs')
    folder.mkdir(parents=True,exist_ok=True)
    cutter=make_source(q,source,folder)
    combined=folder/'input.txt'
    combined.write_bytes(design+cutter)
    result=folder/'result.json'; tmp=folder/'pending.json'
    start=time.monotonic()
    subprocess.run([str(executable),str(combined),str(tmp),'intersection'],check=True)
    elapsed=time.monotonic()-start
    obj=json.loads(tmp.read_bytes()); parents=obj['birth_faces']
    if not (obj['succeeded'] and obj['operation']=='intersection' and len(parents)==len(obj['faces'])
            and min(parents)<94150<=max(parents)<94150+320):
        raise ValueError('no two-carrier retained output')
    tmp.replace(result); combined.unlink()
    record=dict(operation='intersection',design_sha256=D_SHA,cutter_sha256=digest(cutter),
        source_sha256=digest((folder/'source.json').read_bytes()),source_executable=digest(source.read_bytes()),
        csg_executable=digest(executable.read_bytes()),result_sha256=digest(result.read_bytes()),
        seconds=elapsed,vertices=len(obj['vertices']),faces=len(parents),
        design_parent_faces=sum(i<94150 for i in parents),cutter_parent_faces=sum(i>=94150 for i in parents))
    dump(folder/'receipt.json',record)
    print(folder.name,record['faces'],round(elapsed,3),flush=True)
    return record


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for k in ('design','executable','source','output'):p.add_argument('--'+k,required=True,type=Path)
    args=p.parse_args()
    d=args.design.read_bytes()
    if digest(d)!=D_SHA:raise ValueError('M32 exact design hash mismatch')
    q,idx=select_probe(d)
    args.output.mkdir(parents=True,exist_ok=True)
    dump(args.output/'protocol.json',dict(design_sha256=D_SHA,face=idx,q=q.tolist(),
         exponents=EXPONENTS,columns=COLUMNS,retained='Dprime intersect beta>=0',newton=False,
         frame='native q and differential in tip frame; output in physical model coordinates'))
    samples={'base':q}
    for j in COLUMNS:
        for e in EXPONENTS:
            for sign,letter in ((1,'p'),(-1,'m')):
                s=q.copy();s[j]+=sign*2.**-e
                samples[f'c{j}_{letter}{e}']=s
    for name,value in samples.items():
        evaluate(d,value,args.output/name,args.executable,args.source)
