"""Three projected face quadratures for dM/dq of a registered native finite cutter."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from line_operator import basis
from ray_bank import crossings, load_design

REPORT_SHA = '0872197d53d0744fee9968dbf0d80c8bdffab65b942620313d2c70aaf23061aa'


def keys(ray, depth):
    keys = np.empty(len(ray), dtype=[('ray', 'i8'), ('t','f8')])
    keys['ray'], keys['t'] = ray, depth
    return keys


def flux(axis, side, design_keys, source):
    faces = np.asarray(source['faces'])
    vertices = np.asarray(source['vertices_tip'])
    ray, t, f, w, grid0 = crossings(vertices, faces, axis, side)
    cutter_keys = keys(ray, t)
    right = np.searchsorted(design_keys, cutter_keys, side='right')
    if not np.array_equal(right, np.searchsorted(design_keys, cutter_keys, side='left')):
        raise ValueError('coincident sampled material/cutter endpoints; review before use')
    inside = right % 2 == 1
    t, f, w, ray = t[inside], f[inside], w[inside], ray[inside]
    tri = vertices[faces]
    normal = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    if np.sum(tri[:,0]*normal)<=0 or np.any(np.linalg.norm(normal, axis=1)==0):
        raise ValueError('expected oriented, nondegenerate cutter')
    normal /= np.linalg.norm(normal, axis=1)[:,None]
    # Sum of n_axis^2 over axis families is one. Dividing by |n_axis|
    # converts sampled projection area to surface area, giving |n_axis| here.
    vertex_speed = np.einsum('fi,fkin->fkn', normal, np.asarray(source['native_jacobian'])[faces])
    transverse = [d for d in range(3) if d!=axis]
    area = (2./side)**2
    out = np.zeros((10,10))
    for start in range(0,len(f),16384):
        sl = slice(start,start+16384)
        x = np.empty((len(f[sl]),3))
        x[:,axis]=t[sl]
        x[:,transverse[0]]=grid0[0]+(ray[sl]%side)*2./side
        x[:,transverse[1]]=grid0[1]+(ray[sl]//side)*2./side
        speed = np.einsum('nk,nkj->nj', w[sl], vertex_speed[f[sl]])
        out += basis(x).T @ (area * abs(normal[f[sl],axis,None]) * speed)
    return out, int(len(t))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--anchor-source',type=Path,required=True)
    p.add_argument('--sizes', nargs='+',type=int,default=[32,64,128,256,512])
    p.add_argument('--output',type=Path, required=True)
    a=p.parse_args()
    vertices,faces=load_design(a.kit/'inputs/Dprime.txt')
    raw=(a.kit/'native/exact/report.json').read_bytes()
    if sha256(raw).hexdigest()!=REPORT_SHA: raise ValueError('trace mismatch')
    states=[r for r in json.loads(raw)['evaluations'] if 'J_tip' in r]
    qs=[np.asarray(s['q']) for s in states]
    results={'method':'three projected face families on a fixed finite cutter; squared unit-normal partition of unity; Dprime in tip frame', 'reference_sha256':REPORT_SHA, 'rows':[]}
    paths=[a.anchor_source, a.kit/'native/exact/call-006/source.json']
    hashes = ['8e6af3e44ebf1529175ab497fabd245b346519ee9aba098d0980db0afc2e497c', '26e93b0c061ac7aeceb1848de746b4e8fda584c1023d5be205acf152ba05d400']
    for n in a.sizes:
        design=[keys(*crossings(vertices,faces,k,n)[:2]) for k in range(3)]
        for state,q,path,expected in zip(states,qs,paths,hashes):
            raw_source=path.read_bytes()
            source=json.loads(raw_source)
            if sha256(raw_source).hexdigest()!=expected or source['raw_q']!=state['q'] or not source['interior_is_positive'] or any(source['face_is_cap']):
                raise ValueError('source/state mismatch or unsupported geometry')
            components=[flux(k,n,design[k],source) for k in range(3)]
            j=sum(item[0] for item in components)
            ref=np.array(state['J_tip'])
            errors={}
            for name, d in zip(('trajectory','offset','xy'), (qs[1]-qs[0],np.eye(10)[0],np.eye(10)[9])):
                v=d-q*(q@d)/(q@q)
                v/=np.linalg.norm(v)
                errors[name]=float(np.linalg.norm((j-ref)@v)/np.linalg.norm(ref@v))
            results['rows'].append(dict(side=n,state=state['index'], source_sha256=sha256(raw_source).hexdigest(),
                face_hits_per_axis=[c[1] for c in components], Jv_relative=errors,
                native_J_relative=float(np.linalg.norm(j-ref)/np.linalg.norm(ref)),
                native_izz_gradient_relative=float(np.linalg.norm((j-ref)[4]+(j-ref)[5])/np.linalg.norm(ref[4]+ref[5])),
                jacobian=j.tolist()))
            print(n,state['index'],errors,flush=True)
        a.output.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
