"""Screen registered spinning-top assets for physical pivot contact and protected carve space."""
import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path

import numpy as np

from m16_native_acceptance import require


REPRO = Path('experiments/spin_it_reproduction')
MANIFEST_SHA = {'spin2014':'d4c28d438ed4ec4bd07023bda066a35d6e38e6d0949c0ed6813148145b6f0c20',
                'faster':'6f041346f9254fcd6141a2349039ab48a7420abaffe7c7757e1da9be2c232142'}
NOODLES = Path('research/Noodles_v7_20261017')
NOODLES_SHA = {'frozen-parts-v7c/manifest.json':'67364abba55c74ccd377eef6174b8c96ee04e4af1aadbd7e505466bafabd2a3a',
               'models/complete-mm-repaired.obj':'e979bd3980a5efaf9b03d047c4e19e9296ddb426267a7322ff4e1a4b954fe80a',
               'V7_STAGE_STATUS.json':'7b292bbd57d0393b360be5ef4297fbd8ee0678ecc2939474b2102e2af69b1e01',
               'evidence/ideal-pinned-pivot-only/report.json':'9557f1e6720e25672eaaf4674b3ea1d3ffccc76f295d9ca11002b1edd4c6a45d'}


def read(root, path, expected):
    data=(root/path).read_bytes()
    require(sha256(data).hexdigest()==expected, 'INPUT_HASH_MISMATCH:'+str(path))
    return data


def obj(data):
    vertices=[]; faces=[]
    for line in data.splitlines():
        tokens=line.split()
        if tokens[:1]==[b'v']:
            vertices.append(list(map(float,tokens[1:])))
        if tokens[:1]==[b'f']:
            faces.append([int(t)-1 for t in tokens[1:]])
    v=np.asarray(vertices); f=np.asarray(faces)
    require(v.ndim==2 and f.ndim==2 and v.shape[1]==f.shape[1]==3 and f.min()>=0 and f.max()<len(v),
            'INVALID_TRIANGULAR_OBJ')
    return v,f


def point_surface_distance(p, triangles):
    """Binary64 diagnostic using closed triangle faces, including every edge and vertex."""
    a,b,c=triangles.transpose(1,0,2)
    normal=np.cross(b-a,c-a)
    norm2=(normal*normal).sum(axis=1)
    require(np.all(norm2>0),'DEGENERATE_TRIANGLE')
    h=((p-a)*normal).sum(axis=1)/norm2
    projected=p-h[:,None]*normal
    inside=np.ones(len(a),dtype=bool)
    best=float('inf')
    for u,v in ((a,b),(b,c),(c,a)):
        edge=v-u
        inside &= (np.cross(edge,projected-u)*normal).sum(axis=1)>=0
        t=np.clip(((p-u)*edge).sum(axis=1)/(edge*edge).sum(axis=1),0,1)
        best=min(best,float(np.linalg.norm(p-(u+t[:,None]*edge),axis=1).min()))
    if inside.any():
        best=min(best,float((np.abs(h[inside])*np.sqrt(norm2[inside])).min()))
    return best


def voxel_facts(asset):
    dims=tuple(reversed(asset['dimensions']))
    def decode(key):
        raw=[k for k,n in asset[key] for _ in range(n)]
        require(len(raw)==int(np.prod(dims)) and set(raw)<={0,1}, 'INVALID_VOXEL_MASK')
        return np.asarray(raw,dtype=bool).reshape(dims)
    outer=decode('material_rle');design=decode('carveable_rle')
    require(design.any() and not (design & ~outer).any(), 'INVALID_CARVE_DOMAIN')
    # Contact of closed voxel boxes across a face, edge, or vertex gives zero clearance.
    padded=np.pad(outer,1); insulated=outer.copy()
    for oz,oy,ox in itertools.product(range(3), repeat=3):
        insulated &= padded[oz:oz+dims[0],oy:oy+dims[1],ox:ox+dims[2]]
    lo=np.asarray(asset['minimum']); step=(np.asarray(asset['maximum'])-lo)/asset['dimensions']
    bottom=float(lo[2]+np.where(outer)[0].min()*step[2])
    exposed=int((design & ~insulated).sum())
    return dict(outer_cells=int(outer.sum()), design_cells=int(design.sum()), outer_min_z=bottom,
                registered_origin_below_outer_by_at_least=bottom,design_cells_touching_exterior=exposed,
                minimum_clearance_lower_bound=float(step.min()) if exposed==0 else 0.)


def screen(tree,main_tree):
    paper={}
    for source,manifest_sha in MANIFEST_SHA.items():
        directory=REPRO/source; path=directory/'paper_cases.json'
        manifest=json.loads(read(tree,path,manifest_sha));rows=[]
        for case in manifest['cases']:
            if case['mode']!='SPINNING_TOP':
                continue
            row=dict(case=case['id'],classification=case['asset_class'])
            if source=='spin2014':
                require(case['target_axis']==[0,0,1] and 'target_origin' not in case, 'MOVED_2014_FRAME')
                description=case['occupancy']; asset=directory/description['path']
                row.update(file=str(asset),sha256=description['sha256'],**voxel_facts(json.loads(read(tree,asset,description['sha256']))))
            else:
                p=np.asarray(case['target_origin']);axis=np.asarray(case['target_axes'])[:,2]
                row.update(target_origin=p.tolist(), axis=axis.tolist())
                for key in ('outer','design_domain'):
                    asset=directory/case[key]['path']; asset_sha=case[key]['sha256']
                    vertices,faces=obj(read(tree,asset,asset_sha))
                    row[key]=dict(file=str(asset),sha256=asset_sha,vertices=len(vertices),faces=len(faces),
                                  pivot_to_surface_distance=round(point_surface_distance(p,vertices[faces]),12),
                                  minimum_axial_projection=round(float(((vertices-p)@axis).min()),12))
            rows.append(row)
        paper[source]=dict(manifest=str(path),sha256=manifest_sha,rows=rows)
    content={k:read(main_tree,NOODLES/k,v) for k,v in NOODLES_SHA.items()}
    frozen=json.loads(content['frozen-parts-v7c/manifest.json']);spec=frozen['spec']; chain=frozen['construction']
    stage=json.loads(content['V7_STAGE_STATUS.json'])
    pinned=json.loads(content['evidence/ideal-pinned-pivot-only/report.json'])
    v,f=obj(content['models/complete-mm-repaired.obj'])
    target=np.asarray(spec['target_origin_model'])*(spec['model_unit_m']*1000)
    require(spec['target_axis']==[0,0,1] and np.array_equal(target,np.asarray(pinned['tip_m'])*1000)
            and 'pin-fixed' in chain['fixed_parts'] and stage['final_exported_OBJ_sha256']==NOODLES_SHA['models/complete-mm-repaired.obj'],
            'NOODLES_REGISTRATION_DRIFT')
    tip=v[v[:,2]==v[:,2].min()]
    valid=len(tip)==1 and np.array_equal(tip[0],target)
    return dict(schema='w2-m31-registered-candidate-contact-screen-v1',paper_cases=paper,
                spin2014_registered_origin=[0,0,0],spin2014_pivot_contact_count=sum(
                    r['outer_min_z']==0 for r in paper['spin2014']['rows']),
                faster_proxy_pivot_contact_count=sum(r['outer']['pivot_to_surface_distance']==0 for r in paper['faster']['rows']),
                noodles_v7=dict(root=str(NOODLES),sha256=NOODLES_SHA,registered_tip_mm=target.tolist(),
                    archived_terminal_outer_tip_matches_unique_lowest_vertex=valid,vertices=len(v),faces=len(f),
                    fixed_part='pin-fixed',permanent_ellipsoid_shell=chain['permanent_ellipsoid_shell'],
                    original_variable_construction=chain['variable'],original_fixed_construction=chain['fixed'],
                    global_positive_clearance_certified=False,classification=spec['classification'],
                    center_of_mass_weight=spec['center_of_mass_weight'],historical_status=stage['warm_restart_status'],
                    separate_binary64_partition_passed=stage['formal_separate_F_D_A_binary64_partition_passed']),
                measurement='integer voxel occupancy, binary64 proxy triangle distances, exact recorded binary64 V7 tip equality',
                geometry_redesigns=0,new_native_calls=0,new_solver_runs=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--w2-tree',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--main-tree',type=Path,required=True,help='Checkout containing the pinned historical Noodles V7 final OBJ')
    args=parser.parse_args()
    print(json.dumps(screen(args.w2_tree,args.main_tree),indent=2,allow_nan=False,sort_keys=True))
