"""Check frozen Ellipsoid occupied cells and OBJ edges against strict hollowing input gates."""
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT/'experiments/spin_it_reproduction'
ASSET_SHA = '6036f03ff83777a636cbb8cf67bd57675caa44b90c40b3d2e442f5fb27d85983'
MESH_SHA = {'outer.obj':'79c635bdef17b6e95ce0c7436bf24db238f65d26688eaee85f021b071585c45a',
            'design.obj':'bf1471cac1bf4a8e1781675957186e28838eab55f5fff9a6a01d9a9db2e274c1',
            'diagnostics.json':'7fb4d52e920188b710af8e5999e594f086f92d20714db328d3be8f6967e64728'}


def pinned(path, expected):
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError('Input SHA256 mismatch: '+path.name)
    return data


def cells(runs):
    flat = [bit for bit, count in runs for _ in range(count)]
    if len(flat) != 16**3 or set(flat) != {0,1}:
        raise ValueError('Unexpected occupied-cell encoding')
    return {(i%16, i//16%16, i//256) for i,bit in enumerate(flat) if bit}


def shifted(point, axis, shift):
    result = list(point)
    result[axis] += shift
    return tuple(result)


def cell_surface(mask):
    edges = Counter()
    for p in mask:
        for i,sign in product(range(3), (-1,1)):
            if shifted(p,i,sign) in mask:
                continue
            a = shifted(p,i,int(sign==1))
            b = shifted(a,(i+1)%3,1)
            d = shifted(a,(i+2)%3,1)
            c = shifted(b,(i+2)%3,1)
            for start,end in ((a,b),(b,c),(c,d),(d,a)):
                edges[tuple(sorted((start,end)))] += 1
    return edges


def obj_lattice_edges(data):
    vertices, edges = [], set()
    for line in data.decode('ascii').splitlines():
        op,*fields = line.split()
        if op=='v':
            v=[8*(Fraction(x)+offset) for x,offset in zip(fields,(1,1,0))]
            if len(v)!=3 or any(x.denominator!=1 for x in v):
                raise ValueError('Off-grid vertex')
            vertices.append(tuple(map(int,v)))
        elif op=='f':
            triangle=[vertices[int(i)-1] for i in fields]
            if len(triangle)!=3:
                raise ValueError('Expected triangle')
            for a,b in zip(triangle,triangle[1:]+triangle[:1]):
                if sum(abs(u-v) for u,v in zip(a,b))==1:
                    edges.add(tuple(sorted((a,b))))
        else:
            raise ValueError('Unexpected OBJ record')
    return edges


def coordinate(point):
    return [str(Fraction(x,8)-offset) for x,offset in zip(point,(1,1,0))]


def verify(root=EXPERIMENT):
    base = root/'viewer/spin2014/ellipsoid-top/repeat-1'
    registry=json.loads((root/'spin2014/paper_cases.json').read_bytes())
    case=next(c for c in registry['cases'] if c['id']=='ellipsoid-top')
    if case['occupancy']['sha256']!=ASSET_SHA or case['target_axis']!=[0,0,1] or case['author_exact'] is not False:
        raise ValueError('Wrong model registration')
    asset=json.loads(pinned(root/'spin2014'/case['occupancy']['path'],ASSET_SHA))
    if asset['dimensions']!=[16]*3 or asset['minimum']!=[-1,-1,0] or asset['maximum']!=[1,1,2]:
        raise ValueError('Unexpected grid frame')
    files={name:pinned(base/name,h) for name,h in MESH_SHA.items()}
    diag=json.loads(files['diagnostics.json'])
    if diag['target_origin_model']!=[0,0,0] or diag['target_axes_model']!=[[1,0,0],[0,1,0],[0,0,1]]:
        raise ValueError('Target changed')
    outer,design=cells(asset['material_rle']),cells(asset['carveable_rle'])
    if not design < outer:
        raise ValueError('Invalid design mask')
    mesh_edges={name:obj_lattice_edges(files[name+'.obj']) for name in ('outer','design')}
    for name,mask in [('outer',outer),('design',design)]:
        surface=cell_surface(mask)
        if set(surface)!=mesh_edges[name] or any(v!=2 for v in surface.values()):
            raise ValueError('Voxel to OBJ edge mismatch or nonmanifold input: '+name)
    common=sorted(mesh_edges['outer'] & mesh_edges['design'])
    fixed=cell_surface(outer-design)
    bottom=min(p[2] for p in outer)
    patch=[p for p in outer if p[2]==bottom]
    on_axis=any(p[0] in (7,8) and p[1] in (7,8) for p in patch)
    if not common or not on_axis or bottom<=0:
        raise ValueError('Frozen geometric witnesses changed')
    return dict(schema='w2-m31-ellipsoid-strict-input-preflight-v1',case='ellipsoid-top',status='STRICT_INPUT_BLOCKED',
                # The same outward voxel faces can be triangulated with different internal diagonals.
                input_sha256={'occupancy':ASSET_SHA,**MESH_SHA},source_w2='1cdf704660c0ca5c70714915b4e2dae53e409715',
                outer_cells=len(outer),design_cells=len(design),fixed_cells=len(outer-design),
                common_design_outer_lattice_edges=len(common),sample_common_edge_model=[coordinate(p) for p in common[0]],
                exact_whole_design_clearance_model='0',fixed_boundary_four_face_edges=sum(v==4 for v in fixed.values()),
                target_origin=diag['target_origin_model'],target_axis=case['target_axis'],on_axis_bottom_z_model=str(Fraction(bottom,8)),
                lowest_horizontal_surface_area_model=str(Fraction(len(patch),64)),target_to_outer_distance_model=str(Fraction(bottom,8)),
                blockers=['POSITIVE_WHOLE_DESIGN_WALL_IMPOSSIBLE','NONMANIFOLD_FIXED_MASK_BOUNDARY',
                          'FROZEN_TARGET_ORIGIN_OUTSIDE_SOLID'],
                native_request_emitted=False,native_boolean_calls=0,solver_calls=0)


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
