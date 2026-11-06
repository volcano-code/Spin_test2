"""Verify archived native intersection meshes as regular solids on both input carriers."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
from math import lcm
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from .heart_point0_seams import (anchor_check, cross, cycles, dot, frac, input_seams, normal,
                                 observed_seams, winding)
from .heart_point0_solid import export_sources, ROOT, TEACHER, PERFORMANCE
from .real_h8e_archive_m8 import canonical_digest, read_json, require

W1 = '15db682052e9a97779c34ba02000095b78cf4b91'
W2 = 'd69e187b3f93494b3cb0ca85243c1e3ea5a591e4'
CHECKER = '2536f7783ff6552ac930c294879e69b8430fc8e527873859b3ec1187d39c6ea0'
REQUEST = '7084fecb6bbd4d17bca842bfe68299f9711ecead09eb01593364424a3e0c4a8a'


def hash_file(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def surface(binary, vertices, faces):
    """Exact embedding, outward component orientation and disjoint solid interiors."""
    rows = [f'{len(vertices)} {len(faces)}', *(' '.join(map(str,p)) for p in vertices),
            *(' '.join(map(str,f)) for f in faces)]
    cp = subprocess.run([str(binary), '--max-seconds', '120'], input=('\n'.join(rows)+'\n').encode(),
                        capture_output=True, check=True, timeout=150)
    status = dict(line.split('=', 1) for line in cp.stdout.decode().splitlines())
    require(status['status'] == 'EXACT_POST_BOOLEAN_SELF_INTERSECTION_FREE'
            and int(status['exact_pairs_checked']) == int(status['candidate_pairs'])
            and int(status['faces']) == len(faces) and int(status['vertices']) == len(vertices),
            'incomplete embedding check')
    volumes=component_volumes(vertices,faces)
    return dict(vertices=len(vertices),faces=len(faces),components=len(volumes),
                exact_pairs_checked=int(status['exact_pairs_checked']), signed_volumes=list(map(float,volumes)))


def component_volumes(vertices, faces):
    """Check orientations and inter-component containment after the embeddedness test."""
    neighbours = [set() for _ in vertices]
    for a,b,c in faces:
        neighbours[a].update((b,c)); neighbours[b].update((a,c)); neighbours[c].update((a,b))
    owners = {}
    components = defaultdict(list)
    for i in range(len(vertices)):
        if i in owners:
            continue
        todo = [i]
        while todo:
            v = todo.pop()
            if v not in owners:
                owners[v] = i
                todo.extend(w for w in neighbours[v] if w not in owners)
    for a,b,c in faces:
        components[owners[a]].append((vertices[a],vertices[b],vertices[c]))
    volumes = [sum(dot(t[0],cross(t[1],t[2])) for t in soup)/6 for soup in components.values()]
    require(all(v>0 for v in volumes), 'nonpositive oriented component')
    for i, other in components.items():
        norms = list(map(normal,other))
        boxes = [(max(x),min(y),max(y),min(z),max(z)) for x,y,z in (zip(*tri) for tri in other)]
        for j, shell in components.items():
            if i == j:
                continue
            # The globally embedded components cannot cross. Any nonsingular
            # point on the other shell has the same winding about this one.
            found = None
            for tri in shell:
                p = tuple(sum(col)/3 for col in zip(*tri))
                found = winding(p,other,norms,boxes)
                if found is not None:
                    break
            require(found == 0, 'nested shells or no unambiguous containment probe')
    return volumes


def verify_regions(req, material, owners, binary, design_surface):
    require(material['boolean_operation']=='intersection','only regularized intersection is supported')
    coords=[req['design_vertices'],material['quadric_source_vertices']]
    faces=[req['design_faces'],material['quadric_source_faces']]
    rational=[[tuple(map(F,p)) for p in mesh] for mesh in coords]
    scale=lcm(*(x.denominator for mesh in rational for p in mesh for x in p))
    soups=[]
    for mesh,fs in zip(rational,faces):
        lattice=[tuple(int(x*scale) for x in p) for p in mesh]
        soups.append([tuple(lattice[i] for i in f) for f in fs])
    original=[np.asarray(v,dtype=np.float64)[fs] for v,fs in zip(coords,faces)]
    d,c=soups
    dn,cn=[list(map(normal,t)) for t in soups]
    expected,candidates=input_seams(d,c,dn,cn,original)
    # input_seams orients for subtraction. Intersection keeps cutter interior
    # on the design side, reversing the expected directed boundary.
    expected={key:(right,left) for key,(left,right) in expected.items()}
    out=[tuple(map(frac,p)) for p in material['material_proof']['exact_vertices']]
    scaled=[tuple(x*scale for x in p) for p in out]
    saved=material['variable_faces']
    observed=observed_seams(saved,owners,scaled)
    require(expected and observed==expected,'oriented seam mismatch or empty seam')
    cutter=surface(binary,rational[1],faces[1])
    output=surface(binary,out,saved)
    b=anchor_check(d,c,dn,cn,scaled,saved,owners)
    a=anchor_check(c,d,cn,dn,scaled,saved,[(1-o,j) for o,j in owners])
    require(sum(a.values())==len(d) and sum(b.values())==len(c),'missing carrier anchors')
    return dict(design=design_surface,cutter=cutter,output=output,input_candidate_pairs=candidates,
                exact_directed_seams=len(expected), loops=cycles(list(observed.values())),
                anchors=dict(design_inside=a[1],design_outside=a[0],cutter_inside=b[1],cutter_outside=b[0]),
                regularized_intersection_verified=True)


def audit(folder, result_sha, indices):
    require(len(set(indices))==len(indices) and indices and min(indices)>0,'invalid selected indices')
    require(hash_file(folder/'result.json')==result_sha,'public receipt mismatch')
    result=read_json((folder/'result.json').read_bytes())
    for name in ('request.json','evaluations.jsonl'):
        require(hash_file(folder/name)==result['file_sha256'][name],'input file mismatch')
    req=read_json((folder/'request.json').read_bytes())['request']
    require(canonical_digest(req)==result['request_sha256']==REQUEST,'wrong design request')
    audited={}
    with TemporaryDirectory() as temp:
        tmp=Path(temp); perf=export_sources(ROOT,W1,PERFORMANCE,tmp); teacher=export_sources(ROOT,W2,TEACHER,tmp)
        sys.path.insert(0,str(teacher))
        try:
            from m22_point0_operand_replay import audit_geometry
            source=perf/'m26_exact_intersection.cpp'
            require(hash_file(source)==CHECKER,'unexpected GMP checker')
            binary=tmp/'checker'
            subprocess.run(['c++','-std=c++17','-O2',str(source),'-o',str(binary),'-lgmpxx','-lgmp'],check=True)
            design=surface(binary,[tuple(map(F,p)) for p in req['design_vertices']],req['design_faces'])
            with (folder/'evaluations.jsonl').open() as f:
                for line in f:
                    row=json.loads(line)
                    if row['index'] not in indices: continue
                    require(row['index'] not in audited and row['status']=='returned'
                            and canonical_digest(row['material'])==row['material_sha256'],'wrong native material')
                    m=row['material']
                    counts,support,owners=audit_geometry(req,m)
                    require(sum(support.values())==len(owners)==len(m['variable_faces'])
                            and set(support)<={'design_same','quadric_same'},'intersection winding mismatch')
                    audited[row['index']]=dict(material_sha256=row['material_sha256'],constructions=counts,
                            face_winding=support,**verify_regions(req,m,owners,binary,design))
        finally:
            sys.path.remove(str(teacher))
    require(set(indices)==set(audited),'missing endpoint evaluation')
    return dict(run_id=result['run_id'],public_result_sha256=result_sha,evaluations=audited,
                native_calls_in_replay=0,full_mesh_of_top_verified=False)


if __name__ == '__main__':
    sys.set_int_max_str_digits(100000)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path); p.add_argument('public_result_sha256')
    p.add_argument('evaluation_indices',nargs='+',type=int)
    args=p.parse_args()
    print(json.dumps(audit(args.folder,args.public_result_sha256,args.evaluation_indices),indent=2))
