"""Check exact nesting, winding and embedding of the complete saved terminal Heart mesh."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from independent_reference import _triple
from m16_native_acceptance import REQUEST_SHA, file_hash, h256, require, strict_json
from m22_point0_operand_replay import cross, dot, exact_point, sub
from m22_point0_solid_boundaries import CHECKER_SHA, audit_embedding
from native_cherchi_teacher import digest


def winding(point, triangles):
    """Exact projected-ray winding. Retry the two other axes on edge hits and coplanarity."""
    for axis in range(3):
        u,v=(axis+1)%3,(axis+2)%3
        total=0
        for tri in triangles:
            if max(p[axis] for p in tri)<point[axis] or any(
                    not min(p[k] for p in tri)<=point[k]<=max(p[k] for p in tri) for k in (u,v)):
                continue
            n=cross(sub(tri[1],tri[0]),sub(tri[2],tri[0]))
            side=dot(n,sub(point,tri[0]))
            if not n[axis]:
                if not side:
                    break
                continue
            if side*n[axis]>0:
                continue
            turns=[((b[u]-a[u])*(point[v]-a[v])-(b[v]-a[v])*(point[u]-a[u]))*n[axis]
                   for a,b in zip(tri,tri[1:]+tri[:1])]
            if min(turns)<0:
                continue
            if not side or not min(turns):
                break
            total+=1 if n[axis]>0 else -1
        else:
            return total
    return None


def signed_components(vertices, faces):
    """Run after exact mutual disjointness and closed oriented manifold checks."""
    neighbours=[set() for _ in vertices]
    incident=[[] for _ in vertices]
    for i,(a,b,c) in enumerate(faces):
        neighbours[a].update((b,c))
        neighbours[b].update((c,a))
        neighbours[c].update((a,b))
        incident[a].append(i)
    remaining=set(range(len(vertices)))
    components=[]
    while remaining:
        start=min(remaining)
        remaining.remove(start)
        stack=[start]
        indices=[]
        edge_count=0
        incident_faces=[]
        while stack:
            a=stack.pop()
            indices.append(a)
            edge_count+=len(neighbours[a])
            incident_faces.extend(incident[a])
            new=neighbours[a]&remaining
            remaining.difference_update(new)
            stack.extend(new)
        triangles=[tuple(vertices[i] for i in faces[j]) for j in sorted(incident_faces)]
        volume=sum((_triple(*t) for t in triangles),Fraction())/6
        euler=len(indices)-edge_count//2+len(triangles)
        require(triangles and volume and euler%2==0 and euler<=2, 'INVALID_CONNECTED_SHELL')
        components.append((triangles,volume,dict(first_vertex=start,vertices=len(indices),edges=edge_count//2,
             faces=len(triangles),genus=(2-euler)//2,signed_volume=float(volume),volume_exact_sha256=h256(str(volume).encode()))))
    for i,(triangles,volume,summary) in enumerate(components):
        containers=[]
        for j,(other,other_volume,_) in enumerate(components):
            if i==j:
                continue
            count=None
            for tri in triangles[:16]:
                for weights in ((1,1,1),(1,2,4),(3,7,11)):
                    anchor=tuple(Fraction(dot(weights,col),sum(weights)) for col in zip(*tri))
                    count=winding(anchor,other)
                    if count is not None:
                        break
                if count is not None:
                    break
            require(count is not None and count in (0, 1 if other_volume>0 else -1),'AMBIGUOUS_SHELL_CONTAINMENT')
            if count:
                containers.append(j)
        require((volume>0)==(len(containers)%2==0),'WRONG_ORIENTATION_FOR_NESTING_DEPTH')
        summary.update(containing_shells=containers,nesting_depth=len(containers))
    return [summary for _,_,summary in components], [volume for _,volume,_ in components]


def audit_terminal(directory, public_result_sha, packet_sha, checker):
    require(file_hash(directory/'result.json')==public_result_sha and
            file_hash(directory/'terminal-complement.json')==packet_sha,'PUBLIC_RECEIPT_OR_PACKET_DRIFT')
    receipt=strict_json((directory/'result.json').read_bytes())
    packet=strict_json((directory/'terminal-complement.json').read_bytes())
    require(receipt['schema']=='w2-m30-trajectory-v1' and receipt['status']=='NATIVE_TERMINAL' and
            receipt['native_terminal']==receipt['native_terminal_summary']['status']=='CONVERGED' and
            packet['schema']=='w2-m30-converged-complement-v1' and packet['result_sha256']==public_result_sha and
            receipt['run_id']==packet['run_id']==directory.name,'NO_MATCHING_CONVERGED_TRAJECTORY')
    require(file_hash(directory/'request.json')==receipt['file_sha256']['request.json']==REQUEST_SHA and
            all(file_hash(directory/k)==receipt['file_sha256'][k] for k in ('evaluations.jsonl','events.jsonl')),
            'TRAJECTORY_EVIDENCE_DRIFT')
    request=strict_json((directory/'request.json').read_bytes())['request']
    events=[strict_json(line) for line in (directory/'events.jsonl').read_bytes().splitlines()]
    terminal=events[-1]
    idx=packet['evaluation_index']
    require(terminal['type']=='attempt_terminal' and terminal['reason']=='converged' and
            terminal['evaluation_index']==idx, 'NATIVE_TERMINAL_IDENTITY_DRIFT')
    accepted=[e for e in events if e['type']=='evaluation' and e['outcome']=='accepted' and e['restart']==terminal['restart']]
    require(accepted and accepted[-1]['evaluation_index']==idx,'MISSING_TERMINAL_ACCEPTANCE')
    with (directory/'evaluations.jsonl').open('rb') as f:
        row=next((r for r in map(strict_json,f) if r['index']==idx),None)
    require(row and row['status']=='returned', 'MISSING_TERMINAL_MATERIAL')
    kept=row['material']; carved=packet['complement']; mesh=packet['final_mesh']
    require(row['material_sha256']==packet['retained_material_sha256']==digest(kept) and
            kept['quadric_source_identity']==carved['quadric_source_identity'] and
            kept['evaluation_map_identity']==carved['evaluation_map_identity'] and
            kept['quadric_coefficients']==carved['quadric_coefficients']==accepted[-1]['quadric_coefficients']
            ==receipt['native_terminal_summary']['quadric_coefficients'] and
            {kept['boolean_operation'],carved['boolean_operation']}=={'subtraction','intersection'},'WRONG_TERMINAL_PAIR')
    require(request['target_axes']==[[1,0,0],[0,1,0],[0,0,1]] and mesh['frame']=='M23_INPUT_NORMALIZED_TARGET',
            'UNSUPPORTED_TARGET_FRAME')
    vertices=[exact_point(p) for p in mesh['vertices']]
    n=len(request['outer_vertices']); m=len(request['outer_faces'])
    origin=tuple(map(Fraction,request['target_origin']))
    require(packet['outer_vertex_count']==n and packet['outer_face_count']==m and
            vertices[:n]==[sub(tuple(map(Fraction,p)),origin) for p in request['outer_vertices']] and
            vertices[n:]==[exact_point(p) for p in carved['variable_rational_vertices']] and
            mesh['faces'][:m]==request['outer_faces'] and mesh['faces'][m:]==[
                [n+c,n+b,n+a] for a,b,c in carved['variable_faces']],'ASSEMBLED_MESH_DRIFT')
    source=checker.read_bytes()
    require(h256(source)==CHECKER_SHA,'WRONG_CHECKER_SOURCE')
    with TemporaryDirectory() as directory_name:
        tmp=Path(directory_name); cpp=tmp/'checker.cpp'; cpp.write_bytes(source); binary=tmp/'checker'
        subprocess.run(['c++','-std=c++17','-O2',str(cpp),'-o',str(binary),'-lgmpxx','-lgmp'],check=True)
        embedding=audit_embedding(binary,vertices,mesh['faces'],600)
    components,volumes=signed_components(vertices,mesh['faces'])
    require(components[0]['first_vertex']==0 and components[0]['vertices']==n and components[0]['faces']==m and
            components[0]['nesting_depth']==0 and all(0 in c['containing_shells'] for c in components[1:]),
            'CAVITY_COMPONENT_OUTSIDE_FROZEN_OUTER')
    total=sum(volumes,Fraction())
    return dict(schema='w2-m30-terminal-complete-surface-v1',run_id=receipt['run_id'],result_sha256=public_result_sha,
                terminal_packet_sha256=packet_sha,evaluation_index=idx,checker_source_sha256=CHECKER_SHA,
                whole_mesh=dict(vertices=len(vertices),faces=len(mesh['faces']),components=components,
                                positive_volume=float(total),volume_exact_sha256=h256(str(total).encode()),**embedding),
                new_native_boolean_calls=0,whole_boolean_set_equality_verified_by_this_check=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('public_result_sha256')
    parser.add_argument('packet_sha256')
    parser.add_argument('checker_source',type=Path)
    args=parser.parse_args()
    sys.set_int_max_str_digits(100000)
    print(json.dumps(audit_terminal(args.directory,args.public_result_sha256,args.packet_sha256,args.checker_source),indent=2))
