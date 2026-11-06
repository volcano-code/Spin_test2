"""Verify the regularized intersection of registered M32 D' and a star-shaped polyhedral cutter."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import numpy as np

from m28_cut_cell_coverage import contains, edges
from m29_heart_cut_faces import area2, cut_polygon, dot, homogeneous, plane, projected
from m34_noodles_partition import Q, exact_repr, exact_ten, mesh_stream, regular_orientation, sha

DESIGN_SHA = 'c3633c74bc74c2be488eb22de108454276f28941f0477cfee5837f57db99d622'


def packet(path):
    raw = path.read_bytes()
    lines = raw.splitlines()
    nv,nf = map(int,lines[0].split())
    if len(lines)!=1+nv+nf: raise ValueError('bad operand length')
    vertices = [tuple(Q(x.decode()) for x in line.split()) for line in lines[1:1+nv]]
    faces = [tuple(map(int,line.split())) for line in lines[1+nv:]]
    return sha(raw),vertices,faces


def construction_signature(faces, owners, nd, nv):
    """Describe vertices by incident *input parent facets*, independent of CGAL output indices."""
    stars=[set() for _ in range(nv)]; side={}
    for f,b in zip(faces,owners):
        for v in f:stars[v].add(b)
        for a,c in edges(f):side.setdefault(tuple(sorted((a,c))),set()).add(b>=nd)
    ids=[tuple(sorted(v)) for v in stars]
    if any(len(v)<3 for v in ids) or len(ids)!=len(set(ids)):
        return dict(unambiguous_vertex_stars=False)
    def digest(obj):return sha(json.dumps(obj,separators=(',',':')).encode())
    geom=[]
    for f,parent in zip(faces,owners):
        tr=tuple(ids[j] for j in f)
        tr=min(tr[k:]+tr[:k] for k in range(3))
        geom.append((parent,tr))
    seam=sorted(tuple(sorted((ids[a],ids[b]))) for (a,b),owners2 in side.items() if len(owners2)==2)
    return dict(unambiguous_vertex_stars=True,construction_sha256=digest(sorted(ids)),
                seam_sha256=digest(seam),oriented_triangulation_sha256=digest(sorted(geom)))


def seam_loops(faces, owners, nf):
    edge_sides=defaultdict(list)
    for f,o in zip(faces, owners):
        for a,b in edges(f):edge_sides[tuple(sorted((a,b)))].append(o>=nf)
    graph=defaultdict(set)
    for (a,b),sides in edge_sides.items():
        if len(sides)!=2:raise ValueError('open or nonmanifold output')
        if sides[0]!=sides[1]:
            graph[a].add(b);graph[b].add(a)
    if not graph or any(len(g)!=2 for g in graph.values()): raise ValueError('unresolved crossing graph')
    visited=set(); lengths=[]
    for v in graph:
        if v in visited:continue
        todo=[v]; old=len(visited)
        while todo:
            u=todo.pop()
            if u not in visited:
                visited.add(u);todo.extend(graph[u]-visited)
        lengths.append(len(visited)-old)
    return sorted(lengths)


def radial_tetrahedra(vertices, faces):
    """A closed embedded oriented cutter, seen from a strict kernel point,
    is a multiplicity-one union of its facet-to-kernel tetrahedra."""
    center=tuple(sum(coord)/len(vertices) for coord in zip(*vertices))
    hc=homogeneous(center)
    parts=[];boxes=[]
    for f in faces:
        a,b,c=(vertices[i] for i in f)
        ps=[plane(tri) for tri in ((a,b,c),(center,b,a),(center,a,c),(center,c,b))]
        if dot(ps[0],hc)>=0:raise ValueError('kernel is not behind every outer plane')
        bbox=np.array([a,b,c,center],dtype=float)
        parts.append(ps);boxes.append((bbox.min(axis=0),bbox.max(axis=0)))
    lo,hi=np.asarray(boxes).transpose(1,0,2)
    return parts,lo,hi


def intersection_area(tri, axis, parts, lo, hi):
    pts=np.asarray(tri,dtype=float); htri=list(map(homogeneous,tri))
    candidates=np.flatnonzero(np.all((hi>=pts.min(axis=0)) & (lo<=pts.max(axis=0)),axis=1))
    value=0
    for j in candidates:
        ps=parts[j]
        if any(all(dot(p,x)==0 for x in htri) for p in ps):
            raise ValueError('coplanar source/fan face needs independent overlay')
        poly=cut_polygon(tri,htri,ps,rational_type=Q)
        if poly:value+=area2([projected(pt,axis) for pt in poly])
    return value


def strict_source_position(vd,fd,vc,fc):
    """Sufficient open-neighborhood test: disjoint AABBs, or strict vertex/face
    and edge/edge orient3d signs for every possible cross-operand triangle pair."""
    dc=np.asarray(vd,dtype=float)[np.asarray(fd)]
    dlo,dhi=dc.min(axis=1),dc.max(axis=1)
    dh=list(map(homogeneous,vd)); ch=list(map(homogeneous,vc))
    dp={}; edge_plane={}; pairs=0; zeros=0; signs=0
    for cf in fc:
        ct=[vc[i] for i in cf]
        cf32=np.asarray(ct,dtype=float)
        cplane=plane(ct)
        ix=np.flatnonzero(np.all((dhi>=cf32.min(axis=0))&(dlo<=cf32.max(axis=0)),axis=1))
        for i in ix:
            dt=fd[i]; pairs+=1
            if i not in dp:dp[i]=plane([vd[k] for k in dt])
            ds=[dot(dp[i],ch[k]) for k in cf]
            cs=[dot(cplane,dh[k]) for k in dt]
            # A strictly separating plane suffices. Other coplanar extensions
            # do not affect the positive separation of closed triangles.
            if min(ds)>0 or max(ds)<0 or min(cs)>0 or max(cs)<0:continue
            zeros+=sum(not x for x in ds+cs);signs+=6
            for a,b in edges(dt):
                ab=tuple(sorted((a,b)))
                for c,e in edges(cf):
                    key=ab+(c,)
                    if key not in edge_plane:
                        tri=(vd[ab[0]],vd[ab[1]],vc[c])
                        try:edge_plane[key]=plane(tri)
                        except ValueError:edge_plane[key]=None
                    eq=edge_plane[key];signs+=1
                    zeros+=eq is None or not dot(eq,ch[e])
    return dict(input_pair_AABB_candidates=pairs,strict_feature_signs_tested=signs,
                zero_feature_predicates=zeros,locally_transverse_polyhedra=(zeros==0))


def parent_patches(vd,fd,vc,fc,vo,fo,owners):
    """Precondition: the saved output is embedded (hence no overlapping child interiors)."""
    nf=len(fd)
    if len(owners)!=len(fo) or min(owners)<0 or max(owners)>=nf+len(fc):
        raise ValueError('invalid birth face indices')
    parts,lo,hi=radial_tetrahedra(vc,fc)
    ho=[homogeneous(v) for v in vo]
    # Conversion is monotone; inclusive bounding boxes with exact extrema on
    # both sides make the floating broad phase conservative.
    tv=np.array(vd,dtype=float)[np.asarray(fd)];c=np.array(vc,dtype=float)
    ids=set(map(int,np.flatnonzero(np.all((tv.max(axis=1)>=c.min(axis=0)) &
                                         (tv.min(axis=1)<=c.max(axis=0)),axis=1))))
    groups=defaultdict(list)
    for f,key in zip(fo,owners):groups[key].append(f)
    if not {i for i in groups if i<nf} <= ids:raise ValueError('unreachable design parent')
    design_partial=0;design_nonempty=0;cutter_partial=0;cutter_nonempty=0
    for i in sorted(ids | {key for key in groups if key>=nf}):
        inp=[vd[k] for k in fd[i]] if i<nf else [vc[k] for k in fc[i-nf]]
        p=plane(inp); axis=max(range(3),key=lambda k:abs(p[k]))
        parent=[projected(pt,axis) for pt in inp]
        whole=area2(parent)
        saved=0
        for face in groups.get(i,[]):
            if any(dot(p,ho[k])!=0 for k in face):raise ValueError('child off parent plane')
            tr=[projected(vo[k],axis) for k in face]
            signed=area2(tr)
            if signed*whole<=0 or not all(contains(parent,pt) for pt in tr):
                raise ValueError('child outside parent or winding changed')
            if i<nf and intersection_area([vo[k] for k in face],axis,parts,lo,hi)!=signed:
                raise ValueError('design child protrudes outside real cutter')
            saved+=signed
        if i<nf:
            actual=intersection_area(inp,axis,parts,lo,hi)
            if actual!=saved: raise ValueError(f'design parent area mismatch {i}')
            design_nonempty+=bool(actual);design_partial+=0<abs(actual)<abs(whole)
        else:
            if abs(saved)>abs(whole):raise ValueError('duplicate cutter coverage')
            cutter_nonempty+=bool(saved);cutter_partial+=0<abs(saved)<abs(whole)
    if not (design_partial and cutter_partial):raise ValueError('no positive-area partial patch on each carrier')
    return dict(design_parent_count=nf,design_AABB_candidates=len(ids),
                nonempty_design_parents=design_nonempty,partial_design_parents=design_partial,
                nonempty_cutter_parents=cutter_nonempty,partial_cutter_parents=cutter_partial,
                saved_design_faces=sum(i<nf for i in owners),saved_cutter_faces=sum(i>=nf for i in owners),
                exact_seam_loop_lengths=seam_loops(fo,owners,nf))


def audit(folder, design, ten_binary, embedding_binary, evidence, full):
    raw=(folder/'result.json').read_bytes()
    j=json.loads(raw)
    if j['operation']!='intersection' or not j['succeeded']:raise ValueError('expected completed intersection')
    v,f=j['vertices'], list(map(tuple,j['faces']))
    if not v or not f:raise ValueError('no crossing result')
    pdigest,vc,fc=packet(folder/'cutter.txt')
    receipt=json.loads((folder/'receipt.json').read_text())
    if (receipt['result_sha256']!=sha(raw) or receipt['cutter_sha256']!=pdigest or
        receipt['design_sha256']!=DESIGN_SHA or sha(design.read_bytes())!=DESIGN_SHA or
        receipt['operation']!='intersection'):
        raise ValueError('W2 source/operation receipt mismatch')
    report=dict(sample=folder.name,output_sha256=sha(raw),cutter_sha256=pdigest,design_sha256=DESIGN_SHA,
                vertices=len(v),faces=len(f),construction=construction_signature(f,j['birth_faces'],94150,len(v)))
    nums=exact_ten(ten_binary,v,f)
    data=('\n'.join(exact_repr(nums))+'\n').encode('ascii');name=folder.name+'.ten'
    evidence.mkdir(parents=True,exist_ok=True);(evidence/name).write_bytes(data)
    report.update(signed_ten=list(map(float,nums)),exact_ten_sha256=sha(data))
    if full:
        dd,vd,fd=packet(design)
        if dd!=DESIGN_SHA:raise ValueError('M32 design operand changed')
        proc=subprocess.run([str(embedding_binary),'--max-seconds','900'],input=mesh_stream(v,f),
                            capture_output=True,check=True,timeout=940)
        fields=dict(s.split('=',1) for s in proc.stdout.decode().splitlines() if '=' in s)
        if (fields.get('status')!='EXACT_POST_BOOLEAN_SELF_INTERSECTION_FREE' or
            fields['candidate_pairs']!=fields['exact_pairs_checked']):raise ValueError('embedding check failed')
        cproc=subprocess.run([str(embedding_binary),'--max-seconds','120'],
                             input=(folder/'cutter.txt').read_bytes(),capture_output=True,check=True)
        if b'status=EXACT_POST_BOOLEAN_SELF_INTERSECTION_FREE' not in cproc.stdout:
            raise ValueError('cutter is not embedded')
        cgroup=regular_orientation(vc,fc)
        if len(cgroup)!=1:raise ValueError('cutter needs one connected boundary')
        report['cutter_embedding_checked']=True
        report['cutter_exact_tetra_count']=len(fc)
        pts=[tuple(map(Q,p)) for p in v]
        report.update(embedding={k:v for k,v in fields.items() if k!='elapsed_seconds'},
            outward_winding_component_face_counts=regular_orientation(v,f),
            parent_coverage=parent_patches(vd,fd,vc,fc,pts,f,j['birth_faces']),
            source_position=strict_source_position(vd,fd,vc,fc),regularized_intersection_verified=True)
    (evidence/(folder.name+'.w1.json')).write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sample',type=Path);p.add_argument('design',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--ten',type=Path,required=True)
    p.add_argument('--embedding',type=Path,required=True);p.add_argument('--full',action='store_true')
    a=p.parse_args();print(json.dumps(audit(a.sample,a.design,a.ten,a.embedding,a.output,a.full),indent=2))
