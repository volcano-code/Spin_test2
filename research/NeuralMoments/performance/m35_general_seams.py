"""Independent oriented seam and regularized region replay, including disconnected cutters."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import numpy as np

from m28_cut_cell_coverage import contains, edges
from m29_heart_cut_faces import area2, dot, homogeneous, plane, projected
from m34_noodles_partition import Q, mesh_stream, point_winding, regular_orientation, sha
from m35_noodles_crossing import DESIGN_SHA, packet, seam_loops


def classify(a,b):
    """All nontrivial connected components of the oriented triangle mesh."""
    parent=list(range(len(a)))
    def find(i):
        while i!=parent[i]:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,j,k in b:
        i,j,k=find(i),find(j),find(k);parent[j]=parent[k]=i
    groups=defaultdict(list)
    for i,f in enumerate(b):groups[find(f[0])].append(i)
    return list(groups.values())


def section(tr, dist):
    return [tuple(a[k]+(b[k]-a[k])*u/(u-v) for k in range(3))
            for (a,b),(u,v) in zip(edges(tr),edges(dist)) if u*v<0]


def expected_seams(d,fd,c,fc,op):
    tri=np.array(d,dtype=float)[np.array(fd)];lo,hi=tri.min(1),tri.max(1)
    hd=list(map(homogeneous,d));hc=list(map(homogeneous,c));dp={};cnt=0
    out={}
    for j,cf in enumerate(fc):
        ctr=[c[k] for k in cf]; cp=plane(ctr);bounds=np.asarray(ctr,dtype=float)
        cand=np.flatnonzero(np.all((hi>=bounds.min(0))&(lo<=bounds.max(0)),axis=1))
        for i in cand:
            i=int(i);cnt+=1;df=fd[i]
            d_num=[dot(cp,hd[k]) for k in df]
            if min(d_num)>0 or max(d_num)<0:continue
            if i not in dp:dp[i]=plane([d[k] for k in df])
            pi=dp[i]; c_num=[dot(pi,hc[k]) for k in cf]
            if min(c_num)>0 or max(c_num)<0:continue
            if 0 in d_num or 0 in c_num:raise ValueError('VERTEX_ON_OPPOSITE_CANDIDATE_PLANE')
            tangent=tuple(pi[(k+1)%3]*cp[(k+2)%3]-pi[(k+2)%3]*cp[(k+1)%3] for k in range(3))
            axis=max(range(3),key=lambda k:abs(tangent[k]))
            if not tangent[axis]:raise ValueError('PARALLEL_STRADDLING_PLANES')
            s=sorted(section([d[k] for k in df],[Q(x,hd[k][-1]) for k,x in zip(df,d_num)]),key=lambda v:v[axis])
            r=sorted(section(ctr,[Q(x,hc[k][-1]) for k,x in zip(cf,c_num)]),key=lambda v:v[axis])
            a,b=max((s[0],r[0]),key=lambda v:v[axis]),min((s[-1],r[-1]),key=lambda v:v[axis])
            if a[axis]>b[axis]:continue
            if a==b or any(u==v for u in s for v in r):raise ValueError('POINT_OR_EDGE_EDGE_CONTACT')
            out[i,j]=((a,b) if ((op=='intersection')==(tangent[axis]>0)) else (b,a))
    return out,cnt


def measured_seams(o,fo,owners,nd):
    edges_by_id=defaultdict(list)
    for face,oi in zip(fo,owners):
        for a,b in edges(face):edges_by_id[(min(a,b),max(a,b))].append((oi,a,b))
    out={}
    for li in edges_by_id.values():
        if len(li)!=2:raise ValueError('OPEN_EDGE')
        (u,a,b),(v,y,x)=sorted(li)
        if (a,b)!=(x,y):raise ValueError('INCONSISTENT_ORIENTATION')
        if u<nd<=v:
            key=u,v-nd
            if key in out:raise ValueError('NON_SIMPLE_PARENT_PAIR_SEAM')
            out[key]=(o[a],o[b])
    return out


def carriers(v,fs,other,of,o,fo,owners,offset,sign,op,side):
    gr=defaultdict(list);partial=set();pos=0
    for f,w in zip(fo,owners):
        if offset<=w<offset+len(fs):gr[w-offset].append(f)
    for i,ch in gr.items():
        t=tuple(v[k] for k in fs[i])
        if len(ch)==1:
            target=tuple(o[k] for k in ch[0])
            if sign<0:target=target[::-1]
            if any(t[k:]+t[:k]==target for k in range(3)):pos+=1;continue
        eq=plane(t);axis=max(range(3),key=lambda k:abs(eq[k]))
        par=[projected(p,axis) for p in t];whole=area2(par);saved=0
        for f in ch:
            pts=[o[k] for k in f]
            if any(dot(eq,homogeneous(p)) for p in pts):raise ValueError('WRONG_PARENT_PLANE')
            tri=[projected(p,axis) for p in pts]
            val=area2(tri)*sign
            if val*whole<=0 or not all(contains(par,p) for p in tri):raise ValueError('WRONG_PARENT_SUPPORT_OR_SIGN')
            saved+=val
        if abs(saved)>abs(whole):raise ValueError('PARENT_OVER_COVERED')
        if abs(saved)<abs(whole):partial.add(i)
        pos+=1
    coverage=[];other_bounds=None
    for ids in classify(v,fs):
        k=next((i for i in ids if i in partial),None)
        if k is not None:
            coverage.append(dict(facets=len(ids),partial_anchor=k));continue
        # Parentwise 0/full already established. Match one interior anchor to
        # the true operand winding to fix the constant on this component.
        if other_bounds is None:
            tri=np.array(other,dtype=float)[np.array(of)];other_bounds=np.nextafter(tri.min(1),-np.inf),np.nextafter(tri.max(1),np.inf)
        for i in ids:
            p=tuple(sum(t)/3 for t in zip(*(v[j] for j in fs[i])))
            try:w=point_winding(other,of,p,*other_bounds)
            except ValueError:continue
            if w not in (0,1):raise ValueError('BAD_OPERAND_WINDING')
            expected=(1-w) if side==0 and op=='difference' else w
            if bool(gr.get(i))!=bool(expected):raise ValueError('COMPONENT_COVERAGE_MISMATCH')
            coverage.append(dict(facets=len(ids),generic_winding_anchor=i,other_winding=w));break
        else:raise ValueError('NO_GENERIC_COMPONENT_ANCHOR')
    return dict(nonempty=pos,partial=len(partial),components=coverage)


def embedding(binary,stream,seconds):
    p=subprocess.run([str(binary),'--max-seconds',str(seconds)],input=stream,capture_output=True,
                     check=True,timeout=seconds+60)
    fields=dict(s.split('=',1) for s in p.stdout.decode().splitlines() if '=' in s)
    if fields.get('status')!='EXACT_POST_BOOLEAN_SELF_INTERSECTION_FREE' or fields['candidate_pairs']!=fields['exact_pairs_checked']:
        raise ValueError('EMBEDDING_FAILED')
    fields.pop('elapsed_seconds',None)
    return fields


def audit(folder, design, binary):
    raw=(folder/'result.json').read_bytes();obj=json.loads(raw)
    op=obj['operation']
    if not obj['succeeded'] or op not in ('intersection','difference'):raise ValueError('UNSUPPORTED_OPERATION')
    dh,d,fd=packet(design);ch,c,fc=packet(folder/'cutter.txt')
    receipt=json.loads((folder/'receipt.json').read_bytes())
    if dh!=DESIGN_SHA or receipt['result_sha256']!=sha(raw) or receipt['cutter_sha256']!=ch or receipt['design_sha256']!=dh or receipt['operation']!=op:
        raise ValueError('SOURCE_RECEIPT_MISMATCH')
    o=[tuple(map(Q,v)) for v in obj['vertices']];fo=list(map(tuple,obj['faces']));ow=obj['birth_faces']
    if not fo or len(ow)!=len(fo) or min(ow)<0 or max(ow)>=len(fd)+len(fc):raise ValueError('BAD_BIRTH_FACES')
    ce=embedding(binary,(folder/'cutter.txt').read_bytes(),200);co=regular_orientation(c,fc)
    oe=embedding(binary,mesh_stream(obj['vertices'],fo),1200);oo=regular_orientation(o,fo)
    ex,num=expected_seams(d,fd,c,fc,op);obs=measured_seams(o,fo,ow,len(fd))
    if not ex or ex!=obs:raise ValueError('EXACT_ORIENTED_SEAMS_DIFFER')
    dc=carriers(d,fd,c,fc,o,fo,ow,0,1,op,0)
    cc=carriers(c,fc,d,fd,o,fo,ow,len(fd),1 if op=='intersection' else -1,op,1)
    return dict(schema='w1-m35-general-exact-regions-v1',sample=folder.name,
        result_sha256=sha(raw),design_sha256=dh,cutter_sha256=ch,operation=op,
        exact_input_candidate_pairs=num,matching_oriented_seam_segments=len(ex),
        seam_cycles=seam_loops(fo,ow,len(fd)),design_carrier=dc,cutter_carrier=cc,
        cutter_embedding=ce,output_embedding=oe,cutter_components=co,output_components=oo,
        regularized_set_equality=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sample',type=Path);p.add_argument('design',type=Path);p.add_argument('checker',type=Path)
    args=p.parse_args()
    print(json.dumps(audit(args.sample,args.design,args.checker),indent=2))
