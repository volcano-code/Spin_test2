"""Frozen unseen-mesh clipped-moment panel for Spin-PINO. Coordinates are normalized model coordinates."""
import argparse
from collections import Counter
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from m34_noodles_partition import Q, mesh_stream, point_winding, regular_orientation, sha
from m35_bounded_ten import integrate as integrate_ten
from m35_general_seams import audit as check_regions, embedding
from m35_noodles_crossing import packet, construction_signature

OFF_SHA = "dda85ca6dc03ed83a0b754611d5c0b1a3c8ba5234e690479ac4a04063e6a7d4d"
CASES = (
    ("largest-face-small", "area", (.11, .079, .054)),
    ("largest-face-medium", "area", (.32, .25, .20)),
    ("left-face-large", "left", (.58, .44, .31)),
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def material(vertices, faces):
    rows = []
    for point in vertices:
        nums = [float(i).as_integer_ratio() for i in point]
        rows.append([f"{n}/{d}" for n, d in nums])
    return mesh_stream(rows, faces)


def mesh(off):
    raw = off.read_bytes()
    if sha(raw) != OFF_SHA:
        raise ValueError("unexpected Bunny OFF asset")
    rows = raw.decode("ascii").splitlines()
    nv, nf, _ = map(int, rows[1].split())
    v = np.asarray([r.split() for r in rows[2:2+nv]], dtype=float)
    polygons = np.asarray([r.split() for r in rows[2+nv:]], dtype=int)
    if polygons.shape != (nf, 4) or not (polygons[:, 0] == 3).all():
        raise ValueError("expected triangle mesh")
    faces = polygons[:, 1:].tolist()
    edges = Counter((i, j) for f in faces for i, j in zip(f, f[1:]+f[:1]))
    if any(n != 1 or edges[j,i] != 1 for (i,j), n in edges.items()):
        raise ValueError("not closed and consistently oriented")
    center = (v.max(0) + v.min(0)) / 2
    scale = 1.8 / max(v.max(0) - v.min(0))
    return (v-center)*scale, faces, dict(center=center.tolist(), scale=float(scale))


def native_q(center, axes):
    # Fixed shears break inverse-Gauss pole degeneracy without using labels.
    basis = np.diag(axes)
    basis[0,1] = .13*axes[0]
    basis[1,2] = -.11*axes[1]
    a = -np.linalg.inv(basis@basis.T)
    q = np.r_[1+center@a@center, -2*a@center, np.diag(a), a[1,2], a[0,2], a[0,1]]
    return q / np.linalg.norm(q)


def voxels(v, faces, side):
    vertices = [tuple(map(Q, point)) for point in v]
    tris = v[faces]
    lower = np.nextafter(tris.min(1), -np.inf)
    upper = np.nextafter(tris.max(1), np.inf)
    axis = np.linspace(-1, 1, side)
    points = [Q(float(x)) for x in axis]
    occupied = [point_winding(vertices, faces, pt, lower, upper) for pt in product(points, repeat=3)]
    if not set(occupied) <= {0,1}:
        raise ValueError("invalid mesh winding")
    return dict(axis=axis.tolist(), indexing="ij", cube_center_in_tip=[0,0,0], cube_halfwidth=1,
                occupied=sum(occupied), occupancy=np.reshape(occupied,(side,side,side)).tolist())


def canonical_mesh_hash(obj):
    # CGAL can choose a different starting vertex for the same oriented triangle.
    faces = [min(f[i:]+f[:i] for i in range(3)) for f in obj["faces"]]
    canon = dict(vertices=obj["vertices"],faces=faces,birth_faces=obj["birth_faces"])
    return sha(json.dumps(canon,separators=(",",":")).encode())


def invoke(args, design, ref, q, name, base):
    from crossing_report import check_source_motion, midpoint
    from crossing_flux import integrate as flux
    target = args.out/name
    target.mkdir()
    txt = subprocess.check_output([str(args.source)], input=" ".join(map(repr,q.tolist())), text=True)
    (target/"source.txt").write_text(txt)
    rows = txt.splitlines()
    n, nf = map(int, rows[0].split())
    t = np.asarray([r.split() for r in rows[1:1+n]], dtype=float)
    cfaces = [tuple(map(int,r.split())) for r in rows[n+1:]]
    if t.shape != (n,36) or len(cfaces)!=nf:
        raise ValueError("malformed native source map")
    v = t[:,3:6]
    vel = t[:,6:].reshape(n,3,10)
    cutter = material(v,cfaces)
    (target/"cutter.txt").write_bytes(cutter)
    both = target/"operands.txt"; both.write_bytes(design+cutter)
    result = target/"result.json"
    subprocess.run([str(args.boolean),str(both),str(result),"intersection"],check=True)
    both.unlink()
    raw = result.read_bytes(); obj = json.loads(raw)
    nd = ref["fixed_faces"]
    fixed = sum(b < nd for b in obj["birth_faces"])
    if not obj.get("succeeded") or not 0 < fixed < len(obj["faces"]):
        raise ValueError("crossing needs both boundary sources")
    rec=dict(result_sha256=sha(raw), cutter_sha256=sha(cutter), design_sha256=sha(design), operation="intersection")
    write_json(target/"receipt.json", rec)
    bounds=integrate_ten(obj["vertices"],obj["faces"])
    ten=np.array([float((a+b)/2) for a,b in bounds])
    topo=construction_signature(obj["faces"],obj["birth_faces"],nd,len(obj["vertices"]))
    if not base:
        return ten,topo
    geom=check_regions(target,args.out/"design.txt",args.embedding,design_sha=ref["design_sha256"])
    write_json(target/"region.json",geom)
    mesh_vertices=[tuple(map(Fraction,p)) for p in obj["vertices"]]
    parents=[tuple(map(Fraction,p)) for p in v.tolist()]
    velocities=[[[Fraction(val) for val in line] for line in vv] for vv in vel.tolist()]
    lo,hi,nmove=flux(mesh_vertices,obj["faces"],obj["birth_faces"],parents,cfaces,velocities,nd,"intersection")
    J=midpoint(lo,hi)
    s=check_source_motion(dict(q=(q/np.linalg.norm(q)).tolist(),vertices_tip=v.tolist(),
                          native_jacobian=vel.tolist(),normals=t[:,:3].tolist()))
    sample=dict(name=name,q=q.tolist(),result_sha256=sha(raw),variable_ten_tip=ten.tolist(),J_tip=J)
    audit=dict(canonical_mesh_sha256=canonical_mesh_hash(obj),vertices=len(mesh_vertices),faces=len(obj['faces']),moving_faces=nmove,
               fixed_faces=fixed,source_sha256=sha(txt.encode()),source_motion=s,exact_region=geom,
               topology=topo,scaling_null=float(np.max(np.abs(np.asarray(J)@q))))
    write_json(target/"sample.json",sample)
    write_json(target/"audit.json",audit)
    return sample,audit


def differences(args, design, ref, q, sample, base_audit, col):
    outputs = []
    for exponent in (19,20):
        h = 2.**-exponent
        delta = np.eye(10)[col]*h
        sides = []
        for sign, suffix in ((-1,"minus"),(1,"plus")):
            name = f"{sample['name']}-c{col}-{suffix}{exponent}"
            sides.append(invoke(args, design, ref, q+sign*delta, name, False))
        (left, tl), (right, tr) = sides
        anchor = base_audit["topology"]
        all_unique = all(x["unambiguous_vertex_stars"] for x in (anchor,tl,tr))
        changes = [sorted(k for k in anchor if side.get(k)!=anchor[k]) for side in (tl,tr)]
        seam_stable = all_unique and all(set(diff)<={"oriented_triangulation_sha256"} for diff in changes)
        analytical = np.asarray(sample["J_tip"])[:,col]
        numerical = (right-left)/(2*h)
        error = float(np.max(np.abs(numerical-analytical)/np.maximum(1,np.abs(analytical))))
        outputs.append(dict(column=col,step_exponent=exponent,max_scaled_error=error,
                            unambiguous_seam_and_incidence=seam_stable,
                            triangulation_stable=all_unique and not any(changes),
                            changed_hashes_minus_plus=changes))
    return outputs


def execute(args):
    sys.path.insert(0,str(args.w3.resolve()))
    args.out.mkdir(exist_ok=True)
    v, f, transform=mesh(args.off)
    design=material(v,f); (args.out/"design.txt").write_bytes(design)
    triangles=v[f]; centers=triangles.mean(1)
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    index=dict(area=int(np.argmax(area)),left=int(np.argmin(centers[:,0])))
    ref=dict(off_sha256=OFF_SHA,design_sha256=sha(design),fixed_faces=len(f),transform=transform,
             locations=index,source_executable_sha256=sha(args.source.read_bytes()),
             boolean_executable_sha256=sha(args.boolean.read_bytes()),embedding_executable_sha256=sha(args.embedding.read_bytes()),
             embedding=embedding(args.embedding,design,180),
             outward_components=regular_orientation(v.tolist(), f))
    qs=[native_q(centers[index[loc]],axes) for _,loc,axes in CASES]
    write_json(args.out/"protocol.json",dict(ref,probes=[dict(name=c[0],center=centers[index[c[1]]].tolist(),
               axes=c[2],q=q.tolist()) for c,q in zip(CASES,qs)],validation=dict(primary_column=0,supplementary_columns=list(range(1,10)),steps=[19,20])))
    vx=voxels(v,f,8); vx.update(design_sha256=ref["design_sha256"])
    samples=[]; audits=[]
    for (name,_,_),q in zip(CASES,qs):
        sample,a=invoke(args,design,ref,q,name,True)
        samples.append(sample);audits.append(a)
    write_json(args.out/"heldout.json",dict(schema="w6-offline-m36-bunny-discrete-cgal-v1",voxels=vx,samples=samples))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for q,s,a in zip(qs,samples,audits):
            pairs = list(pool.map(lambda col: differences(args,design,ref,q,s,a,col), range(10)))
            a["fd_checks"] = [r for pair in pairs for r in pair]
    write_json(args.out/"audit.json",dict(protocol=ref,cases=audits,heldout_sha256=sha((args.out/"heldout.json").read_bytes())))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("off","source","boolean","embedding","w3","out"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=1)
    execute(parser.parse_args())
