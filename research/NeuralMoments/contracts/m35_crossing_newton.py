"""Drive unmodified native Newton with a fail-closed discrete M32 crossing map in tip coordinates."""
import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

FIXED_SHA = "213e485af9604156e156426a1a302e8bceec4bf23bfd7cbdbc3af1844b2ca6ed"
DSHA = "c3633c74bc74c2be488eb22de108454276f28941f0477cfee5837f57db99d622"
TIP = Q(-0.16)
MAP_ID = "M35-exact-design-discrete-negative-ellipsoid-intersection-v1"


def digest(raw):
    return sha256(raw).hexdigest()


def read_ten(path):
    # Truncating each exact rational to 190 fractional bits before frame
    # conversion introduces <2^-190 absolute error in each incoming moment.
    scale=1<<190
    rows=path.read_text().splitlines()
    if len(rows)!=10:
        raise ValueError("ten moments required")
    out=[]
    for row in rows:
        a,b=[int(v,0) for v in row.split("/")]
        if b<=0:raise ValueError("wrong denominator")
        out.append(Q((a*scale)//b,scale))
    return out


def shift(raw):
    m,x,y,z,xx,yy,zz,yz,xz,xy=raw
    return [m,x,y,z-TIP*m,xx,yy,zz-2*TIP*z+TIP**2*m,yz-TIP*y,xz-TIP*x,xy]


def topology(obj, design_keys, cutter_keys, cutter_faces):
    """An oriented indexed complex identified by stable source features, independent of OBJ renumbering."""
    verts=obj["vertices"];faces=obj["faces"]; parents=obj["birth_faces"]
    adj=[set() for _ in verts]
    for face, owner in zip(faces,parents,strict=True):
        for v in face:adj[v].add(owner)
    identity=[]
    for i,row in enumerate(verts):
        point=tuple(map(Q,row))
        if point in design_keys:
            key=("D",design_keys[point])
        elif point in cutter_keys:
            key=("C",cutter_keys[point])
        else:
            key=("X",*sorted(adj[i]))
            if len(key)<4:raise ValueError("ambiguous_crossing_incidence")
        identity.append(key)
    if len(set(identity))!=len(identity):raise ValueError("duplicate_feature_identity")
    tris=[]
    for f,o in zip(faces,parents):
        t=[identity[k] for k in f]
        tris.append((o,*min(t[i:]+t[:i] for i in range(3))))
    return digest(json.dumps([cutter_faces,sorted(tris)],separators=(",",":")).encode())


def keys(v):
    vertices=[tuple(map(Q,row)) for row in v]
    if len(set(vertices))!=len(vertices):raise ValueError("nonunique_source_vertices")
    return {p:i for i,p in enumerate(vertices)}


def ref_state(q, moments):
    # Independent values and derivatives for the original normalized V7 InertiaAndLowering objective.
    def physics(r):
        m,x,y,z,xx,yy,zz,yz,xz,xy=r
        a,b,c=yy+zz-(y*y+z*z)/m,xx+zz-(x*x+z*z)/m,xx+yy-(x*x+y*y)/m
        d=-xy+x*y/m
        return np.array([(a*a+b*b+2*d*d)/(c*c)+z/m,x,y,-xz+x*z/m,-yz+y*z/m])
    r=np.array(moments,float);p=np.array(q,float)
    p[7:]*=2;p/=np.linalg.norm(p)
    val=physics(r)
    grad=np.column_stack([physics(r+1e-24j*e).imag/1e-24 for e in np.eye(10)])
    mat=np.column_stack((-grad[1:].T,p));lm,_,rank,_=np.linalg.lstsq(mat,-grad[0],rcond=None)
    return dict(objective=float(val[0]),raw_residual=float(np.linalg.norm(np.r_[grad[0]+mat@lm,val[1:]])),
                physical_relative=float(np.linalg.norm(val[1:])/max(1,np.linalg.norm(r))),mu=float(lm[-1]),
                projection_rank=int(rank),constraints=val[1:].tolist())


def exact_moments(args, folder, out, result_hash):
    # Branch-specific W1 and W2 modules have the same import name.
    # Run W1 in a separate interpreter.
    subprocess.run([sys.executable, str(args.w1/"m35_noodles_crossing.py"), str(folder),
                    str(args.design), str(out/"moments"), "--ten", str(args.ten),
                    "--embedding", str(args.embedding)], check=True, capture_output=True)
    rec=json.loads((out/"moments"/(folder.name+".w1.json")).read_text())
    if rec["output_sha256"] != result_hash:
        raise ValueError("moment packet mismatch")
    return shift(read_ten(out/"moments"/(folder.name+".ten")))


def bounded_moments(args, folder, result_hash):
    raw=subprocess.check_output([sys.executable,str(args.w1/"m35_bounded_ten.py"),str(folder/"result.json")],text=True)
    report=json.loads(raw)
    if report["source_sha256"]!=result_hash or len(report["ten_bounds"])!=10:
        raise ValueError("moment packet mismatch")
    pairs=[tuple(map(Q,row)) for row in report["ten_bounds"]]
    if any(lo>hi or hi-lo>Q(1,1<<90) for lo,hi in pairs):
        raise ValueError("uncertain input moments")
    (folder/"moments.json").write_text(raw)
    return shift([(lo+hi)/2 for lo,hi in pairs])


def cached_sample(original, folder, q, design, source, boolean, bounds):
    rec=json.loads((original/"receipt.json").read_text())
    src=json.loads((original/"source.json").read_text())
    if (rec["design_sha256"]!=digest(design) or rec["source_executable"]!=digest(source.read_bytes())
            or rec["csg_executable"]!=digest(boolean.read_bytes()) or src["raw_q"]!=q
            or src["bound_reference_sha256"]!=digest(bounds.read_bytes())):
        raise ValueError("cached sample identity mismatch")
    folder.mkdir()
    for name, key in (("source.json","source_sha256"),("cutter.txt","cutter_sha256"),("result.json","result_sha256")):
        raw=(original/name).read_bytes()
        if digest(raw)!=rec[key]:raise ValueError("cached file digest mismatch")
        (folder/name).write_bytes(raw)
    shutil.copyfile(original/"receipt.json",folder/"receipt.json")


def execute(args):
    for d in (args.w1,args.w2,args.w3):sys.path.insert(0,str(d.resolve()))
    from m35_exact_crossing import evaluate
    from crossing_report import sample_flux
    from m35_general_crossing import general_evaluate
    design=args.design.read_bytes()
    if digest(design)!=DSHA or digest(args.fixed_ten.read_bytes())!=FIXED_SHA:raise ValueError("unbound M32 input")
    lines=design.decode().splitlines()
    design_keys=keys([r.split() for r in lines[1:1+int(lines[0].split()[0])]])
    fixed=shift(read_ten(args.fixed_ten))
    if args.chart=="full-rank" and args.bounds is None:raise ValueError("missing finite-source bounds")
    if args.start_index is not None:
        starts=subprocess.check_output([str(args.kernel),"--starts"],text=True).splitlines()
        q0=list(map(float,starts[args.start_index].split()))
    else:
        q0=json.loads(args.start.read_text())["q"]
    map_id=MAP_ID if args.chart=="ellipsoid" else "M35-exact-design-stock-finite-quadric-v1"
    if args.replay_first and args.chart!="full-rank":
        raise ValueError("first-sample replay requires a bound full-rank source")
    if args.moment_method=="exact" and (args.ten is None or args.embedding is None or args.chart!="ellipsoid"):
        raise ValueError("exact elliptic replay requires W1 moment and embedding executables")
    terminal=None
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    ledger=[];events=[];started=time.monotonic()
    with subprocess.Popen([str(args.kernel),str(args.iterations),str(args.limit)],stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,stderr=out.joinpath("kernel.stderr").open("w"),text=True,bufsize=1) as proc:
        def send(data):
            proc.stdin.write(data+"\n");proc.stdin.flush()
        send(" ".join(map(str,q0)))
        for line in proc.stdout:
            fields=line.split()
            if not fields:continue
            if fields[0]=="EVENT":
                row=dict(zip(("type","restart","iteration","index","outcome","reason","step","raw_residual",
                              "physical_relative","objective","mu"),fields[1:],strict=True))
                events.append(row)
                if row["type"]=="evaluation" and row["raw_residual"]!="null":
                    ref=ledger[int(row["index"])-1]["independent"]
                    for k in ("raw_residual","physical_relative","objective","mu"):
                        if not np.isclose(float(row[k]),ref[k],rtol=1e-8,atol=1e-9):
                            raise ValueError("independent native state mismatch "+k)
                continue
            if fields[0]=="TERMINAL":
                terminal=dict(zip(("status","calls","starts","trace_entries","event_hash"),fields[1:],strict=True))
                break
            if fields[0]!="QUERY":raise ValueError("bad kernel record")
            q=list(map(float,fields[1:]));idx=len(ledger)+1
            if len(q)!=10 or idx>args.limit:raise ValueError("invalid kernel query")
            row=dict(index=idx,q=q);ledger.append(row)
            folder=out/f"call-{idx:03d}"
            try:
                if time.monotonic()-started>=args.seconds:raise ValueError("capture_budget_expired")
                replay=(idx==1 and args.replay_first is not None)
                row["boolean_execution"]="replayed_hash_bound" if replay else "fresh_requested"
                if replay:
                    cached_sample(args.replay_first,folder,q,design,args.source,args.boolean,args.bounds)
                elif args.chart=="full-rank":
                    general_evaluate(design,np.asarray(q),folder,args.boolean,args.source,args.bounds)
                else:
                    evaluate(design,q,folder,args.boolean,args.source)
                receipt=json.loads((folder/'receipt.json').read_text())
                source=json.loads((folder/'source.json').read_text())
                if q!=source["raw_q"]:raise ValueError("native_query_changed")
                obj=json.loads((folder/"result.json").read_text());jac=sample_flux(folder)
                if args.moment_method=="exact":
                    variable=exact_moments(args,folder,out,receipt["result_sha256"])
                else:
                    variable=bounded_moments(args,folder,receipt["result_sha256"])
                total=[x+y for x,y in zip(variable,fixed)]
                r=list(map(float,total));v=list(map(float,variable))
                if jac["result_sha256"]!=receipt["result_sha256"]:
                    raise ValueError("unpaired material")
                th=topology(obj,design_keys,keys(source["vertices"]),source["faces"])
                row.update(result_sha256=receipt["result_sha256"],topology=th,
                           retained_ten_tip=r,variable_ten_tip=v,J_tip=jac["J_tip"],independent=ref_state(q,r),
                           variable_mesh=[len(obj["vertices"]),len(obj["faces"])])
                send(" ".join(["MAP",receipt["result_sha256"],th,map_id,"libigl-CGAL-rational-M34","moving-parent-flux-M35",
                               *map(repr,r),*map(repr,v),*map(repr,np.asarray(jac["J_tip"]).ravel().tolist())]))
            except (ValueError,subprocess.CalledProcessError) as exc:
                row["error"]=str(exc)
                send("FAIL "+type(exc).__name__+"_"+"_".join(str(exc).split())[:120])
            (out/"partial.json").write_text(json.dumps(dict(ledger=ledger,events=events),allow_nan=False))
        proc.stdin.close()
        if proc.wait(timeout=10)!=0:raise RuntimeError("native solver exited unsuccessfully")
    if terminal is None or len(ledger)!=int(terminal["calls"]):raise ValueError("receipt call count")
    report=dict(schema="W0_M35_NATIVE_NEWTON_DIAGNOSTIC_V1",map=map_id,start_index=args.start_index,moment_method=args.moment_method,kernel_sha256=digest(args.kernel.read_bytes()),
                limits=dict(iterations=args.iterations,evaluations=args.limit,wall_admission_seconds=args.seconds),
                start=q0,terminal=terminal,evaluations=ledger,events=events,elapsed_seconds=time.monotonic()-started,
                formal_speed_comparison=False,native_cherchi_calls=0)
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("w1","w2","w3","kernel","source","boolean","design","fixed-ten","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--iterations",type=int,default=2);p.add_argument("--limit",type=int,default=8)
    p.add_argument("--seconds",type=float,default=600.)
    p.add_argument("--chart",choices=("ellipsoid","full-rank"),default="ellipsoid")
    initial=p.add_mutually_exclusive_group(required=True)
    initial.add_argument("--start",type=Path);initial.add_argument("--start-index",type=int,choices=range(8))
    p.add_argument("--bounds",type=Path);p.add_argument("--replay-first",type=Path)
    p.add_argument("--moment-method",choices=("exact","bounded"),default="bounded")
    p.add_argument("--ten",type=Path);p.add_argument("--embedding",type=Path)
    execute(p.parse_args())
