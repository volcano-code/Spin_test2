"""M72 formal-corpus adapter, GPU preflight and resumable trainer for M80."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import tempfile
from time import perf_counter

import numpy as np
import torch

from .native_pcraft import (SCHEMA, SEMANTICS, PRESETS, PcraftNativeMomentOperator,
                            conservative_coarse_grid, load_geometry, make_config, load_model)

TRAIN_SCHEMA = "m80.pcraft-native-training.v1"


def file_sha(path):
    h=sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def manifest_fingerprint(doc):
    body={k:v for k,v in doc.items() if k!="fingerprint"}
    raw=json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return sha256(raw).hexdigest()


def safe_member(root,relative):
    if not isinstance(relative,str) or not relative or "\\" in relative or ":" in relative or "\x00" in relative:
        raise ValueError("manifest path must be a relative POSIX path")
    p=PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("manifest path escapes corpus root")
    base=Path(root).resolve()
    result=(base/Path(*p.parts)).resolve()
    if not result.is_relative_to(base) or result==base:
        raise ValueError("manifest path escapes corpus root")
    return result


def source_fingerprints():
    here=Path(__file__).resolve().parent
    parent=here.parent
    paths={"native_pcraft.py":here/"native_pcraft.py",
           "native_train.py":here/"native_train.py",
           "pino_backbone.py":here/"pino_backbone.py",
           "model.py":parent/"model.py"}
    return {name:file_sha(path) for name,path in paths.items()}


def runtime_info(device,amp):
    result=dict(python=platform.python_version(),torch=str(torch.__version__),
        numpy=np.__version__,device=device,amp=amp,threads=torch.get_num_threads(),
        threading={k:os.environ.get(k) for k in ("MKL_THREADING_LAYER","OMP_NUM_THREADS",
          "OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","MKL_SERVICE_FORCE_INTEL","KMP_DUPLICATE_LIB_OK")})
    if str(device).startswith("cuda"):
        result.update(cuda=str(torch.version.cuda),gpu=torch.cuda.get_device_name(device),
            bf16_supported=torch.cuda.is_bf16_supported(),
            capability=list(torch.cuda.get_device_capability(device)))
    return result


class M72Corpus:
    """Strict reader for frozen M69/M72 manifests and their audit receipt."""
    def __init__(self,manifest,audit,grid_size,*,min_queries=1,cache_mib=512):
        self.path=Path(manifest).resolve(); self.root=self.path.parent
        self.audit_path=Path(audit).resolve()
        if not self.path.is_file() or not self.audit_path.is_file():
            raise FileNotFoundError("manifest and audit JSON are both required")
        self.manifest_sha=file_sha(self.path); self.audit_sha=file_sha(self.audit_path)
        self.doc=json.loads(self.path.read_text()); self.audit=json.loads(self.audit_path.read_text())
        if self.doc.get("schema")!="m69.corpus.v1" or self.doc.get("scope")!="formal" or self.doc.get("frozen") is not True:
            raise ValueError("M80 large training requires a frozen formal M69/M72 corpus")
        if self.doc.get("fingerprint")!=manifest_fingerprint(self.doc):
            raise ValueError("M69/M72 manifest fingerprint mismatch")
        if (self.audit.get("schema") != "m69.corpus-audit.v1" or self.audit.get("valid") is not True
                or self.audit.get("ready_for_training") is not True or self.audit.get("blockers")):
            raise ValueError("M72 corpus audit is not a valid ready_for_training receipt")
        if self.audit.get("fingerprint")!=self.doc.get("fingerprint") or self.audit.get("scope")!="formal":
            raise ValueError("M72 audit does not bind the supplied formal manifest")
        self.rows=[r for r in self.doc.get("records",[]) if r.get("status")=="qualified"]
        self.splits={s:sorted([r for r in self.rows if r.get("split")==s],key=lambda r:r["id"])
                     for s in ("train","val","test")}
        if any(not self.splits[s] for s in self.splits):
            raise ValueError("formal corpus must have qualified train/val/test rows")
        groups={}
        for split,rows in self.splits.items():
            for row in rows:
                gid=row.get("group_id")
                if not gid: raise ValueError(f"{row.get('id')}: missing source group")
                if gid in groups and groups[gid]!=split: raise ValueError("source group leakage across splits")
                groups[gid]=split
        self.train_groups=self._group(self.splits["train"])
        self.min_queries=int(min_queries)
        self.grid_size=int(grid_size)
        self.cache_budget=int(cache_mib*1024**2)
        if self.cache_budget<0: raise ValueError("cache_mib must be nonnegative")
        self.cache={}; self.cache_bytes=0

    @staticmethod
    def _group(rows):
        by=defaultdict(list)
        for r in rows: by[r["group_id"]].append(r)
        return [by[k] for k in sorted(by)]

    def get(self,row):
        gp=safe_member(self.root,row["geometry_file"])
        lp=safe_member(self.root,row["labels_file"])
        key=row["id"]
        if key in self.cache: return self.cache[key]
        if file_sha(gp)!=row["geometry_sha256"] or file_sha(lp)!=row["labels_sha256"]:
            raise ValueError(f"{key}: geometry or label SHA-256 mismatch")
        geometry=load_geometry(gp)
        geometry["pcraft_grid"]=conservative_coarse_grid(geometry,self.grid_size)
        with np.load(lp,allow_pickle=False) as z:
            labels={k:np.array(z[k],copy=True) for k in ("queries","moments","jacobian","derivative_valid")}
        n=len(labels["queries"])
        if labels["queries"].shape!=(n,10) or labels["moments"].shape!=(n,10) or labels["jacobian"].shape!=(n,10,10) or labels["derivative_valid"].shape!=(n,):
            raise ValueError(f"{key}: invalid M72 label array shapes")
        if n<self.min_queries: raise ValueError(f"{key}: {n} queries < required {self.min_queries}")
        if not np.isfinite(labels["queries"]).all() or not np.isfinite(labels["moments"]).all():
            raise ValueError(f"{key}: nonfinite query/moment labels")
        valid=labels["derivative_valid"].astype(bool)
        if not np.isfinite(labels["jacobian"][valid]).all():
            raise ValueError(f"{key}: nonfinite derivative-valid Jacobian")
        for a in labels.values(): a.flags.writeable=False
        size=sum(v.nbytes for v in geometry.values() if isinstance(v,np.ndarray))+sum(v.nbytes for v in labels.values())
        if size<=self.cache_budget:
            while self.cache and self.cache_bytes+size>self.cache_budget:
                old_key=next(iter(self.cache)); old=self.cache.pop(old_key)
                self.cache_bytes-=old[2]
            self.cache[key]=(geometry,labels,size); self.cache_bytes+=size
            return geometry,labels
        return geometry,labels

    def largest_train_row(self):
        best=None
        for row in self.splits["train"]:
            geometry,_=self.get(row)
            n=len(geometry["xyz"])
            if best is None or n>best[0]: best=(n,row)
        return best[1]

    def verify_query_coverage(self, minimum):
        """Check every qualified label bank before a formal long run."""
        counts = {}
        for row in self.rows:
            path = safe_member(self.root, row["labels_file"])
            if file_sha(path) != row["labels_sha256"]:
                raise ValueError(f"{row['id']}: label SHA-256 mismatch during coverage audit")
            with np.load(path, allow_pickle=False) as data:
                n = len(data["queries"])
                if n < minimum:
                    raise ValueError(f"{row['id']}: only {n} queries; formal run requires {minimum}")
                counts[row["id"]] = n
        return counts

    def summary(self):
        return dict(manifest_sha256=self.manifest_sha,audit_sha256=self.audit_sha,
            fingerprint=self.doc["fingerprint"],registered=len(self.doc["records"]),
            qualified=len(self.rows),split_rows={k:len(v) for k,v in self.splits.items()},
            split_groups={k:len(self._group(v)) for k,v in self.splits.items()},
            cache_bytes=self.cache_bytes)


def autocast(device,amp):
    return torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16,enabled=amp=="bf16")


def sync(device):
    if str(device).startswith("cuda"): torch.cuda.synchronize(device)


def loss_for_batch(model,cache,labels,indices,jacobian_weight,device):
    q=torch.tensor(np.array(labels["queries"][indices],copy=True),device=device,dtype=torch.float64)
    target=torch.tensor(np.array(labels["moments"][indices],copy=True),device=device,dtype=torch.float64)
    pred,jac=model.moments_and_jacobian(cache,q,create_graph=True)
    moment=((pred-target)/cache.moment_scales).square().mean()
    valid=labels["derivative_valid"][indices].astype(bool)
    derivative=moment*0
    if valid.any():
        mask=torch.as_tensor(valid,device=device)
        jt=torch.tensor(np.array(labels["jacobian"][indices][valid],copy=True),device=device,dtype=torch.float64)
        scale=cache.moment_scales[:,None]*cache.coordinate_scales[None,:]
        derivative=((jac[mask]-jt)/scale).square().mean()
    return moment+jacobian_weight*derivative,moment,derivative


def atomic_save(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=None
    try:
        with tempfile.NamedTemporaryFile("wb",dir=path.parent,prefix=path.name+".",suffix=".tmp",delete=False) as f:
            tmp=Path(f.name); torch.save(payload,f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp is not None: tmp.unlink(missing_ok=True)


def validate(model,corpus,device,amp,chunk,max_records=8,max_queries=256):
    model.eval(); rows=[]; scores=[]; jac_scores=[]
    for row in corpus.splits["val"][:max_records]:
        geometry,labels=corpus.get(row)
        with torch.no_grad(),autocast(device,amp):
            cache=model.encode(geometry,chunk).detached()
        n=min(len(labels["queries"]),max_queries)
        q=torch.tensor(np.array(labels["queries"][:n],copy=True),device=device,dtype=torch.float64)
        target=torch.tensor(np.array(labels["moments"][:n],copy=True),device=device,dtype=torch.float64)
        with torch.enable_grad(),autocast(device,amp):
            pred,jac=model.moments_and_jacobian(cache,q,create_graph=False)
        ml=((pred-target)/cache.moment_scales).square().mean()
        valid=labels["derivative_valid"][:n].astype(bool)
        jl=ml*0
        if valid.any():
            mask=torch.as_tensor(valid,device=device)
            jt=torch.tensor(np.array(labels["jacobian"][:n][valid],copy=True),device=device,dtype=torch.float64)
            scale=cache.moment_scales[:,None]*cache.coordinate_scales[None,:]
            jl=((jac[mask]-jt)/scale).square().mean()
            jac_scores.append(float(jl.detach()))
        scores.append(float(ml.detach()))
        rows.append(dict(id=row["id"],group_id=row["group_id"],queries=n,
                         moment_scaled_mse=float(ml.detach()),jacobian_scaled_mse=float(jl.detach()),
                         derivative_valid=int(valid.sum())))
    model.train()
    return dict(moment_scaled_mse=float(np.mean(scores)),jacobian_scaled_mse=float(np.mean(jac_scores)) if jac_scores else None,
                records=len(rows),source_groups=len({r["group_id"] for r in rows}),details=rows)


def preflight(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if args.amp=="bf16" and args.device.startswith("cuda") and not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 unsupported")
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed); torch.cuda.reset_peak_memory_stats(args.device)
    cfg=make_config(args.preset,min_native_resolution=args.min_native_resolution,chunk_intervals=args.chunk)
    corpus=M72Corpus(args.manifest,args.audit,cfg.grid_size,min_queries=1,cache_mib=args.cache_mib)
    row=corpus.largest_train_row(); geometry,labels=corpus.get(row)
    valid=np.flatnonzero(labels["derivative_valid"].astype(bool))
    if len(valid)<1: raise ValueError("largest training geometry has no derivative-valid queries")
    indices=valid[:args.batch_queries] if len(valid)>=args.batch_queries else np.resize(valid,args.batch_queries)
    model=PcraftNativeMomentOperator(cfg).to(args.device).train()
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    if args.device.startswith("cuda"): torch.cuda.reset_peak_memory_stats(args.device)
    sync(args.device); start=perf_counter()
    with autocast(args.device,args.amp):
        cache=model.encode(geometry,args.chunk)
        loss,ml,jl=loss_for_batch(model,cache,labels,indices,args.jacobian_weight,args.device)
    if not torch.isfinite(loss): raise FloatingPointError("nonfinite preflight loss")
    opt.zero_grad(set_to_none=True); loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip,error_if_nonfinite=True)
    opt.step(); sync(args.device); seconds=perf_counter()-start
    model.eval()
    with torch.no_grad(),autocast(args.device,args.amp):
        c2=model.encode(geometry,args.chunk).detached()
        q=torch.tensor(np.array(labels["queries"][indices],copy=True),device=args.device,dtype=torch.float64)
        expected=model.predict(c2,q)
    tmp=Path(args.out).with_suffix(".roundtrip.pt")
    atomic_save(tmp,dict(schema=SCHEMA,semantics=SEMANTICS,config=asdict(cfg),
                         model={k:v.detach().cpu() for k,v in model.state_dict().items()}))
    restored,_=load_model(tmp,args.device); tmp.unlink(missing_ok=True)
    with torch.no_grad():
        c3=restored.encode(geometry,args.chunk).detached()
        actual=restored.predict(c3,q)
    parity=float((expected-actual).abs().max())
    if parity>1e-10: raise AssertionError(f"checkpoint round-trip mismatch {parity}")
    report=dict(schema="m80.pcraft-native-preflight.v1",status="passed",record_id=row["id"],
        native_intervals=len(geometry["xyz"]),native_resolution=int(geometry["native_resolution"]),
        parameter_count=sum(p.numel() for p in model.parameters()),config=asdict(cfg),
        batch_queries=len(indices),amp=args.amp,device=args.device,loss=float(loss.detach()),
        moment_loss=float(ml.detach()),jacobian_loss=float(jl.detach()),gradient_norm=float(grad),
        optimizer_step_seconds=seconds,checkpoint_roundtrip_max_abs=parity,
        peak_allocated=(torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None),
        peak_reserved=(torch.cuda.max_memory_reserved(args.device) if args.device.startswith("cuda") else None),
        data=corpus.summary(),source=source_fingerprints(),runtime=runtime_info(args.device,args.amp),
        limitations=["preflight is not a multi-day training result","query fit is not design-success proof",
                     "Spin-It Faster Newton bridge is a separate gate"])
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))


def train(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if args.amp=="bf16" and args.device.startswith("cuda") and not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 unsupported")
    if args.max_steps<1 or args.max_seconds<=0: raise ValueError("positive step/time budgets required")
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    if args.device.startswith("cuda"): torch.cuda.manual_seed_all(args.seed)
    rng=np.random.default_rng(args.seed)
    cfg=make_config(args.preset,min_native_resolution=args.min_native_resolution,chunk_intervals=args.chunk)
    corpus=M72Corpus(args.manifest,args.audit,cfg.grid_size,min_queries=args.min_queries_per_record,cache_mib=args.cache_mib)
    query_coverage=corpus.verify_query_coverage(args.min_queries_per_record)
    groups={s:corpus._group(corpus.splits[s]) for s in ("train","val","test")}
    if len(groups["train"])<args.min_train_groups or len(groups["val"])<args.min_val_groups or len(groups["test"])<args.min_test_groups:
        raise ValueError("formal corpus does not meet configured independent source-group minimums")
    model=PcraftNativeMomentOperator(cfg).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    contract=dict(schema=TRAIN_SCHEMA,config=asdict(cfg),manifest_sha256=corpus.manifest_sha,
        audit_sha256=corpus.audit_sha,corpus_fingerprint=corpus.doc["fingerprint"],source=source_fingerprints(),
        runtime=runtime_info(args.device,args.amp),seed=args.seed,batch_queries=args.batch_queries,
        chunk=args.chunk,lr=args.learning_rate,weight_decay=args.weight_decay,jacobian_weight=args.jacobian_weight,
        gradient_clip=args.gradient_clip,min_queries=args.min_queries_per_record)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    step=0; best_step=0; best_score=float("inf"); best_state=None; history=[]
    if args.resume:
        saved=torch.load(args.resume,map_location="cpu",weights_only=False)
        if saved.get("training_schema")!=TRAIN_SCHEMA or saved.get("contract")!=contract:
            raise ValueError("resume rejected: exact data/source/runtime/training contract mismatch")
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["optimizer"])
        step=saved["step"]; best_step=saved["best_step"]; best_score=saved["best_score"]
        best_state=saved["best_state"]; history=saved["history"]
        rng.bit_generator.state=saved["numpy_rng"]; torch.set_rng_state(saved["torch_rng"])
        if args.device.startswith("cuda"): torch.cuda.set_rng_state_all(saved["cuda_rng"])
    initial=validate(model,corpus,args.device,args.amp,args.chunk,args.val_max_records,args.val_max_queries)
    score=initial["moment_scaled_mse"]+args.jacobian_weight*(initial["jacobian_scaled_mse"] or 0.)
    if best_state is None: best_score=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    start=perf_counter(); stop="step_budget"
    def save():
        common=dict(schema=SCHEMA,semantics=SEMANTICS,config=asdict(cfg),contract=contract)
        last=common|dict(training_schema=TRAIN_SCHEMA,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer=opt.state_dict(),step=step,best_step=best_step,best_score=best_score,best_state=best_state,
            history=history,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [])
        atomic_save(out/"last.pt",last)
        atomic_save(out/"best.pt",common|dict(model=best_state,step=best_step,selection_score=best_score))
    while step<args.max_steps:
        if perf_counter()-start>=args.max_seconds:
            stop="walltime_budget"; break
        group=groups["train"][int(rng.integers(len(groups["train"])))]
        row=group[int(rng.integers(len(group)))]
        geometry,labels=corpus.get(row)
        idx=rng.choice(len(labels["queries"]),size=args.batch_queries,replace=False)
        opt.zero_grad(set_to_none=True)
        with autocast(args.device,args.amp):
            cache=model.encode(geometry,args.chunk)
            loss,ml,jl=loss_for_batch(model,cache,labels,idx,args.jacobian_weight,args.device)
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss at step {step+1}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip,error_if_nonfinite=True)
        opt.step(); step+=1
        history.append(dict(step=step,id=row["id"],group=row["group_id"],loss=float(loss.detach()),
                            moment=float(ml.detach()),jacobian=float(jl.detach())))
        if step%args.val_every==0:
            val=validate(model,corpus,args.device,args.amp,args.chunk,args.val_max_records,args.val_max_queries)
            score=val["moment_scaled_mse"]+args.jacobian_weight*(val["jacobian_scaled_mse"] or 0.)
            if score<best_score:
                best_score,best_step=score,step
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if step%args.save_every==0: save()
    final=validate(model,corpus,args.device,args.amp,args.chunk,args.val_max_records,args.val_max_queries)
    save()
    report=dict(schema=TRAIN_SCHEMA,steps_completed=step,stop_reason=stop,elapsed_seconds=perf_counter()-start,
        time_budget_scope="checked between complete optimizer steps; final validation/save is additional",
        best_step=best_step,best_validation_score=best_score,initial_validation=initial,final_validation=final,
        parameter_count=sum(p.numel() for p in model.parameters()),config=asdict(cfg),contract=contract,
        data=corpus.summary(),query_coverage_min=int(min(query_coverage.values())),query_coverage_max=int(max(query_coverage.values())),counters=model.counters,
        limitations=["validation is query-response selection, not design acceptance",
                     "no Faster Newton speedup is claimed until solver bridge tests pass"])
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd",required=True)
    for name in ("preflight","train"):
        q=sub.add_parser(name)
        q.add_argument("--manifest",type=Path,required=True); q.add_argument("--audit",type=Path,required=True)
        q.add_argument("--out",type=Path,required=True); q.add_argument("--preset",choices=tuple(PRESETS),default="xl")
        q.add_argument("--device",default="cpu"); q.add_argument("--amp",choices=("none","bf16"),default="none")
        q.add_argument("--chunk",type=int,default=2048); q.add_argument("--seed",type=int,default=800071)
        q.add_argument("--threads",type=int,default=1); q.add_argument("--cache-mib",type=float,default=512.)
        q.add_argument("--min-native-resolution",type=int,default=512); q.add_argument("--batch-queries",type=int,default=16)
        q.add_argument("--learning-rate",type=float,default=3e-4); q.add_argument("--weight-decay",type=float,default=1e-4)
        q.add_argument("--jacobian-weight",type=float,default=.05); q.add_argument("--gradient-clip",type=float,default=1.)
        if name=="train":
            q.add_argument("--max-steps",type=int,required=True); q.add_argument("--max-seconds",type=float,required=True)
            q.add_argument("--min-queries-per-record",type=int,default=256)
            q.add_argument("--min-train-groups",type=int,default=50); q.add_argument("--min-val-groups",type=int,default=7)
            q.add_argument("--min-test-groups",type=int,default=10); q.add_argument("--save-every",type=int,default=25)
            q.add_argument("--val-every",type=int,default=20); q.add_argument("--val-max-records",type=int,default=8)
            q.add_argument("--val-max-queries",type=int,default=256); q.add_argument("--resume",type=Path)
    args=p.parse_args()
    return preflight(args) if args.cmd=="preflight" else train(args)


if __name__=="__main__": main()
