from __future__ import annotations

import hashlib
import json
import numpy as np
import pytest

from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train import M72Corpus


def fingerprint(doc):
    body={k:v for k,v in doc.items() if k!="fingerprint"}
    raw=json.dumps(body,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_corpus(root):
    records=[]
    for split in ("train","val","test"):
        gp=root/f"{split}-geometry.npz"; lp=root/f"{split}-labels.npz"
        cols=np.asarray([(x,y,1,7) for x in range(1,7) for y in range(1,7)],dtype=np.int32)
        np.savez(gp,columns=cols,origin=np.asarray([-1.,-1.,-1.]),h=np.asarray(.25),
            side=np.asarray(8,dtype=np.int32),fixed_moments=np.zeros(10),density=np.asarray(1.))
        q=np.asarray([[.1,.1,0,0,.01,0,0,0,0,0],[-.1,.1,0,0,.01,0,0,0,0,0],
            [.1,0,.1,0,0,.01,0,0,0,0],[-.1,0,.1,0,0,.01,0,0,0,0]],dtype=np.float64)
        np.savez(lp,queries=q,moments=np.zeros((4,10)),jacobian=np.zeros((4,10,10)),
                 derivative_valid=np.asarray([True,True,False,True]))
        records.append(dict(id=f"{split}-0",status="qualified",split=split,group_id=f"group-{split}",
            geometry_file=gp.name,geometry_sha256=sha(gp),labels_file=lp.name,labels_sha256=sha(lp)))
    doc=dict(schema="m69.corpus.v1",scope="formal",frozen=True,records=records)
    doc["fingerprint"]=fingerprint(doc)
    mp=root/"operator-manifest.json"; mp.write_text(json.dumps(doc,sort_keys=True))
    audit=dict(schema="m69.corpus-audit.v1",valid=True,ready_for_training=True,
        fingerprint=doc["fingerprint"],scope="formal",blockers=[],counts=dict(qualified_tasks=3))
    ap=root/"audit.json"; ap.write_text(json.dumps(audit))
    return mp,ap


def test_m72_manifest_audit_hashes_and_native_geometry(tmp_path):
    manifest,audit=make_corpus(tmp_path)
    corpus=M72Corpus(manifest,audit,grid_size=8,min_queries=4)
    assert len(corpus.rows)==3 and len(corpus.train_groups)==1
    geometry,labels=corpus.get(corpus.splits["train"][0])
    assert int(geometry["native_resolution"])==8
    assert geometry["pcraft_grid"].shape==(4,8,8,8)
    assert labels["queries"].shape==(4,10)
    assert int(labels["derivative_valid"].sum())==3
    assert len(corpus.verify_query_coverage(4))==3


def test_m72_audit_not_ready_is_rejected(tmp_path):
    manifest,audit=make_corpus(tmp_path)
    doc=json.loads(audit.read_text()); doc["ready_for_training"]=False; doc["blockers"]=["fixture"]
    audit.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="audit"):
        M72Corpus(manifest,audit,grid_size=8,min_queries=1)


def test_manifest_path_traversal_is_rejected(tmp_path):
    manifest,audit=make_corpus(tmp_path)
    doc=json.loads(manifest.read_text()); doc["records"][0]["geometry_file"]="../escape.npz"
    doc["fingerprint"]=fingerprint(doc); manifest.write_text(json.dumps(doc,sort_keys=True))
    a=json.loads(audit.read_text()); a["fingerprint"]=doc["fingerprint"]; audit.write_text(json.dumps(a))
    corpus=M72Corpus(manifest,audit,grid_size=8,min_queries=1)
    with pytest.raises(ValueError,match="escapes"):
        corpus.get(corpus.splits["train"][0])


def test_cpu_preflight_runs_real_pcraft_m72_backward_and_checkpoint_roundtrip(tmp_path):
    from argparse import Namespace
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train import preflight
    manifest,audit=make_corpus(tmp_path)
    args=Namespace(
        device="cpu", amp="none", threads=1, seed=800071, preset="smoke",
        min_native_resolution=1, manifest=manifest, audit=audit, cache_mib=16.,
        batch_queries=2, chunk=16, learning_rate=3e-4, weight_decay=1e-4,
        jacobian_weight=.05, gradient_clip=1., out=tmp_path/"preflight.json",
    )
    report=preflight(args)
    assert report["status"]=="passed"
    assert report["selected_native_intervals"]>0
    assert report["checkpoint_roundtrip_max_abs"]<=1e-10
    assert report["parameter_count"]>0


def test_cpu_preflight_runs_real_pcraft_m72_backward_and_checkpoint_roundtrip(tmp_path):
    from argparse import Namespace
    from research.NeuralMoments.models.pcraft_ufno.spin_pino.native_train import preflight
    manifest,audit=make_corpus(tmp_path)
    args=Namespace(device="cpu",amp="none",threads=1,seed=800071,preset="smoke",
        min_native_resolution=1,manifest=manifest,audit=audit,cache_mib=16.,
        batch_queries=2,chunk=16,learning_rate=3e-4,weight_decay=1e-4,
        jacobian_weight=.05,gradient_clip=1.,out=tmp_path/"preflight.json")
    report=preflight(args)
    assert report["status"]=="passed"
    assert report["selected_native_intervals"]>0
    assert report["checkpoint_roundtrip_max_abs"]<=1e-10
    assert report["parameter_count"]>0
