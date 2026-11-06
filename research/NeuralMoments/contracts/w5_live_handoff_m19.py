"""W0 M19: fetch-independent W5 source and W2 manifest provenance gate.

Caller (CI) obtains immutable HEAD:blob pairs using git rev-parse/show from
actual remote refs. Old W5 source yields PENDING; exact reviewed M18 bytes
allow a *structural-only* smoke; unknown W5 source or new W2 release claim
fails closed. This never grants native labels or training.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tempfile

LEGACY_W5_BLOB = "14b2ab225a032188fe16570414ef321af9a81e00"
REVIEWED_M18_BLOB = "4c97aacc1d50366c7cc97876064f4183469d9910"
# Reviewed together at W2 04ea5c86: terminal-transcribed, excluded Heart
# diagnostic. Neither hash is a replacement for replaying the raw archive.
M21_MANIFEST_BLOB = "97d1576b3fbfbdcd557bf17c4319c17b1f65f78f"
M21_RECEIPT_BLOB = "b7938b4d76822079df72b46ef7d9c7b88d66c1d5"
M21_RECEIPT_PATH = "research/NeuralMoments/teacher/M21_HEART_NATIVE_DIAGNOSTIC_RESULT.json"
M22_P1_MANIFEST_BLOB = "1d88d8071d1b7c8ad5e17959f697db8972461e3e"
M22_AUDIT_PATH = "research/NeuralMoments/teacher/M22_M21_HEART_ARCHIVE_AUDIT_20261106.json"
M22_AUDIT_BLOB = "d92d475641ca46501e5360fa82a72d3b7c8cb20c"
M22_P2_MANIFEST_BLOB = "973a796b1720ed2b6e0e844a758b0325cd530d65"
M22_REPLAY_PATH = "research/NeuralMoments/teacher/M22_P2_HEART_BOUNDARY_REPLAY_20261106.json"
M22_REPLAY_BLOB = "feb06c694345030a2a123b2e2f84b1eaca442453"
W2_SCHEMA = "w2-cherchi-teacher-v2-pre-w0"
W0_UNIT_TOLERANCE = 1e-12
REQUIRED_W2_EXCLUDED = 81
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
W5_BRANCH = "research/neuralmoments-w5-deeponet-20261019"
W2_BRANCH = "research/neuralmoments-w2-teacher-20261019"


class HandoffRejected(ValueError):
    pass


def demand(condition, code):
    if not condition:
        raise HandoffRejected(code)


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def no_duplicate_json(raw):
    def unique(rows):
        data = {}
        for key, value in rows:
            demand(key not in data, "DUPLICATED_EXTERNAL_JSON_KEY")
            data[key] = value
        return data
    def invalid(token):
        raise HandoffRejected("NONFINITE_EXTERNAL_JSON:" + token)
    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    except (ValueError, UnicodeError, TypeError) as exc:
        if isinstance(exc, HandoffRejected):
            raise
        raise HandoffRejected("INVALID_EXTERNAL_JSON") from exc


def regular_bytes(path, *, max_size=300000):
    path = Path(path)
    demand(path.is_file() and not path.is_symlink() and
           0 < path.stat().st_size <= max_size, "UNAVAILABLE_UNSAFE_OR_OVERSIZED_FILE")
    return path.read_bytes()


def _real_m18_runtime(candidate_path, original_bytes):
    """Only load a reviewed SHA-exact W5 module, never unknown remote code."""
    import numpy as np
    demand(git_blob(original_bytes) == REVIEWED_M18_BLOB,
           "W5_REMOTE_SOURCE_NOT_THE_REVIEWED_M18_CODE")
    spec = importlib.util.spec_from_file_location("w0_m19_exact_w5_candidate", candidate_path)
    demand(spec is not None and spec.loader is not None, "REVIEWED_W5_IMPORT_UNAVAILABLE")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    demand(hasattr(mod, "inspect_w2_v2_excluded_record"), "M18_V2_REVIEW_FUNCTION_MISSING")
    # Real runtime counterexample from the source-bound W0 M17 defect.
    # The reviewed W5 module must reject it at exactly the W0 <=1e-12 gate.
    example = {
        "schema": "MomentContractV1",
        "source_sha": "0bcc187b3005478baa586a6d6f450e73811ca342",
        "geometry": {
            "original_sha256": "1"*64, "fixed_F_sha256":"2"*64,
            "design_D_sha256":"3"*64,
            "density": 1.0, "fixed_r_computed_once":True,
            "frame": {"kind":"INPUT_NORMALIZED_TARGET_V1",
                      "length_original_per_normalized":1.0,
                      "density_original_per_normalized":1.0,
                      "origin_original":[0.0,0.0,0.0]}},
        "coefficients": {"c":[1.+4.e-10]+[0.]*9,
                         "order":"CONST,X,Y,Z,XX,YY,ZZ,YZ,XZ,XY",
                         "positive_scale_equivalent": True,
                         "negative_sign_equivalent": False,
                         "retained_side":"BETA_POSITIVE",
                         "cross_mapping":"q=Bc;Bdiag=1,1,1,1,1,1,1,2,2,2"},
        "moments": {"order":"M,LX,LY,LZ,QXX,QYY,QZZ,QYZ,QXZ,QXY",
                    "status":"NOT_RUN","fixed_r":[0.]*10,
                    "variable_r":None,"total_r":None},
        "jacobian": {"type":"UNAVAILABLE","derivative_defined":False,
                     "values":None,"native_tangent_basis":None,
                     "hessian_rule":"ONLY_IF_SMOOTH_AMBIENT_CHECK_Jc_B_INVERSE"}}
    rejected = False
    try:
        mod.check_w0_contract(example)
    except mod.AdmissionError:
        rejected = True
    demand(rejected, "M18_W5_STILL_ACCEPTS_NONUNIT_NATIVE_C")
    example["coefficients"]["c"]=[1.]+[0.]*9
    result = mod.check_w0_contract(example)
    demand(isinstance(result, dict) and result.get("j_type")=="UNAVAILABLE",
           "REVIEWED_W5_REJECTS_W0_VALID_CASE")
    # Excluded-only W2 v2 data may be inspected without exporting r/J labels.
    c = [1.000004]+[0.]*6+[3e-6,-4e-6,5e-6]
    geometry = {"family":"reconstructed-h8", "original_sha256":"a"*64}
    payload = {
        "schema":W2_SCHEMA,"order":["m","x","y","z","xx","yy","zz","yz","xz","xy"],
        "coefficient_convention":"NATIVE_MATRIX_ENTRIES_C10_POLYNOMIAL_Q_EQUALS_BC_V1",
        "native_c_order":"CONST,X,Y,Z,XX,YY,ZZ,YZ,XZ,XY".split(","),
        "native_c":c,"q":c[:],
        "polynomial_q":[v*(2 if i>=7 else 1) for i,v in enumerate(c)],
        "model_sha256":"b"*64,"native_module_sha256":"c"*64,
        "geometry":geometry,"density":1.0,
        "frame":{"kind":"INPUT_NORMALIZED_TARGET_V1",
                 "length_original_per_normalized":1.,
                 "density_original_per_normalized":1.,
                 "origin_original":[0.,0.,0.]},
        "r_D":[0.]*10,"r_F":[0.]*10,"J_D":None,
        "topology_status":"OBSERVED_DISCRETE_MAP",
        "derivative_status":"UNKNOWN","derivative_contract":None,
        "stage_seconds":{"native":0.0},"peak_rss_bytes":0,
        "exact_proof":{},"split":"excluded","failure_reason":"not-qualified",
        "total_policy":"r_total=r_F+r_D;J_total=J_D;fixed_never_in_r_D"}
    record={"payload":payload,"payload_sha256":mod.digest(payload),
            "identity":{"q_sha256":mod.digest(c),
                        "geometry_sha256":mod.digest(geometry),
                        "model_sha256":"b"*64,"native_module_sha256":"c"*64}}
    with tempfile.TemporaryDirectory(prefix="w0-m19-v2-") as tmp:
        file = Path(tmp)/"excluded.json"
        file.write_text(mod.canonical(record),encoding="utf-8")
        checked = mod.inspect_w2_v2_excluded_record(
            file, expected_native_c=c,expected_model_sha256="b"*64,
            expected_native_module_sha256="c"*64,
            expected_geometry_sha256=mod.digest(geometry))
        demand(checked["training_admitted"] is False and
               checked["derivative_J_D"] is None and
               checked["native_material_certified"] is False and
               "r_D" not in checked and "J_native" not in checked,
               "M18_V2_EXCLUDED_DATA_LEAKED_TO_TRAINING")
        record["payload"]["split"]="train"
        record["payload_sha256"] = mod.digest(record["payload"])
        file.write_text(mod.canonical(record),encoding="utf-8")
        rejected = False
        try:
            mod.inspect_w2_v2_excluded_record(
                file,expected_native_c=c,expected_model_sha256="b"*64,
                expected_native_module_sha256="c"*64,
                expected_geometry_sha256=mod.digest(geometry))
        except mod.AdmissionError:
            rejected=True
        demand(rejected,"M18_FORGED_TRAIN_SPLIT_ACCEPTED")
    return {"actual_W5_candidate_runtime_exercised":True,
            "w0_norm_false_positive_removed":True,
            "valid_w0_structural_record_accepted":True,
            "excluded_w2_v2_readonly_only":True,
            "forged_train_split_rejected":True,
            "test_data_are_synthetic_not_native_training":True}


def inspect(*, source_repository, w5_source, w5_head, w5_blob, w2_manifest, w2_head, w2_blob,
            reviewed_candidate):
    demand(isinstance(source_repository, str) and
           re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+", source_repository),
           "INVALID_SOURCE_REPOSITORY")
    for label, head in (("W5",w5_head),("W2",w2_head),
                        ("W5_BLOB",w5_blob),("W2_BLOB",w2_blob)):
        demand(isinstance(head,str) and HEX40.fullmatch(head),
               "INVALID_GIT_IDENTITY:"+label)
    w5_raw=regular_bytes(w5_source)
    w2_raw=regular_bytes(w2_manifest)
    proposal_raw=regular_bytes(reviewed_candidate)
    demand(git_blob(w5_raw)==w5_blob, "W5_REMOTE_GIT_BLOB_MISMATCH")
    demand(git_blob(w2_raw)==w2_blob, "W2_REMOTE_GIT_BLOB_MISMATCH")
    demand(git_blob(proposal_raw)==REVIEWED_M18_BLOB,
           "W0_REVIEWED_M18_PROPOSAL_BYTE_DRIFT")
    w2=no_duplicate_json(w2_raw)
    demand(type(w2) is dict and w2.get("schema")==W2_SCHEMA,
           "UNREVIEWED_W2_FORMAT_CHANGE")
    splits=w2.get("data_splits")
    demand(type(splits) is dict and set(splits)=={"train","val","test","excluded"}
           and all(type(v) is list for v in splits.values()),
           "W2_SPLITS_SCHEMA_CHANGED")
    counts = [x.get("count") for x in splits["excluded"] if type(x) is dict]
    runs = w2.get("native_teacher_runs")
    archive_audited = w2_blob in (M22_P1_MANIFEST_BLOB, M22_P2_MANIFEST_BLOB)
    boundary_replayed = w2_blob == M22_P2_MANIFEST_BLOB
    demand(counts==[REQUIRED_W2_EXCLUDED] and
           not splits["train"] and not splits["val"] and not splits["test"] and
           type(runs) is int and
           ((runs == 0 and "latest_native_diagnostic" not in w2) or
            (runs == 1 and (w2_blob == M21_MANIFEST_BLOB or archive_audited))) and
           type(w2.get("training_samples")) is int and
           w2["training_samples"]==0,
           "W2_NATIVE_RELEASE_CHANGED_REQUIRES_INDEPENDENT_REVIEW")
    # Every receipt must belong to the same fetched W2 commit as its manifest.
    required_receipts = []
    if runs == 1:
        required_receipts.append((M21_RECEIPT_PATH, M21_RECEIPT_BLOB,
                                  "W2_M21_RECEIPT_REQUIRES_INDEPENDENT_REVIEW"))
    if archive_audited:
        required_receipts.append((M22_AUDIT_PATH, M22_AUDIT_BLOB,
                                  "W2_M22_AUDIT_REQUIRES_INDEPENDENT_REVIEW"))
    if boundary_replayed:
        required_receipts.append((M22_REPLAY_PATH, M22_REPLAY_BLOB,
                                  "W2_M22_REPLAY_REQUIRES_INDEPENDENT_REVIEW"))
    for path, expected, rejection in required_receipts:
        receipt = subprocess.run(["git", "rev-parse", "--verify", w2_head + ":" + path],
                                 capture_output=True, text=True)
        demand(receipt.returncode == 0 and receipt.stdout.strip() == expected, rejection)
    if w5_blob==LEGACY_W5_BLOB:
        status="PENDING_W5_OWNER_ADOPTION"
        smoke={"actual_W5_candidate_runtime_exercised":False}
    elif w5_blob==REVIEWED_M18_BLOB:
        status="REVIEWED_W5_M18_SOURCE_PRESENT_ON_W5_BRANCH"
        smoke=_real_m18_runtime(w5_source,w5_raw)
    else:
        raise HandoffRejected("W5_UNREVIEWED_REMOTE_SOURCE_DRIFT")
    return {"schema":"W0_M19_LIVE_W5_W2_CONSUMER_HANDOFF_V1",
            "status":status,
            "source_repository":source_repository,
            "source_branch":source_repository,  # Legacy field for the repository slug in v1 receipts.
            "W5":{"branch":W5_BRANCH,"commit_sha":w5_head,"source_git_blob":w5_blob,
                  "reviewed_candidate_applied":status.startswith("REVIEWED_")},
            "W2":{"branch":W2_BRANCH,"commit_sha":w2_head,"manifest_git_blob":w2_blob,
                  "schema":w2["schema"],"excluded_h8f_samples":81,
                  "native_teacher_runs":runs,"training_samples":0,
                  "reviewed_excluded_receipt_blob": M21_RECEIPT_BLOB if runs else None,
                  "reviewed_archive_audit_blob": M22_AUDIT_BLOB if archive_audited else None,
                  "reviewed_historical_boundary_replay_blob": M22_REPLAY_BLOB if boundary_replayed else None,
                  "raw_diagnostic_archive_verified_by_W0":False},
            "runtime":smoke,
            "scope":"CODE_IDENTITY_AND_STRUCTURAL_COMPATIBILITY_NOT_NATIVE_TRUTH",
            "W5_full_branch_tests_verified_by_W0":False,
            "W0_full_scientific_admission_certified":False,
            "training_admitted":False,
            "Newton_accepted":False,
            "W5_native_teacher_data_released":False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-repository",required=True,help="GitHub owner/repo used to fetch both refs")
    p.add_argument("--w5-source",required=True,type=Path)
    p.add_argument("--w5-head",required=True)
    p.add_argument("--w5-blob",required=True)
    p.add_argument("--w2-manifest",required=True,type=Path)
    p.add_argument("--w2-head",required=True)
    p.add_argument("--w2-blob",required=True)
    p.add_argument("--reviewed-candidate",type=Path,
                   default=Path(__file__).parent/"migration_proposals"/"w5_contract_gate_v2_m18.py")
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError("DO_NOT_OVERWRITE_LIVE_REVIEW")
    result=inspect(**{k:getattr(a,k) for k in (
        "source_repository","w5_source","w5_head","w5_blob","w2_manifest","w2_head","w2_blob",
        "reviewed_candidate")})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print(result["status"],"NO_TRAINING_RELEASE")
    return 0 # PENDING is an expected, explicitly reported dependency.

if __name__=="__main__":
    raise SystemExit(main())
