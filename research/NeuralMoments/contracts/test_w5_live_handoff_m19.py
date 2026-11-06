"""M19 regression: actual reviewed W5 adapter bytes and live remote identity gate.

The migrated-case test is a *fixture copied from W0*, not evidence of a W5
merge. Production CI separately reads current W5 and W2 refs from git show.
"""
from __future__ import annotations
import copy
from pathlib import Path
import json
import pytest
from . import w5_live_handoff_m19 as gate

ROOT=Path(__file__).resolve().parent
M18=ROOT/"migration_proposals"/"w5_contract_gate_v2_m18.py"
W5_LEGACY=ROOT/"frozen_external_m16"/"w5_contract_gate.py"
W2_FROZEN=ROOT/"frozen_external_m16"/"w2_teacher_manifest.json"


def write_inputs(tmp_path, monkeypatch):
    # A minimal *synthetic* W2 v2 manifest with the archived exclusion states.
    manifest={"schema":gate.W2_SCHEMA,"native_teacher_runs":0,
              "training_samples":0,"data_splits":{
                   "train":[],"val":[],"test":[],
                   "excluded":[{"count":81,"derivatives_admitted":0}]}}
    w2=tmp_path/"w2.json"
    w2.write_text(json.dumps(manifest))
    w5=tmp_path/"legacy.py"
    w5.write_text('"""Synthetic W5 pre-migration fixture."""\n')
    candidate=tmp_path/"candidate.py"
    candidate.write_text('"""Synthetic candidate for identity-only tests."""\n')
    monkeypatch.setattr(gate,"LEGACY_W5_BLOB",gate.git_blob(w5.read_bytes()))
    monkeypatch.setattr(gate,"REVIEWED_M18_BLOB",gate.git_blob(candidate.read_bytes()))
    args=dict(source_repository="volcano-code/Spin_test2",w5_source=w5,w5_head="a"*40,w5_blob=gate.git_blob(w5.read_bytes()),
              w2_manifest=w2,w2_head="b"*40,w2_blob=gate.git_blob(w2.read_bytes()),
              reviewed_candidate=candidate)
    return args,manifest


def test_current_pending_is_not_training_release(tmp_path,monkeypatch):
    args,_=write_inputs(tmp_path,monkeypatch)
    result=gate.inspect(**args)
    assert result["status"]=="PENDING_W5_OWNER_ADOPTION"
    assert not result["W5"]["reviewed_candidate_applied"]
    assert result["W2"]["excluded_h8f_samples"]==81
    assert not result["training_admitted"] and not result["Newton_accepted"]


@pytest.mark.parametrize("repository", ["volcano-code/Spin_test", "volcano-code/Spin_test2"])
def test_receipt_records_the_repository_that_supplied_both_refs(tmp_path, monkeypatch, repository):
    args, _ = write_inputs(tmp_path, monkeypatch)
    args["source_repository"] = repository
    result = gate.inspect(**args)
    assert result["source_repository"] == result["source_branch"] == repository


def test_malformed_repository_rejected(tmp_path, monkeypatch):
    args, _ = write_inputs(tmp_path, monkeypatch)
    args["source_repository"] = "refs/heads/main"
    with pytest.raises(gate.HandoffRejected, match="INVALID_SOURCE_REPOSITORY"):
        gate.inspect(**args)


def test_unrecognized_w5_source_rejected_without_code_execution(tmp_path,monkeypatch):
    args,_=write_inputs(tmp_path,monkeypatch)
    args["w5_source"].write_text('"""unreviewed file"""\n')
    args["w5_blob"]=gate.git_blob(args["w5_source"].read_bytes())
    with pytest.raises(gate.HandoffRejected,match="W5_UNREVIEWED_REMOTE_SOURCE_DRIFT"):
        gate.inspect(**args)


def test_w5_source_bytes_must_match_git_blob(tmp_path,monkeypatch):
    args,_=write_inputs(tmp_path,monkeypatch)
    args["w5_blob"]="d"*40
    with pytest.raises(gate.HandoffRejected,match="W5_REMOTE_GIT_BLOB_MISMATCH"):
        gate.inspect(**args)


def test_w2_manifest_bytes_must_match_git_blob(tmp_path,monkeypatch):
    args,_=write_inputs(tmp_path,monkeypatch)
    args["w2_blob"]="d"*40
    with pytest.raises(gate.HandoffRejected,match="W2_REMOTE_GIT_BLOB_MISMATCH"):
        gate.inspect(**args)


@pytest.mark.parametrize("updates",[
    {"native_teacher_runs":1},
    {"training_samples":1},
    {"schema":"w2-cherchi-teacher-v1-pre-w0"},
    {"data_splits":{"train":[{"fake":True}],"val":[],"test":[],"excluded":[{"count":81}]}},
    {"data_splits":{"train":[],"val":[],"test":[],"excluded":[{"count":82}]}},
    {"native_teacher_runs":True},
])
def test_upstream_w2_positive_claim_does_not_promote(tmp_path,monkeypatch,updates):
    args,manifest=write_inputs(tmp_path,monkeypatch)
    new=copy.deepcopy(manifest);new.update(updates)
    args["w2_manifest"].write_text(json.dumps(new))
    args["w2_blob"]=gate.git_blob(args["w2_manifest"].read_bytes())
    with pytest.raises(gate.HandoffRejected): gate.inspect(**args)


def test_duplicate_json_keys_rejected():
    with pytest.raises(gate.HandoffRejected,match="DUPLICATED_EXTERNAL_JSON_KEY"):
        gate.no_duplicate_json('{"x":1,"x":2}')


def test_nonfinite_json_values_rejected():
    with pytest.raises(gate.HandoffRejected,match="NONFINITE_EXTERNAL_JSON"):
        gate.no_duplicate_json('{"x":NaN}')


def test_file_path_symlink_rejected(tmp_path,monkeypatch):
    args,_=write_inputs(tmp_path,monkeypatch)
    p=tmp_path/"alias.py";p.symlink_to(args["w5_source"])
    args["w5_source"]=p
    with pytest.raises(gate.HandoffRejected,match="UNAVAILABLE_UNSAFE_OR_OVERSIZED_FILE"):
        gate.inspect(**args)


def test_record_must_be_new(tmp_path,monkeypatch):
    args,_=write_inputs(tmp_path,monkeypatch)
    out=tmp_path/"old.json";out.write_text("original")
    with pytest.raises(FileExistsError,match="DO_NOT_OVERWRITE"):
        gate.main(["--source-repository",args["source_repository"],"--w5-source",str(args["w5_source"]),"--w5-head",args["w5_head"],
                   "--w5-blob",args["w5_blob"],"--w2-manifest",str(args["w2_manifest"]),
                   "--w2-head",args["w2_head"],"--w2-blob",args["w2_blob"],
                   "--reviewed-candidate",str(args["reviewed_candidate"]),
                   "--output",str(out)])
    assert out.read_text()=="original"


def test_actual_old_w5_source_is_pending_and_no_release():
    if not M18.exists() or not W5_LEGACY.exists() or not W2_FROZEN.exists():
        pytest.skip("actual frozen W5/W2 snapshots available on complete W0 checkout")
    assert gate.git_blob(W5_LEGACY.read_bytes())==gate.LEGACY_W5_BLOB
    assert gate.git_blob(M18.read_bytes())==gate.REVIEWED_M18_BLOB
    result=gate.inspect(source_repository="volcano-code/Spin_test",w5_source=W5_LEGACY,w5_head="a"*40,
                        w5_blob=gate.LEGACY_W5_BLOB,
                        w2_manifest=W2_FROZEN,w2_head="b"*40,
                        w2_blob=gate.git_blob(W2_FROZEN.read_bytes()),
                        reviewed_candidate=M18)
    assert result["status"]=="PENDING_W5_OWNER_ADOPTION"
    assert not result["training_admitted"]


def test_reviewer_can_replay_real_m18_candidate_without_releasing_labels(tmp_path):
    if not M18.exists() or not W2_FROZEN.exists():
        pytest.skip("reviewed exact W5 candidate and frozen W2 manifest required")
    assert gate.git_blob(M18.read_bytes())==gate.REVIEWED_M18_BLOB
    mirror=tmp_path/"w5_merged_candidate.py"
    mirror.write_bytes(M18.read_bytes())  # SIMULATED adopted branch, not actual W5.
    result=gate.inspect(source_repository="volcano-code/Spin_test",w5_source=mirror,w5_head="a"*40,
       w5_blob=gate.REVIEWED_M18_BLOB,
       w2_manifest=W2_FROZEN,w2_head="b"*40,
       w2_blob=gate.git_blob(W2_FROZEN.read_bytes()),reviewed_candidate=M18)
    assert result["status"]=="REVIEWED_W5_M18_SOURCE_PRESENT_ON_W5_BRANCH"
    assert result["runtime"]["w0_norm_false_positive_removed"]
    assert result["runtime"]["forged_train_split_rejected"]
    assert not result["W5_full_branch_tests_verified_by_W0"]
    assert not result["training_admitted"]


def test_m21_excluded_receipt_tracks_the_same_w2_commit(tmp_path, monkeypatch):
    args, manifest = write_inputs(tmp_path, monkeypatch)
    manifest.update(native_teacher_runs=1, latest_native_diagnostic={
        "J_D": None, "split": "excluded", "training_admitted": False})
    args["w2_manifest"].write_text(json.dumps(manifest))
    args["w2_blob"] = gate.git_blob(args["w2_manifest"].read_bytes())
    monkeypatch.setattr(gate, "M21_MANIFEST_BLOB", args["w2_blob"])
    receipt = gate.subprocess.CompletedProcess([], 0, stdout=gate.M21_RECEIPT_BLOB + "\n")
    def git_lookup(command, **kwargs):
        assert command == ["git", "rev-parse", "--verify", args["w2_head"] + ":" + gate.M21_RECEIPT_PATH]
        return receipt
    monkeypatch.setattr(gate.subprocess, "run", git_lookup)
    result = gate.inspect(**args)
    assert result["W2"]["native_teacher_runs"] == 1
    assert result["W2"]["training_samples"] == 0
    assert result["W2"]["reviewed_excluded_receipt_blob"] == gate.M21_RECEIPT_BLOB
    assert not result["W2"]["raw_diagnostic_archive_verified_by_W0"]
    assert not result["training_admitted"] and not result["W0_full_scientific_admission_certified"]
    assert not result["Newton_accepted"] and not result["W5_native_teacher_data_released"]

    # A changed receipt at this head is rejected despite the unchanged manifest.
    receipt.stdout = "f"*40
    with pytest.raises(gate.HandoffRejected, match="M21_RECEIPT"):
        gate.inspect(**args)
    receipt.returncode = 128
    receipt.stdout = ""
    with pytest.raises(gate.HandoffRejected, match="M21_RECEIPT"):
        gate.inspect(**args)

    # Even a correctly rehashed document loses its reviewed identity.
    for update in ({"native_teacher_runs": 2}, {"native_teacher_runs": 0},
                   {"latest_native_diagnostic": {"J_D": [], "split": "train"}},
                   {"training_samples": 1}):
        args["w2_manifest"].write_text(json.dumps({**manifest, **update}))
        args["w2_blob"] = gate.git_blob(args["w2_manifest"].read_bytes())
        with pytest.raises(gate.HandoffRejected, match="W2_NATIVE_RELEASE_CHANGED"):
            gate.inspect(**args)


@pytest.mark.parametrize("version", ["P1", "P2"])
def test_m22_receipts_are_bound_to_same_commit(tmp_path, monkeypatch, version):
    args, manifest = write_inputs(tmp_path, monkeypatch)
    manifest.update(native_teacher_runs=1, latest_native_diagnostic={"split": "excluded"})
    args["w2_manifest"].write_text(json.dumps(manifest))
    args["w2_blob"] = gate.git_blob(args["w2_manifest"].read_bytes())
    monkeypatch.setattr(gate, f"M22_{version}_MANIFEST_BLOB", args["w2_blob"])
    receipts = {gate.M21_RECEIPT_PATH: gate.M21_RECEIPT_BLOB,
                gate.M22_AUDIT_PATH: gate.M22_AUDIT_BLOB}
    if version == "P2":
        receipts[gate.M22_REPLAY_PATH] = gate.M22_REPLAY_BLOB

    def lookup(command, **kwargs):
        head, path = command[-1].split(":")
        assert head == args["w2_head"]
        return gate.subprocess.CompletedProcess(command, 0 if path in receipts else 128,
                                                stdout=receipts.get(path, ""))

    monkeypatch.setattr(gate.subprocess, "run", lookup)
    result = gate.inspect(**args)
    assert result["W2"]["reviewed_archive_audit_blob"] == gate.M22_AUDIT_BLOB
    assert result["W2"]["reviewed_historical_boundary_replay_blob"] == receipts.get(gate.M22_REPLAY_PATH)
    assert result["W2"]["training_samples"] == 0
    assert not result["training_admitted"] and not result["Newton_accepted"]
    assert not result["W0_full_scientific_admission_certified"]
    assert not result["W5_native_teacher_data_released"]
    for path, expected in receipts.copy().items():
        receipts[path] = "f" * 40
        with pytest.raises(gate.HandoffRejected, match="REQUIRES_INDEPENDENT_REVIEW"):
            gate.inspect(**args)
        del receipts[path]
        with pytest.raises(gate.HandoffRejected, match="REQUIRES_INDEPENDENT_REVIEW"):
            gate.inspect(**args)
        receipts[path] = expected
