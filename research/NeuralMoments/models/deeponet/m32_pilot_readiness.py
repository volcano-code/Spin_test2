"""Reproduce the blocked W5 D1 pilot inputs without running an optimizer."""
import argparse
from pathlib import Path
from math import isclose
import json

from .contract_gate import file_sha256, strict_json, W2_V2_SCHEMA, AdmissionError
from .contract_gate import audit_family_split


def readiness(manifest_path, witness_path):
    protocol_path=Path(__file__).with_name("M32_PAIRED_PILOT.json")
    p=strict_json(protocol_path)
    for path, key in ((manifest_path,"w2_manifest"), (witness_path,"w4_reference")):
        if file_sha256(path)!=p[key]["sha256"]:
            raise AdmissionError(f"pinned {key} changed")
    manifest, witness = strict_json(manifest_path), strict_json(witness_path)
    splits=manifest["data_splits"]
    if manifest["schema"]!=W2_V2_SCHEMA or manifest["training_samples"]!=0 or any(splits[k] for k in ("train","val","test")):
        raise AdmissionError("independent admission and protocol revision required")
    if p["admitted_real_training_hashes"] or p["admitted_real_evaluation_hashes"] or p["completed_real_runs"]:
        raise AdmissionError("empty real-data and result ledger required by this preregistration")
    start, restarts = p["start_list"]
    if (restarts["mode"],restarts["deterministic_seed"],restarts["maximum_restarts"],restarts["source_git_blob"]) != (
            "SPINNING_TOP",0,8,"93862501b6e3c27f44ceadb8d48a5c713619bbeb"):
        raise AdmissionError("unregistered restart schedule")
    c=start["c_native"]
    if c!=witness[start["source_field"]] or len(c)!=10 or not isclose(sum(x*x for x in c),1.,abs_tol=1e-14):
        raise AdmissionError("unregistered start")
    if witness["strict_newton_terminal"] or witness["represented_native_coefficient_vector_validated"]:
        raise AdmissionError("unexpected witness scope")
    empty=audit_family_split([],require_train_and_test=False)
    return dict(schema="W5_M32_PAIRED_READINESS_V1",protocol_sha256=file_sha256(protocol_path),
                split_check=empty, native_training_rows=0, real_evaluation_rows=0,
                numerical_reference_terminal=False,benchmark_ready=False,
                actual_solver_runs=0, timing_measured=False,speedup=None)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path);parser.add_argument("witness",type=Path)
    args=parser.parse_args()
    print(json.dumps(readiness(args.manifest,args.witness),sort_keys=True,indent=2))
