"""Regenerate compact exact-Boolean receipts from the M35 research packet."""
import argparse
import json
from pathlib import Path

from m35_exact_crossing import COLUMNS, D_SHA, EXPONENTS, digest, dump
from m35_crop_control import compare


def verify(run):
    samples = ["base", *(f"c{k}_{sign}{p}" for k in COLUMNS for p in EXPONENTS for sign in ("p","m")), "seed0"]
    protocol = json.loads((run/"protocol.json").read_bytes())
    if (protocol["design_sha256"] != D_SHA or protocol["columns"] != list(COLUMNS)
            or protocol["exponents"] != list(EXPONENTS) or protocol["face"] != 81227):
        raise ValueError("unexpected probe registration")
    entries = {}
    for case in samples:
        folder = run/case
        receipt = json.loads((folder/"receipt.json").read_bytes())
        replay = json.loads((folder/"native-source-replay.json").read_bytes())
        if any(receipt[key+"_sha256"] != digest((folder/file).read_bytes())
               for key,file in (("cutter","cutter.txt"),("result","result.json"),("source","source.json"))):
            raise ValueError("unbound input or output")
        expected_q = list(protocol["q"])
        if case not in ("seed0","base"):
            column, sign_exp = case[1:].split("_")
            expected_q[int(column)] += (-1 if sign_exp[0] == "m" else 1) * 2.0**-int(sign_exp[1:])
        if case == "seed0":
            expected_q = json.loads((run/"seed0.json").read_bytes())
        if replay["raw_q"] != expected_q or replay["source_sha256"] != receipt["source_sha256"]:
            raise ValueError("unbound source or wrong finite-difference coordinate")
        if receipt["design_sha256"] != D_SHA or not replay["inside_positive"] or receipt["operation"] != "intersection":
            raise ValueError("unexpected operand or side")
        if receipt["design_parent_faces"] <= 0 or receipt["cutter_parent_faces"] <= 0:
            raise ValueError("not a crossing")
        if replay["cap_aabbs_disjoint_from_exact_design"] != replay["cap_faces"]:
            raise ValueError("cap proximity requires separate verification")
        if replay["stored_q_max_abs_error"] != 0:
            raise ValueError("unexpected normalized coefficient mismatch in reference suite")
        entries[case] = dict(receipt, native_replay_sha256=digest((folder/"native-source-replay.json").read_bytes()),
                             source_faces=replay["oriented_faces"], native_source_hash=replay["source_hash"],
                             source_velocity_scalars=replay["velocity_scalars"], cap_faces=replay["cap_faces"])
    return dict(base_tree="2f9cd722fa024b0144a73809933743831d5fe6b3", protocol=protocol,
                samples=entries, local_crop_control=compare(run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dump(args.output, verify(args.directory))
