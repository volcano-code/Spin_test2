"""Audit exact query receipts and native acceptance for a matched trajectory experiment."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def verify_callback(folder, row, protocol):
    """Return the verified receipt and whether a moment packet binds to its material."""
    receipt_path = folder / "receipt.json"
    if not receipt_path.exists():
        if row.get("result_sha256"):
            raise ValueError("missing receipt for recorded geometry")
        return None, False
    receipt = read(receipt_path)
    for key, filename in (("result_sha256", "result.json"), ("source_sha256", "source.json"),
                          ("cutter_sha256", "cutter.txt")):
        if sha(folder / filename) != receipt[key]:
            raise ValueError("hash mismatch for " + filename)
    if row.get("result_sha256", receipt["result_sha256"]) != receipt["result_sha256"]:
        raise ValueError("trial geometry differs from receipt")
    source = read(folder / "source.json")
    if source["raw_q"] != row["q"]:
        raise ValueError("exact source used a different query")
    hashes = protocol["hashes"]
    for key, name in (("design_sha256", "design"), ("csg_executable", "boolean"), ("source_executable", "source")):
        if receipt[key] != hashes[name]:
            raise ValueError("receipt does not match protocol: " + key)
    if source["bound_reference_sha256"] != hashes["bounds"]:
        raise ValueError("bound reference mismatch")
    moments = folder / "moments.json"
    if not moments.exists():
        return receipt, False
    packet = read(moments)
    if packet["source_sha256"] != receipt["result_sha256"] or len(packet["ten_bounds"]) != 10:
        raise ValueError("moment packet is not bound to the trial geometry")
    return receipt, True


def summarize(directory, target, physical_target):
    if not all(finite(t) and t >= 0 for t in (target, physical_target)):
        raise ValueError("expected a finite nonnegative residual threshold")
    path = directory / "report.json"
    if not path.exists():
        path = directory / "partial.json"
    snapshot = path.read_bytes()
    data = json.loads(snapshot)
    rows = data.get("evaluations", data.get("ledger"))
    events = [event for event in data["events"] if event["type"] == "evaluation"]
    by_index = {int(event["index"]): event for event in events}
    if len(by_index) != len(events) or any(i < 1 or i > len(rows) for i in by_index):
        raise ValueError("duplicate or unbound evaluation event")
    terminal = data.get("terminal")
    if terminal and int(terminal["calls"]) != len(rows):
        raise ValueError("terminal and ledger disagree on callback count")
    if terminal and by_index.keys() != set(range(1, len(rows)+1)):
        raise ValueError("final trace does not account for every evaluation")
    metrics = Counter()
    phases = Counter()
    costs_present = 0
    measured_csg = Counter()
    measured_callbacks = Counter()
    accepted = []
    hit = None
    previous_elapsed = 0
    protocol = data.get("protocol") or {}
    for index, row in enumerate(rows, 1):
        if row["index"] != index:
            raise ValueError("out-of-order query ledger")
        execution = row.get("boolean_execution")
        metrics["queries"] += 1
        if execution == "fresh_requested":
            metrics["fresh_requested"] += 1
        elif execution == "replayed_hash_bound":
            metrics["replayed"] += 1
        else:
            metrics["other_or_unknown_execution"] += 1
        event = by_index.get(index)
        if event is None and not row.get("phase_seconds") and not row.get("result_sha256"):
            metrics["pending_native_decisions"] += 1
            metrics["unfinished_callbacks"] += 1
            continue
        receipt, moments = verify_callback(directory / f"call-{index:03d}", row, protocol)
        geometry = receipt is not None
        metrics["verified_meshes"] += geometry
        metrics["verified_moment_packets"] += moments
        metrics["fresh_completed_meshes"] += geometry and execution == "fresh_requested"
        gate = row.get("inertia_gate")
        if gate and not gate["passes"]:
            metrics["inertia_rejections"] += 1
        costs = row.get("phase_seconds")
        if costs is not None:
            if any(not finite(v) or v < 0 for v in costs.values()):
                raise ValueError("invalid phase seconds")
            phases.update(costs)
            costs_present += 1
        if receipt is not None and "seconds" in receipt:
            seconds = receipt["seconds"]
            if not finite(seconds) or seconds < 0:
                raise ValueError("invalid receipt CSG seconds")
            if execution == "fresh_requested" and costs and "exact_callback" in costs:
                remainder = costs["exact_callback"] - seconds
                if remainder < -1e-5:
                    raise ValueError("CSG time exceeds its enclosing fresh callback")
                measured_csg["fresh_csg"] += seconds
                measured_csg["fresh_non_csg"] += max(0.0, remainder)
                measured_callbacks["fresh"] += 1
            if execution == "replayed_hash_bound":
                measured_csg["historical_csg"] += seconds
                measured_callbacks["replayed"] += 1
        elapsed = row.get("elapsed_seconds")
        if elapsed is not None:
            if not finite(elapsed) or elapsed < previous_elapsed:
                raise ValueError("nonmonotonic cumulative query timing")
            previous_elapsed = elapsed
        if event is None:
            metrics["pending_native_decisions"] += 1
            continue
        if event["outcome"] != "accepted":
            metrics["native_rejected_or_failed"] += 1
            continue
        if not moments or not (gate and gate["passes"]):
            raise ValueError("accepted state lacks exact moments or passing inertia")
        for name in ("raw_residual", "physical_relative"):
            value = row["independent"][name]
            if not finite(value) or value < 0 or not math.isclose(float(event[name]), value, rel_tol=1e-8, abs_tol=1e-9):
                raise ValueError("independent and native residuals differ: " + name)
        physical = row["independent"]["physical_relative"]
        if not finite(gate["lower"]) or gate["lower"] < 0.9 or not finite(gate["mass_fraction"]):
            raise ValueError("accepted state fails the registered inertia floor or mass check")
        raw = row["independent"]["raw_residual"]
        # The initial state (and any later restart) is not an optimized step.
        initial = event["reason"] == "initial_evaluation_succeeded"
        metrics["accepted_steps"] += not initial
        state = dict(index=index, raw_residual=raw, inertia_lower=gate["lower"],
                     mass_fraction=gate["mass_fraction"], physical_relative=physical, initial=initial)
        accepted.append(state)
        if hit is None and not initial and raw <= target and physical <= physical_target:
            hit = dict(**state, queries=index, fresh_requested=metrics["fresh_requested"],
                       elapsed_seconds=elapsed, replayed=metrics["replayed"],
                       unknown_execution=metrics["other_or_unknown_execution"])
    elapsed = data.get("elapsed_seconds")
    if elapsed is not None and (not finite(elapsed) or elapsed < previous_elapsed):
        raise ValueError("invalid run wall time")
    # An interrupted job might have created a call directory after its last atomic ledger flush.
    logged = {f"call-{i:03d}" for i in range(1, len(rows) + 1)}
    pending = sorted(p.name for p in directory.glob("call-*") if p.is_dir() and p.name not in logged)
    return dict(arm=data.get("arm"), trace_file=path.name, trace_sha256=hashlib.sha256(snapshot).hexdigest(), terminal=terminal,
                physical_target=physical_target, protocol=protocol, limits=data.get("limits"), kernel_sha256=data.get("kernel_sha256"),
                start=data.get("start"), anchor_sha256=rows[0].get("result_sha256") if rows else None,
                elapsed_seconds=elapsed, phase_seconds=dict(phases), rows_with_phase_seconds=costs_present,
                csg_breakdown_seconds=dict(measured_csg), csg_breakdown_rows=dict(measured_callbacks),
                counts=dict(metrics), accepted=accepted, residual_target=target, first_hit=hit,
                unledgered_call_directories=pending)


def compare(directories, target, physical_target):
    arms = [summarize(path, target, physical_target) for path in directories]
    if len(arms) < 2 or not all(a["arm"] for a in arms) or len({a["arm"] for a in arms}) != len(arms):
        raise ValueError("need at least two uniquely named arms")
    differing = [k for k in ("protocol", "start", "kernel_sha256", "limits", "anchor_sha256")
                 if not arms[0][k] or any(a[k] != arms[0][k] for a in arms[1:])]
    all_reach = all(a["first_hit"] is not None for a in arms)
    time_comparable = all_reach and not differing and all(
        a["first_hit"]["replayed"] == 0 and a["first_hit"]["unknown_execution"] == 0
        and a["first_hit"]["elapsed_seconds"] is not None for a in arms
    ) and len({a["first_hit"]["fresh_requested"] for a in arms}) == 1
    return dict(schema="M37_MATCHED_TRAJECTORY_AUDIT_V1", residual_target=target, physical_target=physical_target,
                unmatched_fields=differing, same_target_all_reached=all_reach,
                equal_accuracy_query_counts={a["arm"]: a["first_hit"]["queries"] for a in arms}
                if all_reach and not differing else None,
                equal_accuracy_seconds={a["arm"]: a["first_hit"]["elapsed_seconds"] for a in arms}
                if time_comparable else None,
                arms=arms)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=float, required=True)
    p.add_argument("--physical-target", type=float, required=True)
    p.add_argument("runs", type=Path, nargs="+")
    args = p.parse_args()
    print(json.dumps(compare(args.runs, args.target, args.physical_target), indent=2, allow_nan=False))
