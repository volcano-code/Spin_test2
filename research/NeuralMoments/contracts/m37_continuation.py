"""Continue the three M36 Noodles trajectories in the original native solver."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .m35_crossing_newton import DSHA, FIXED_SHA, FULL_TEN_SHA, digest
from .m36_one_step import EVENT_FIELDS, evaluate_trial, response

M36 = Path(__file__).with_name("m36")
MANIFEST_SHA = "d3b507b41f00bb47541cdda28f3b6ea4f86a4156658cee5b011a9ace9f877933"
FILES = ("source.json", "receipt.json", "result.json", "moments.json", "cutter.txt")
LIMITS = {"iterations": 2, "evaluations": 5}


def load(path):
    return json.loads(path.read_bytes())


def prefix(arm):
    native = load(M36 / "large-native.json")
    anchor = native["evaluations"][0]
    rows = [("anchor", anchor)]
    if arm == "exact":
        return rows + [("exact-first", native["evaluations"][1])], None
    path = M36 / f"seed131_{arm}.json"
    predictor = load(path)
    if predictor["anchor_q"] != anchor["q"] or predictor["arm"] != arm:
        raise ValueError("M36 predictor does not belong to this anchor and arm")
    label = "pino" if arm == "learned" else "unit"
    for run, step in (("trial", "first"), ("half", "half")):
        packet = load(M36 / f"{run}-{arm}.json")
        if (packet["anchor_geometry"] != anchor["result_sha256"] or
                packet["predictor_sha256"] != digest(path.read_bytes()) or
                packet["kernel_sha256"] != native["kernel_sha256"] or packet["arm"] != arm):
            raise ValueError("M36 proposal receipt mismatch")
        rows.append((f"{label}-{step}", packet["trial"]))
    return rows, predictor


def restore(raw, manifest, label, row, folder, q):
    """Verify the frozen raw packet and bind its geometry to the native query."""
    if q != row["q"]:
        raise ValueError("native prefix query changed")
    folder.mkdir()
    for name in FILES:
        key = f"Noodles/proposals/{label}/{name}"
        data = (raw / key).read_bytes()
        if digest(data) != manifest[key]:
            raise ValueError("M36 raw digest mismatch: " + key)
        (folder / name).write_bytes(data)
    receipt = load(folder / "receipt.json")
    if (receipt["result_sha256"] != row["result_sha256"] or
            receipt["result_sha256"] != manifest[f"Noodles/proposals/{label}/result.json"] or
            load(folder / "source.json")["raw_q"] != q):
        raise ValueError("prefix mesh is not bound to the query")


def run(args):
    rows, predictor = prefix(args.arm)
    q0 = rows[0][1]["q"]
    native = load(M36 / "large-native.json")
    hashes = {k: digest(getattr(args, k).read_bytes())
              for k in ("design", "fixed_ten", "original_ten", "bounds", "source", "boolean", "kernel")}
    receipt = load(args.raw / "Noodles/proposals/anchor/receipt.json")
    anchor_source = load(args.raw / "Noodles/proposals/anchor/source.json")
    expected = {"design": DSHA, "fixed_ten": FIXED_SHA, "original_ten": FULL_TEN_SHA,
                "kernel": native["kernel_sha256"], "source": receipt["source_executable"],
                "boolean": receipt["csg_executable"], "bounds": anchor_source["bound_reference_sha256"]}
    if hashes != expected or digest((args.raw / "sha256.json").read_bytes()) != MANIFEST_SHA:
        raise ValueError("M36 raw manifest, binary or input mismatch")
    manifest = load(args.raw / "sha256.json")
    args.out.mkdir(parents=True, exist_ok=False)
    trace = dict(schema="M37-native-continuation-v1", arm=args.arm, start=q0,
                 kernel_sha256=hashes["kernel"], limits=LIMITS, evaluations=[], events=[],
                 terminal=None, protocol={"hashes": hashes, "inertia_floor": 0.90,
                 "diagnostic_target": {"raw_residual": 0.1, "physical_relative": 0.003},
                 "after_anchor": "exact_moving_face_flux", "m36_manifest_sha256": MANIFEST_SHA})
    started = time.monotonic()

    def checkpoint(name):
        trace["elapsed_seconds"] = time.monotonic() - started
        tmp = args.out / "checkpoint.tmp"
        tmp.write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.out / name)

    with (args.out / "kernel.stderr").open("w") as err, subprocess.Popen(
            [str(args.kernel), str(LIMITS["iterations"]), str(LIMITS["evaluations"])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1) as proc:
        def send(text):
            proc.stdin.write(text + "\n")
            proc.stdin.flush()

        send(" ".join(map(str, q0)))
        for line in proc.stdout:
            tag, *fields = line.split()
            if tag == "EVENT":
                trace["events"].append(dict(zip(EVENT_FIELDS, fields, strict=True)))
                checkpoint("partial.json")
                continue
            if tag == "TERMINAL":
                trace["terminal"] = dict(zip(("status", "calls", "starts", "trace_entries", "event_hash"),
                                             fields, strict=True))
                break
            if tag != "QUERY":
                raise ValueError("unexpected native output")
            idx = len(trace["evaluations"])
            q = list(map(float, fields))
            if len(q) != 10 or idx >= LIMITS["evaluations"]:
                raise ValueError("invalid query or excess evaluations")
            folder = args.out / f"call-{idx+1:03d}"
            cached = idx < len(rows)
            row = {"index": idx+1, "q": q, "boolean_execution":
                   "replayed_hash_bound" if cached else "fresh_requested"}
            trace["evaluations"].append(row)
            checkpoint("partial.json")
            t0 = time.monotonic()
            if cached:
                label, prior = rows[idx]
                restore(args.raw, manifest, label, prior, folder, q)
                row.update({k: v for k, v in prior.items() if k not in ("index", "boolean_execution")})
                row["m36_prefix_packet"] = label
            else:
                try:
                    row.update(evaluate_trial(args, q, folder))
                except (ValueError, subprocess.CalledProcessError) as exc:
                    row["error"] = str(exc)
            key = "prefix_verification" if cached else "exact_callback"
            row["phase_seconds"] = {key: time.monotonic() - t0}
            row["elapsed_seconds"] = time.monotonic() - started
            gate = row.get("inertia_gate")
            if "error" in row:
                answer = "FAIL exact_callback_error"
            elif not gate["passes"]:
                answer = "FAIL fixed_axis_inertia_below_90_percent"
            else:
                use_model = idx == 0 and predictor is not None
                jac = predictor["J_tip"] if use_model else row["J_tip"]
                derivative = args.arm if use_model else "exact_moving_face_flux"
                answer = response(row, jac, derivative)
            checkpoint("partial.json")
            send(answer)
        proc.stdin.close()
        if proc.wait() or trace["terminal"] is None or int(trace["terminal"]["calls"]) != len(trace["evaluations"]):
            raise RuntimeError("incomplete native terminal receipt")
    checkpoint("report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("exact", "learned", "unit"), required=True)
    for option in ("raw", "out", "kernel", "boolean", "source", "design", "bounds", "w1",
                   "w2", "w3", "fixed-ten", "original-ten"):
        parser.add_argument("--" + option, type=Path, required=True)
    run(parser.parse_args())
