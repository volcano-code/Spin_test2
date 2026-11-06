"""Use one Jacobian to propose one native Newton step and independently measure the trial."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from .m35_crossing_newton import (
    DSHA, FIXED_SHA, FULL_TEN_SHA, bounded_moments, digest, inertia_gate, keys,
    read_ten, ref_state, shift, topology,
)

MAP_ID = "M36-exact-Noodles-retained-material"
BACKEND = "libigl-CGAL-rational"
EVENT_FIELDS = ("type","restart","iteration","index","outcome","reason","step",
                "raw_residual","physical_relative","objective","mu")


def response(row, jacobian, derivative):
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.shape != (10, 10) or not np.isfinite(jacobian).all():
        raise ValueError("invalid proposal Jacobian")
    values = [*row["retained_ten_tip"], *row["variable_ten_tip"], *jacobian.ravel().tolist()]
    return " ".join(["MAP", row["result_sha256"], row["topology"], MAP_ID, BACKEND,
                     derivative, *map(str, values)])


def native_step(kernel, anchor, jacobian, derivative, candidate, preceding=()):
    """Allow one native trial per supplied trial callback; keep a one-iteration cap."""
    inputs = [" ".join(map(str, anchor["q"])), response(anchor, jacobian, derivative), *preceding, candidate]
    run = subprocess.run([str(kernel), "1", str(2+len(preceding))], input="\n".join(inputs)+"\n",
                         capture_output=True, text=True, check=True)
    rows = [line.split() for line in run.stdout.splitlines()]
    queries = [list(map(float, parts[1:])) for parts in rows if parts[0]=="QUERY"]
    if not queries or queries[0] != anchor["q"] or len(queries)>2+len(preceding) or rows[-1][0]!="TERMINAL":
        raise ValueError("native query identity or budget mismatch")
    events = [dict(zip(EVENT_FIELDS, parts[1:], strict=True)) for parts in rows if parts[0]=="EVENT"]
    return queries[1:], events, rows[-1][1:]


def evaluate_trial(a, q, folder):
    for root in (a.w2, a.w3):
        sys.path.insert(0, str(root.resolve()))
    from m35_general_crossing import general_evaluate
    from crossing_report import sample_flux

    design = a.design.read_bytes()
    if digest(design)!=DSHA or digest(a.fixed_ten.read_bytes())!=FIXED_SHA:
        raise ValueError("unbound geometry or fixed moments")
    original = read_ten(a.original_ten)
    if digest(a.original_ten.read_bytes())!=FULL_TEN_SHA:
        raise ValueError("unbound reference moments")
    rec = general_evaluate(design, np.asarray(q), folder, a.boolean, a.source, a.bounds)
    variable = bounded_moments(SimpleNamespace(w1=a.w1), folder, rec["result_sha256"])
    fixed = shift(read_ten(a.fixed_ten))
    total = [x+y for x,y in zip(variable,fixed,strict=True)]
    row = dict(q=q, result_sha256=rec["result_sha256"], operation=rec["operation"],
               inertia_gate=inertia_gate(total,original), retained_ten_tip=list(map(float,total)),
               variable_ten_tip=list(map(float,variable)), variable_mesh=[rec["vertices"],rec["faces"]],
               independent=ref_state(q,total))
    if row["inertia_gate"]["passes"]:
        src = json.loads((folder/"source.json").read_bytes())
        obj = json.loads((folder/"result.json").read_bytes())
        j = sample_flux(folder)
        if j["result_sha256"]!=rec["result_sha256"] or src["raw_q"]!=q:
            raise ValueError("unbound trial derivative")
        lines = design.decode().splitlines()
        design_keys = keys([r.split() for r in lines[1:1+int(lines[0].split()[0])]])
        row.update(topology=topology(obj,design_keys,keys(src["vertices"]),src["faces"]),J_tip=j["J_tip"])
    return row


def run(a):
    data = json.loads(a.anchor.read_bytes())
    anchor = data.get("evaluations", data.get("ledger"))[0]
    arm = json.loads(a.predictor.read_bytes())
    if arm["anchor_q"]!=anchor["q"] or not anchor["inertia_gate"]["passes"]:
        raise ValueError("predictor and exact feasible-inertia anchor differ")
    J = arm["J_tip"]
    preceding = []
    first = None
    if a.after_inertia_rejection is not None:
        previous = json.loads(a.after_inertia_rejection.read_bytes())
        first = previous["trial"]
        if (previous["anchor_geometry"] != anchor["result_sha256"] or previous["arm"] != a.arm
            or previous["predictor_sha256"] != digest(a.predictor.read_bytes())
            or "error" in first or first["inertia_gate"]["passes"]):
            raise ValueError("replay requires a matching exact first-trial inertia rejection")
        preceding.append("FAIL fixed_axis_inertia_below_90_percent")
    trial, _, _ = native_step(a.kernel, anchor, J, a.arm, "FAIL proposal_only", preceding)
    if len(trial)!=1+len(preceding) or (first is not None and first["q"] != trial[0]):
        raise ValueError("native kernel declined to issue a trial")
    a.out.mkdir(parents=True, exist_ok=False)
    answer = "FAIL exact_trial_unavailable"
    result = {"q":trial[-1]}
    try:
        result = evaluate_trial(a, trial[-1], a.out/"trial")
        if result["inertia_gate"]["passes"]:
            answer = response(result, result["J_tip"], "exact_moving_face_flux")
        else:
            answer = "FAIL fixed_axis_inertia_below_90_percent"
    except (ValueError,subprocess.CalledProcessError) as exc:
        result["error"]=str(exc)
    second, events, terminal = native_step(a.kernel, anchor, J, a.arm, answer, preceding)
    if second!=trial:
        raise ValueError("native proposal replay mismatch")
    out = dict(schema="m36-one-proposal-one-exact-trial-v1", arm=a.arm, anchor_q=anchor["q"], prior_rejected_trials=len(preceding),
               anchor_geometry=anchor["result_sha256"], anchor_mass_fraction=anchor["inertia_gate"]["mass_fraction"],
               anchor_equation_residual=anchor["independent"]["raw_residual"],
               predictor_sha256=digest(a.predictor.read_bytes()), kernel_sha256=digest(a.kernel.read_bytes()),
               true_anchor_moments=True, trial_true_callback=True, trial=result, events=events, terminal=terminal)
    (a.out/"report.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for opt in ("anchor","predictor","kernel","w1","w2","w3","boolean","source","design","bounds",
                "fixed-ten","original-ten","out"):
        p.add_argument("--"+opt, type=Path, required=True)
    p.add_argument("--arm",required=True)
    p.add_argument("--after-inertia-rejection",type=Path,
                   help="allow one native backtrack after replaying the exact previous trial rejection")
    run(p.parse_args())
