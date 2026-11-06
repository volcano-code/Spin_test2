"""Independently feed the archived moment maps to a newly compiled stock Newton kernel."""
import argparse
import json
from pathlib import Path
import subprocess


def replay(kernel, report):
    limits=report["limits"]; inputs=[" ".join(map(str,report["start"]))]
    for row in report["evaluations"]:
        if "error" in row:
            msg=row["error"]
            kind = "CalledProcessError" if msg.startswith("Command '") else "ValueError"
            inputs.append("FAIL "+kind+"_"+"_".join(msg.split())[:120])
            continue
        values=[*row["retained_ten_tip"], *row["variable_ten_tip"],
                *(x for Jrow in row["J_tip"] for x in Jrow)]
        inputs.append(" ".join(["MAP",row["result_sha256"],row["topology"],report["map"],
                    "libigl-CGAL-rational-M34","moving-parent-flux-M35",*map(str,values)]))
    run=subprocess.run([str(kernel),str(limits["iterations"]),str(limits["evaluations"])],
        input="\n".join(inputs)+"\n",capture_output=True,text=True,check=True)
    lines=[line.split() for line in run.stdout.splitlines()]
    queries=[list(map(float,rec[1:])) for rec in lines if rec[0]=="QUERY"]
    if queries != [r["q"] for r in report["evaluations"]]:
        raise ValueError("native proposal sequence mismatch")
    events=[r[1:] for r in lines if r[0]=="EVENT"]
    order=("type","restart","iteration","index","outcome","reason","step","raw_residual","physical_relative","objective","mu")
    if events != [[r[k] for k in order] for r in report["events"]]:
        raise ValueError("native event stream mismatch")
    end=report["terminal"]; keys=("status","calls","starts","trace_entries","event_hash")
    if lines[-1] != ["TERMINAL",*(end[k] for k in keys)]:
        raise ValueError("native terminal or event digest mismatch")
    return {"queries":len(queries),"events":len(events),"terminal":end}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("kernel",type=Path);p.add_argument("reports",nargs="+",type=Path)
    args=p.parse_args()
    for path in args.reports:print(json.dumps(replay(args.kernel,json.loads(path.read_text()))))
