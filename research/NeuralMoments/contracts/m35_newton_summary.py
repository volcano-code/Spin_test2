"""Evaluate the original fixed-axis 90% inertia gate for every measured native callback."""
import json
from pathlib import Path
import argparse

from .m32_mass_bound import FULL_TEN_SHA
from .m35_crossing_newton import digest, read_ten


def summarize(report, full_ten):
    if digest(full_ten.read_bytes()) != FULL_TEN_SHA:
        raise ValueError("unexpected original solid moment file")
    outer = read_ten(full_ten)
    mass, axial = float(outer[0]), float(outer[4]+outer[5])
    by_id = {int(r["index"]): r for r in report["events"] if r["type"]=="evaluation"}
    states = []
    for row in report["evaluations"]:
        e = by_id[row["index"]]
        state = dict(index=row["index"], outcome=e["outcome"], reason=e["reason"], step=e["step"])
        if "retained_ten_tip" in row:
            r = row["retained_ten_tip"]
            # x and y axes are unchanged by the model-to-tip z translation.
            fraction = (r[4]+r[5])/axial
            state.update(mass_fraction=r[0]/mass, axial_inertia_fraction=fraction,
                         passes_axial_inertia_90=fraction>=.90, raw_residual=float(e["raw_residual"]),
                         physical_relative=float(e["physical_relative"]),
                         result_sha256=row["result_sha256"])
        states.append(state)
    return dict(map=report["map"], fixed_axis="registered z axis through the unique tip",
                thresholds=dict(minimum_inertia_ratio=.9), terminal=report["terminal"],
                states=states, native_cherchi_calls=report["native_cherchi_calls"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("report",type=Path);p.add_argument("full_ten",type=Path)
    a=p.parse_args()
    print(json.dumps(summarize(json.loads(a.report.read_text()),a.full_ten),indent=2))
