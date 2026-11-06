"""Cross-check saved M37 meshes by integrating the ten moments along three axes."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from .m35_crossing_newton import FIXED_SHA, FULL_TEN_SHA, digest, read_ten, ref_state, shift

# x^a y^b z^c in the registered W1 raw-moment order.
POWERS = np.array([(0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,0,0),
                   (0,2,0), (0,0,2), (0,1,1), (1,0,1), (1,1,0)])


def flux_ten(vertices, faces):
    """Three divergence-theorem fluxes with a degree-three triangle quadrature."""
    tri = vertices[faces]
    normals = np.cross(tri[:,1] - tri[:,0], tri[:,2] - tri[:,0]) / 2
    center = tri.mean(axis=1)
    # The antiderivative of each quadratic monomial has degree at most three.
    points = [(-27/48, center), *((25/48, 0.6*center + 0.4*tri[:,i]) for i in range(3))]
    values = np.zeros((3,10))
    for weight, point in points:
        x, y, z = point.T
        basis = np.column_stack((np.ones(len(x)), x, y, z, x*x, y*y, z*z, y*z, x*z, x*y))
        for axis in range(3):
            flux = np.einsum("i,ij->j", point[:,axis] * normals[:,axis], basis)
            values[axis] += weight * flux / (POWERS[:,axis] + 1)
    return values


def check(runs, fixed_path, original_path):
    if digest(fixed_path.read_bytes()) != FIXED_SHA or digest(original_path.read_bytes()) != FULL_TEN_SHA:
        raise ValueError("unregistered Noodles references")
    fixed = np.array(read_ten(fixed_path), float)
    original = np.array(read_ten(original_path), float)
    cache = {}
    checks = {}
    for directory in runs:
        data = json.loads((directory / "report.json").read_bytes())
        results = []
        for row in data["evaluations"]:
            folder = directory / f"call-{row['index']:03d}"
            if "result_sha256" not in row:
                results.append({"index": row["index"], "no_recorded_mesh": True})
                continue
            raw = (folder / "result.json").read_bytes()
            mesh_id = digest(raw)
            packet = json.loads((folder / "moments.json").read_bytes())
            if mesh_id != row["result_sha256"] or packet["source_sha256"] != mesh_id:
                raise ValueError("raw geometry or moment receipt changed")
            if mesh_id not in cache:
                mesh = json.loads(raw)
                vertices = np.array([[float(Fraction(v)) for v in xyz] for xyz in mesh["vertices"]])
                cache[mesh_id] = flux_ten(vertices, np.array(mesh["faces"]))
            flux = cache[mesh_id]
            bounds = np.array([[float(Fraction(v)) for v in pair] for pair in packet["ten_bounds"]])
            deviation = np.maximum(np.maximum(bounds[:,0]-flux, flux-bounds[:,1]), 0)
            total = fixed + flux.mean(axis=0)
            retained_inertia = (total[4]+total[5]) / (original[4]+original[5])
            mass_fraction = total[0] / original[0]
            gate = row["inertia_gate"]
            if (deviation.max() > 1e-10 or abs(gate["lower"]-retained_inertia) > 1e-10 or
                    abs(gate["mass_fraction"]-mass_fraction) > 1e-10 or
                    bool(retained_inertia >= 0.9) != gate["passes"]):
                raise ValueError("independent flux is inconsistent with recorded moments or gate")
            rec = dict(index=row["index"], mesh_sha256=mesh_id,
                       max_axis_moment_disagreement=float(deviation.max()),
                       reconstructed_mass_fraction=mass_fraction,
                       reconstructed_axis_inertia=retained_inertia)
            if "independent" in row:
                state = ref_state(row["q"], shift(total))
                for key in ("raw_residual", "physical_relative"):
                    if abs(state[key] - row["independent"][key]) > 1e-9:
                        raise ValueError("reconstructed physical state differs")
                    rec[key] = state[key]
            results.append(rec)
        checks[data["arm"]] = results
    return dict(schema="m37-independent-binary64-surface-flux-v1", unique_meshes=len(cache), arms=checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixed-ten", type=Path, required=True)
    p.add_argument("--original-ten", type=Path, required=True)
    p.add_argument("runs", type=Path, nargs="+")
    args = p.parse_args()
    print(json.dumps(check(args.runs, args.fixed_ten, args.original_ten), allow_nan=False, indent=2))
