"""Isolate occupancy resolution with unit-density quadrature; no learning or geometry-query tuning."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .real_cgal_shadow import translation
from .refined_quadrature import cartesian, localized
from .refinement_ablation import digest


def score(input_path, paths):
    data=json.loads(input_path.read_bytes())
    scores=[]
    unit=np.ones((6,6,6))
    for path in (None, *paths):
        grid=json.loads(path.read_bytes()) if path else data["voxels"]
        side=len(grid["axis"])
        occ=np.array(grid["occupancy"])
        if (grid["indexing"]!="ij" or grid["cube_halfwidth"]!=1 or
                grid["cube_center_in_tip"]!=data["voxels"]["cube_center_in_tip"] or
                occ.shape!=(side,)*3 or int(occ.sum())!=grid["occupied"] or not np.isin(occ,[0,1]).all() or
                not np.array_equal(grid["axis"],np.linspace(-1,1,side)) or
                (grid.get("design_sha256") and grid["design_sha256"]!=data["voxels"]["design_sha256"])):
            raise ValueError("unpaired geometry grid")
        rho=torch.from_numpy(occ.astype(np.float32))[None,None]
        S,T=translation(grid["cube_center_in_tip"])
        for row in data["samples"]:
            q=T @ row["q"]
            for arm in ("fine_cool","chart"):
                if arm=="chart":
                    r,J,method=localized(rho,unit,q)
                else:
                    r,J=cartesian(rho,unit,q,24,.0075)
                    method="midpoint_logistic"
                r=S@r; J=S@J@T
                truth=np.array(row["variable_ten_tip"]); jref=np.array(row["J_tip"])
                scores.append(dict(case=row["name"],arm=arm,method=method,occupancy_side=side,
                    grid_file_sha256=digest(path or input_path),mass=float(r[0]),relative_mass=float(abs(r[0]/truth[0]-1)),
                    r=float(np.linalg.norm(r-truth)/np.linalg.norm(truth)), J=float(np.linalg.norm(J-jref)/np.linalg.norm(jref))))
    return dict(schema="M36_SUPPLEMENTARY_UNIT_DENSITY_OCCUPANCY_RESOLUTION",input_sha256=digest(input_path),
                scores=scores, scope="Same comparison arms after primary experiment. Geometry acquisition cost excluded.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path);p.add_argument("grids",nargs="+",type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    args.output.write_text(json.dumps(score(args.input,args.grids),indent=2,allow_nan=False)+"\n")
