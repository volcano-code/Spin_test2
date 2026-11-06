"""Run the nine-dimensional projective tangent FD represented by ten ambient columns, on the bounded local crop."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from m35_exact_crossing import D_SHA, digest, dump, make_source
from m35_crop_control import BOUNDS, packet, signature

CROP_SHA = "6021c41f5b9cbb481a47ab2ffbaddf46447ee6b4563ff3b91a77bcc866cc6bfe"


def remap(parents, faces):
    result = []
    for idx in faces:
        if idx < 0 or idx >= len(parents) + 320:
            raise ValueError("invalid local parent")
        if idx < len(parents):
            if parents[idx] >= 94150:
                raise ValueError("crop box survives")
            result.append(parents[idx])
        else:
            result.append(94150 + idx - len(parents))
    return result


def run(parent, out, source, csg):
    protocol = json.loads((parent/"protocol.json").read_bytes())
    base = np.array(protocol["q"])
    native_crop = (parent/"boxcache/crop.json").read_bytes()
    if digest(native_crop) != CROP_SHA:
        raise ValueError("incorrect full-design crop certificate")
    crop = json.loads(native_crop)
    crop_input = (parent/"boxcache/crop.txt").read_bytes()
    if packet(crop["vertices"], crop["faces"]) != crop_input:
        raise ValueError("crop transport disagrees with exact Boolean")
    out.mkdir(exist_ok=True, parents=True)
    csg_hash = digest(csg.read_bytes())
    source_hash = digest(source.read_bytes())
    dump(out/"protocol.json", dict(parent_protocol_sha=digest((parent/"protocol.json").read_bytes()),
         columns=list(range(10)), exponents=[19,20], q=base.tolist(), design_sha256=D_SHA,
         crop_sha256=digest(crop_input), executor_sha256=csg_hash, source_sha256=source_hash,
         purpose="complete local ellipsoid FD against exact boxed Dprime; original direct 13-point reference unchanged"))
    cases = {"base": base}
    for k in range(10):
        for exponent in (19,20):
            for sign, label in ((1,"p"),(-1,"m")):
                q = base.copy()
                q[k] += sign*2.0**-exponent
                cases[f"c{k}_{label}{exponent}"] = q
    stats = {}
    for name, q in cases.items():
        folder = out/name
        folder.mkdir(exist_ok=True)
        cutter = make_source(q, source, folder)
        geometry = json.loads((folder/"source.json").read_bytes())
        if any(not lo < Fraction(x)*32 < hi for row in geometry["vertices"] for lo,x,hi in zip(BOUNDS[0],row,BOUNDS[1])):
            raise ValueError("cutting mesh outside strict crop interior")
        raw = folder/"crop-result.json"
        existing = parent/"boxcache"/("cached_" + name + ".json")
        if existing.exists():
            shutil.copyfile(existing, raw)
            elapsed = None  # prior timing in the separate five-case benchmark
        else:
            operand = folder/"input.txt"
            operand.write_bytes(crop_input+cutter)
            start = time.monotonic()
            subprocess.run([str(csg), str(operand), str(raw), "intersection"], check=True)
            elapsed = time.monotonic() - start
            operand.unlink()
        obj = json.loads(raw.read_bytes())
        if not obj["succeeded"] or obj["operation"] != "intersection" or not obj["faces"]:
            raise ValueError("wrong Boolean")
        ids = remap(crop["birth_faces"], obj["birth_faces"])
        obj["birth_faces"] = ids
        direct = parent/name/"result.json"
        if direct.exists() and signature(json.loads(direct.read_bytes()),lambda x:x) != signature(obj,lambda x:x):
            raise ValueError("direct oriented mesh or parent differs")
        (folder/"result.json").write_text(json.dumps(obj,separators=(",",":"))+"\n")
        stats[name] = dict(operation="intersection", design_sha256=D_SHA, cutter_sha256=digest(cutter),
             source_sha256=digest((folder/"source.json").read_bytes()), source_executable=source_hash,
             csg_executable=csg_hash, result_sha256=digest((folder/"result.json").read_bytes()), seconds=elapsed,
             vertices=len(obj["vertices"]), faces=len(ids), design_parent_faces=sum(p<94150 for p in ids),
             cutter_parent_faces=sum(p>=94150 for p in ids), crop_operand_sha256=digest(crop_input),
             crop_raw_output_sha256=digest(raw.read_bytes()), original_parent_remapped=True,
             direct_oriented_face_and_parent_match=direct.exists())
        dump(folder/"receipt.json", stats[name])
    dump(out/"summary.json", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("parent", "out", "source", "csg"):
        parser.add_argument("--"+k, required=True, type=Path)
    a = parser.parse_args()
    run(a.parent,a.out,a.source,a.csg)
