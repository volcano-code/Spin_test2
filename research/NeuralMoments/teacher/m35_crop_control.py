"""Compare cold and cached intersections on exactly the same five bounded cutters."""
import argparse
from collections import Counter
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import subprocess
import time

from m35_exact_crossing import D_SHA, digest, dump

CASES = ("base", "c0_p19", "c0_m19", "c0_p20", "c0_m20")
BOUNDS = ((7, -4, 36), (18, 4, 43))  # dyadic numerators over 32, in the model frame
BOX_FACES = ((6, 0, 2), (6, 4, 0), (5, 0, 4), (5, 1, 0), (5, 4, 6), (5, 6, 7),
             (3, 2, 0), (3, 0, 1), (3, 6, 2), (3, 7, 6), (3, 1, 5), (3, 5, 7))


def packet(vertices, faces):
    def token(x):
        value = Fraction(x)
        return f"{value.numerator}/{value.denominator}"
    return ("\n".join([f"{len(vertices)} {len(faces)}",
                      *(" ".join(map(token, v)) for v in vertices),
                      *(" ".join(map(str, f)) for f in faces)]) + "\n").encode()


def signature(mesh, born):
    """Exact cyclic order and original operand face, independent of output IDs."""
    if len(mesh["birth_faces"]) != len(mesh["faces"]):
        raise ValueError("incomplete provenance")
    tris = Counter()
    for f, parent in zip(mesh["faces"], mesh["birth_faces"]):
        p = tuple(tuple(mesh["vertices"][j]) for j in f)
        tris[(min(p, p[1:] + p[:1], p[2:] + p[:2]), born(parent))] += 1
    return tris


def validate(crop, direct, cached, source):
    for v in source["vertices"]:
        if not all(lo < Fraction(x)*32 < hi for x, lo, hi in zip(v, *BOUNDS)):
            raise ValueError("cutting mesh outside strict crop interior")
    parents = crop["birth_faces"]
    def original_birth(i):
        if i < len(parents):
            p = parents[i]
            if p >= 94150:
                raise ValueError("crop box face survives")
            return p
        return i - len(parents) + 94150
    if signature(cached, original_birth) != signature(direct, lambda x: x):
        raise ValueError("direct and cached oriented triangles or original parents differ")


def compare(directory):
    worker = directory/"boxcache"
    capture = json.loads((worker/"result.json").read_bytes())
    crop = json.loads((worker/"crop.json").read_bytes())
    rows = []
    for sample, recorded in zip(CASES, capture["results"], strict=True):
        if sample != recorded["sample"]:
            raise ValueError("wrong sample")
        folder = directory/sample
        direct = (folder/"result.json").read_bytes()
        cached = (worker/f"cached_{sample}.json").read_bytes()
        receipt = json.loads((folder/"receipt.json").read_bytes())
        if digest(direct) != receipt["result_sha256"] or digest(cached) != recorded["sha256"]:
            raise ValueError("mesh hashes differ")
        validate(crop, json.loads(direct), json.loads(cached), json.loads((folder/"source.json").read_bytes()))
        rows.append(dict(sample=sample, cold_seconds=receipt["seconds"], cached_seconds=recorded["seconds"],
                         direct_sha256=digest(direct), cached_sha256=digest(cached), identical_faces=recorded["faces"]))
    cold = sum(s["cold_seconds"] for s in rows)
    warm = sum(s["cached_seconds"] for s in rows)
    total = warm + capture["crop_seconds"]
    return dict(design_sha256=D_SHA, box_sha256=digest((worker/"box.txt").read_bytes()),
                crop_sha256=digest((worker/"crop.json").read_bytes()), crop_faces=len(crop["faces"]),
                setup_seconds=capture["crop_seconds"], five_direct_seconds=cold, five_cached_seconds=warm,
                setup_plus_five_cached_seconds=total, measured_ratio_including_setup=cold/total, samples=rows,
                scope="CGAL subprocess wall time, overlapping machine load, no PINO or end-to-end solver timing")


def run(design, csg, directory):
    if digest(design) != D_SHA:
        raise ValueError("wrong scientific operand")
    worker = directory/"boxcache"
    worker.mkdir(parents=True, exist_ok=True)
    box = packet(list(product(*(tuple(Fraction(x,32) for x in axis) for axis in zip(*BOUNDS)))), BOX_FACES)
    (worker/"box.txt").write_bytes(box)
    def cut(a, b, name):
        combined = worker/(name + ".in")
        result = worker/(name + ".json")
        combined.write_bytes(a + b)
        start = time.monotonic()
        subprocess.run([str(csg), str(combined), str(result), "intersection"], check=True)
        seconds = time.monotonic() - start
        combined.unlink()
        return json.loads(result.read_bytes()), seconds
    crop, seconds = cut(design, box, "crop")
    crop_input = packet(crop["vertices"], crop["faces"])
    (worker/"crop.txt").write_bytes(crop_input)
    rows = []
    for sample in CASES:
        mesh, elapsed = cut(crop_input, (directory/sample/"cutter.txt").read_bytes(), "cached_" + sample)
        rows.append(dict(sample=sample, seconds=elapsed, faces=len(mesh["faces"]),
                         sha256=digest((worker/f"cached_{sample}.json").read_bytes())))
    dump(worker/"result.json", dict(crop_faces=len(crop["faces"]), crop_seconds=seconds, results=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--csg", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        if not (args.design and args.csg):
            parser.error("--run requires --design and --csg")
        run(args.design.read_bytes(), args.csg, args.directory)
    dump(args.directory/"boxcache"/"comparison.json", compare(args.directory))
