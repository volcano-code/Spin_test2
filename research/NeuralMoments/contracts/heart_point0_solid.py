"""Replay the archived H8f point-000 regularized polyhedral difference from source."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

from .heart_point0_seams import FACE_MAP_SHA
from .real_h8e_archive_m8 import canonical_digest, digest_bytes, require
from .real_h8f_81_m9 import ARCHIVE_SHA256


W1 = "15db682052e9a97779c34ba02000095b78cf4b91"
W2 = "519906bc1447eddc74fb3ca0c2117b1d3c9eca09"
PERFORMANCE = "research/NeuralMoments/performance"
TEACHER = "research/NeuralMoments/teacher"
ROOT = Path(__file__).resolve().parents[3]


def reconcile(design_rows, cutter_rows, face_map, design_count, cutter_count):
    """Every parent is checked, and each saved triangle has exactly its W2 owner."""
    seen = set()
    for side, rows, count in ((0, design_rows, design_count), (1, cutter_rows, cutter_count)):
        require(len(rows) == count, "missing input parent")
        for parent, (index, state, ids) in enumerate(rows):
            require(index == parent, "wrong input parent order")
            require(state in ("removed", "retained", "cut", "partial"), "unknown parent state")
            require((state == "removed") == (not ids), "state conflicts with saved coverage")
            for i in ids:
                require(type(i) is int and 0 <= i < len(face_map), "invalid output face")
                require(i not in seen, "duplicate output face")
                require(face_map[i] == [side, parent], "W1/W2 disagree on parent")
                seen.add(i)
    require(len(seen) == len(face_map), "missing output face")
    return len(seen)


def export_sources(repository, commit, subdir, destination):
    # Export only tracked source bytes from a specific Git object.
    archive = destination / "source.tar"
    subprocess.run(["git", "-C", str(repository), "archive", "--format=tar", "-o",
                    str(archive), commit, subdir], check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")
    archive.unlink()
    return destination / subdir


def replay(archive, pilot_dir, out):
    archive, pilot_dir, out = (p.resolve() for p in (archive, pilot_dir, out))
    out.mkdir(exist_ok=False)

    def run(args, output):
        with (out / output).open("x") as stream:
            subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, stdout=stream, check=True)

    def read(name):
        return json.loads((out / name).read_bytes())

    ledger = out / "design.jsonl"
    face_map_path = out / "face-map.json"
    with TemporaryDirectory() as temp:
        perf = export_sources(ROOT, W1, PERFORMANCE, Path(temp))
        teacher = export_sources(ROOT, W2, TEACHER, Path(temp))
        run([perf / "m29_heart_cut_faces.py", archive, "--out", out / "design.json",
             "--all-design-ledger", ledger], "design.log")
        run([perf / "m30_heart_cutter_faces.py", archive, "--out", out / "cutter.json"], "cutter.log")
        run([teacher / "m22_point0_operand_replay.py", "--h8f", archive, "--pilot-dir",
             pilot_dir, "--face-map", face_map_path], "operands.json")
        run([teacher / "m22_point0_solid_boundaries.py", "--h8f", archive, "--checker-source",
             perf / "m26_exact_intersection.cpp"], "surfaces.json")
        run(["-m", "research.NeuralMoments.contracts.heart_point0_seams", archive, face_map_path],
            "seams.json")

    design, cutter, operands, surfaces, seams = [
        read(f"{name}.json") for name in ("design", "cutter", "operands", "surfaces", "seams")]
    require(all(x["archive_sha256"] == ARCHIVE_SHA256 for x in
                (design, cutter, operands, surfaces, seams)), "historical archive mismatch")
    require(design["historical_compressed_material_sha256"] == cutter["material_gzip_sha256"]
            == operands["historical_point_gzip_sha256"], "historical material packet mismatch")
    require(surfaces["material_sha256"] == operands["historical_material_sha256"],
            "physical surface material mismatch")
    require(design["failed"] == cutter["failed"] == 0 and all(row["passes"] is True and
            all(row[k] == "0" for k in ("missing", "extra", "overlap"))
            for row in cutter["records"]), "face coverage failure")
    design_data = ledger.read_bytes()
    require(digest_bytes(design_data) == design["design_parent_ledger_sha256"], "design ledger mismatch")
    face_map = read("face-map.json")
    require(canonical_digest(face_map) == operands["face_map_canonical_sha256"]
            == seams["face_map_sha256"] == FACE_MAP_SHA, "ownership map mismatch")
    shells = surfaces["surfaces"]
    require(len(face_map) == shells["saved_difference"]["faces"], "surface count mismatch")
    design_rows = [json.loads(line) for line in design_data.splitlines()]
    cutter_rows = [(r["parent"], r["status"], r["output_faces"]) for r in cutter["records"]]
    n = reconcile(design_rows, cutter_rows, face_map, shells["design"]["faces"], shells["cutter"]["faces"])
    return {
        "point": "H8f/point-000", "frame": "physical_model", "archive_sha256": ARCHIVE_SHA256,
        "material_sha256": operands["historical_material_sha256"],
        "w1_commit": W1, "w2_commit": W2,
        "historical_regularized_polyhedral_difference_verified": True,
        "design_parent_counts": dict(Counter(state for _, state, _ in design_rows)),
        "cutter_parent_counts": dict(Counter(state for _, state, _ in cutter_rows)),
        "cut_design_cells": design["cells_on_partially_cut_faces"], "cutter_cells": cutter["cells"],
        "w1_w2_output_faces_matched": n, "output_face_ownership": operands["saved_face_support_and_winding"],
        "oriented_seam_segments": seams["oriented_seam_segments"], "seam_loop_lengths": seams["seam_loop_lengths"],
        "output_volume_model_units_cubed": shells["saved_difference"]["volume_approx"],
        "m21_material_sha256": operands["m21_material_sha256"], "fresh_native_boolean_calls": 0,
        "m21_full_same_call_material_verified": False, "J_D": None, "training_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("m21_pilot_dir", type=Path)
    parser.add_argument("new_output_dir", type=Path)
    args = parser.parse_args()
    summary = replay(args.archive, args.m21_pilot_dir, args.new_output_dir)
    with (args.new_output_dir / "result.json").open("x") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
