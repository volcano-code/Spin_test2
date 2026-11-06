"""W0 exact tetrahedral moment replay on the material saved by a fresh W2 run."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

from .real_h8e_archive_m8 import (canonical_digest, check_closed_indexed_surface, digest_bytes,
                                   exact_rational_ten_moments, read_json, require)


POINT0_REQUEST = "7084fecb6bbd4d17bca842bfe68299f9711ecead09eb01593364424a3e0c4a8a"
POINT0_EXACT_TEN = "cf1d21a9d3b046f98a7ef92e275e616691fc9d4adc7fd3987dedee38a7fe495f"


def compare(vertices, faces, density, observed, reference):
    check_closed_indexed_surface(faces, len(vertices))
    values = exact_rational_ten_moments(vertices, faces, density)
    require(len(observed) == len(reference) == len(values) == 10, "ten components required")
    require(values == [Fraction(x) for x in reference], "saved exact moments differ")
    errors = [abs(exact - Fraction(value)) for exact, value in zip(values, observed)]
    require(max(errors) <= Fraction(1e-10), "fresh value exceeds component budget")
    return {"exact_ten_sha256": canonical_digest([str(x) for x in values]),
            "maximum_exact_error": float(max(errors)),
            "maximum_error_against_rounded_reference": max(abs(float(x)-y) for x,y in zip(values,observed))}


def audit(folder, expected_run_id):
    capture = read_json((folder / "capture.json").read_bytes())
    require(capture["run_id"] == expected_run_id, "unexpected run identity")
    records = []
    for name in ("spec.json", "material.json", "label.json"):
        raw = (folder / name).read_bytes()
        require(digest_bytes(raw) == capture["files"][name], "saved file changed: " + name)
        records.append(read_json(raw))
    spec, material, label = records
    payload = label["payload"]
    require(canonical_digest(payload) == label["payload_sha256"], "invalid sealed label")
    require(canonical_digest(spec["request"]) == capture["request_sha256"] == POINT0_REQUEST,
            "unexpected Heart input")
    require(spec["geometry"]["capture_run_id"] == payload["geometry"]["capture_run_id"] == expected_run_id,
            "spec/label run identity mismatch")
    require(spec["reference_absolute_tolerances"] == [1e-10] * 10, "unexpected error budget")
    proof = payload["exact_proof"]
    require(proof["native_cherchi"] is True and proof["native_stage_call_counts"]["boolean"] == 1
            and capture["native_stage_call_counts"] == proof["native_stage_call_counts"],
            "single native call evidence missing")
    material_sha = canonical_digest(material)
    require(material_sha == capture["same_call_material_sha256"] ==
            proof["same_call_attachment"]["serialized_material_sha256"], "material not bound to label")
    require(payload["density"] == spec["request"]["material_density"] == 1.
            and payload["r_D"] == material["variable_raw_moments"], "material values or density mismatch")
    result = compare(material["variable_rational_vertices"], material["variable_faces"], payload["density"],
                     payload["r_D"], proof["independent_ten_moment_audit"]["reference_raw_moments_exact"])
    return dict(run_id=expected_run_id, material_sha256=material_sha, **result,
                same_exact_ten_as_certified_historical_point=(result["exact_ten_sha256"] == POINT0_EXACT_TEN),
                native_calls_in_w0_replay=0)


if __name__ == "__main__":
    sys.set_int_max_str_digits(100000)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("expected_run_id")
    args = parser.parse_args()
    print(json.dumps(audit(args.capture_directory,args.expected_run_id), indent=2))
