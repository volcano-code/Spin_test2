"""A conservative mass lower bound under the fixed M32 axis and 90% inertia constraint."""
import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

from .mass90_cylinder_reference import TAU

FULL_SOLID_SHA = "b0a29616092d002daa8e0c0b1e6a5e227ccba7843cd7f2ed6dcdeb32584ba8c8"
FULL_TEN_SHA = "5912e9f1dc44378919f5c00690983d284189edadc857ea3b354f09c4b86a1986"
INNER_SHA = "b68f810266a94e0c4e2489f2a1005ae0be1f7f321e9d9f092b047e374dd12133"


def enclose(token, places=16):
    """Round an exact pair of large (possibly hexadecimal) integers outwards."""
    n, d = (int(s, 0) for s in token.split("/"))
    if d <= 0:
        raise ValueError("expected positive denominator")
    scale = 10**places
    return Q(n*scale//d, scale), Q(-((-n*scale)//d), scale)


def removed_volume_upper(height, inertia_allowance, eta):
    """At unit density, 1 <= eta*r² + max(0,1-eta*r²) pointwise.

    Every removable point is in a slab of the given height. Over an infinite
    slab the last term integrates to pi*height/(2*eta). Use pi < 22/7.
    """
    if height <= 0 or inertia_allowance < 0 or eta <= 0:
        raise ValueError("invalid height, inertia allowance, or multiplier")
    return eta*inertia_allowance + Q(22, 7)*height/(2*eta)


def bound(outer_report, ten_path, inset_path):
    report = json.loads(outer_report.read_text())
    ten = ten_path.read_bytes()
    inset = inset_path.read_bytes()
    if (not report["embedding_passed"] or not report["exact_fixed_tip"] or
            report["source_sha256"] != FULL_SOLID_SHA or
            report["exact_ten_payload_sha256"] != FULL_TEN_SHA or
            hashlib.sha256(ten).hexdigest() != FULL_TEN_SHA or
            hashlib.sha256(inset).hexdigest() != INNER_SHA):
        raise ValueError("unverified M32 reference")
    tokens = ten.decode("ascii").splitlines()
    if len(tokens) != 10:
        raise ValueError("expected ten signed raw moments")
    volume_lo, _ = enclose(tokens[0])
    inertia_hi = enclose(tokens[4])[1] + enclose(tokens[5])[1]
    rows = inset.decode("ascii").splitlines()
    nvertices, nfaces = map(int, rows[0].split())
    if len(rows) != 1 + nvertices + nfaces:
        raise ValueError("incomplete rational inset operand")
    zs = [Q(row.split()[2]) for row in rows[1:1+nvertices]]
    height = max(zs) - min(zs)
    eta = Q(5)  # any positive multiplier is valid, in model-length^-2
    allowed = (1-TAU)*inertia_hi
    removed_hi = removed_volume_upper(height, allowed, eta)
    ratio_lo = max(Q(0), 1-removed_hi/volume_lo)
    return dict(schema="w0-m32-mass90-slab-dual-v1", unit_density_model_frame=True,
                source_A_sha256=FULL_SOLID_SHA, source_A_ten_sha256=FULL_TEN_SHA, Eprime_sha256=INNER_SHA,
                required_axial_fraction=str(TAU), slab_height=str(height), eta=str(eta),
                initial_mass_lower=str(volume_lo), allowed_removed_I_upper=str(allowed),
                removed_mass_upper=str(removed_hi), retained_mass_fraction_lower=str(ratio_lo),
                retained_mass_fraction_lower_floor_1e9=f"{(ratio_lo*10**9)//1}/1000000000",
                scope="all removable subsets of Eprime subject to the inertia budget; balance and connectivity relaxed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("outer_report", type=Path)
    p.add_argument("ten_file", type=Path)
    p.add_argument("inset_operand", type=Path)
    args = p.parse_args()
    print(json.dumps(bound(args.outer_report, args.ten_file, args.inset_operand), indent=2))
