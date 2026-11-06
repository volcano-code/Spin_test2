"""Recompute a cropped W2 stencil sample's ten moments and verify its full-design contact guard."""
import argparse
import json
from pathlib import Path

from m34_noodles_partition import sha, surface_data, token
from m35_bounded_ten import integrate
from m35_noodles_crossing import DESIGN_SHA, audit, packet, strict_source_position


def verify(sample, design, evidence, ten_binary):
    receipt = json.loads((sample / 'receipt.json').read_bytes())
    crop_raw = sha((sample / 'crop-result.json').read_bytes())
    native = sha((sample / 'source.json').read_bytes())
    if (receipt['original_parent_remapped'] is not True or
            crop_raw != receipt['crop_raw_output_sha256'] or native != receipt['source_sha256']):
        raise ValueError('CROP_PARENT_OR_NATIVE_SOURCE_MISMATCH')

    # The exact M25 pass needs the original parent IDs, not the crop-local IDs.
    quick = audit(sample, design, ten_binary, None, evidence, full=False)
    result_sha, vertices, faces = surface_data(sample / 'result.json')
    reference_sha, d, fd = packet(design)
    cutter_sha, c, fc = packet(sample / 'cutter.txt')
    if reference_sha != DESIGN_SHA or quick['output_sha256'] != result_sha:
        raise ValueError('DESIGN_OR_RETAINED_REGION_MISMATCH')
    bounds = integrate(vertices, faces, bits=112)
    values = [token(row) for row in (evidence / (sample.name + '.ten')).read_text().splitlines()]
    if len(values) != 10 or not all(lo <= value <= hi for (lo, hi), value in zip(bounds, values)):
        raise ValueError('EXACT_TEN_OUTSIDE_OUTWARD_BOUNDS')

    position = strict_source_position(d, fd, c, fc)
    if not position['locally_transverse_polyhedra']:
        raise ValueError('NONSTRICT_CONTACT_GUARD')
    return dict(sample=sample.name, design_sha256=reference_sha, cutter_sha256=cutter_sha,
                result_sha256=result_sha, source_sha256=native, crop_raw_output_sha256=crop_raw,
                construction=quick['construction'], source_position=position,
                ten_bounds=[[str(lo), str(hi)] for lo, hi in bounds], ten=quick['signed_ten'],
                exact_ten_sha256=quick['exact_ten_sha256'], enclosure_bits=112,
                exact_ten_inside_all_bounds=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('sample', 'design', 'evidence', 'ten_binary'):
        parser.add_argument(arg, type=Path)
    print(json.dumps(verify(**vars(parser.parse_args())), indent=2))
