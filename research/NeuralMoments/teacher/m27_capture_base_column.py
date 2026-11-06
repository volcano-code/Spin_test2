"""Capture the four preregistered H8f perturbations of one remaining Heart base coefficient."""
import argparse
from functools import partial
import json
import re
from pathlib import Path

from m16_native_acceptance import file_hash, require, strict_json
from m23_fresh_point0 import HERE
from m24_capture_c0 import H, campaign, plan as centered_plan

COLUMNS = range(3, 10)
LEDGER = HERE / 'M13_H8F_81_PROJECTIVE_HANDOFF_INDEX.json'
LEDGER_SHA = 'f2e5ba4efc3d9489e944112985ab2d9d8653a9f49f59ecc98aa06fc0943e1e04'


def plan(group_id, modules, source_checkout, *, column, half=False):
    require(type(column) is int and column in COLUMNS, 'UNREGISTERED_BASE_COLUMN')
    require(not half or column == 7, 'UNREGISTERED_HALF_RADIUS_COLUMN')
    require(file_hash(LEDGER) == LEDGER_SHA, 'H8F_COEFFICIENT_LEDGER_DRIFT')
    entries = strict_json(LEDGER.read_bytes())['entries']
    # The first four H8f points are centered at scales H and H/2.
    # H8f local indices 5 and 6 give the -H/4 and +H/4 points.
    radius = H/2 if half else H
    indices = (2, 5, 6, 3) if half else (1, 2, 3, 4)
    offsets = (-radius, -radius/2, radius/2, radius)
    samples = []
    for local_index, offset in zip(indices, offsets):
        index = 8*column + local_index
        entry = entries[index]
        require(entry['index'] == index, 'H8F_POINT_INDEX_MISMATCH')
        samples.append((offset, entry['archived_native_c_sha256']))
    result = centered_plan(group_id, modules, source_checkout, columns={column: samples})
    result['h'] = radius
    result['schema'] = 'w2-m28-base-c7-half-plan-v1' if half else 'w2-m27-base-column-plan-v1'
    return result


def stencil_from_group(group_id):
    match = re.fullmatch(r'heart-base-c([3-9])(-half)?-[1-9][0-9]*-[1-9][0-9]*', group_id)
    require(match is not None, 'UNREGISTERED_BASE_GROUP')
    column, half = int(match[1]), bool(match[2])
    require(not half or column == 7, 'UNREGISTERED_HALF_RADIUS_COLUMN')
    return column, half


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-request', type=Path, required=True)
    parser.add_argument('--module-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    column, half = stencil_from_group(args.output.name)
    result = campaign(args.output, args.module_manifest, args.prepared_request,
                      event_plan=partial(plan, column=column, half=half))
    print(json.dumps(result, indent=2))
