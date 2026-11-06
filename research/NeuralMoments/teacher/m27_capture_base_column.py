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


def plan(group_id, modules, source_checkout, *, column):
    require(type(column) is int and column in COLUMNS, 'UNREGISTERED_BASE_COLUMN')
    require(file_hash(LEDGER) == LEDGER_SHA, 'H8F_COEFFICIENT_LEDGER_DRIFT')
    entries = strict_json(LEDGER.read_bytes())['entries']
    # H8f stored eight points per column after point-000. The first four
    # are the centered stencil; the remaining four use smaller centered offsets.
    samples = []
    for index, offset in enumerate((-H, -H/2, H/2, H), start=8*column + 1):
        entry = entries[index]
        require(entry['index'] == index, 'H8F_POINT_INDEX_MISMATCH')
        samples.append((offset, entry['archived_native_c_sha256']))
    result = centered_plan(group_id, modules, source_checkout, columns={column: samples})
    result['schema'] = 'w2-m27-base-column-plan-v1'
    return result


def column_from_group(group_id):
    match = re.fullmatch(r'heart-base-c([3-9])-[1-9][0-9]*-[1-9][0-9]*', group_id)
    require(match is not None, 'UNREGISTERED_BASE_GROUP')
    return int(match.group(1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-request', type=Path, required=True)
    parser.add_argument('--module-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = campaign(args.output, args.module_manifest, args.prepared_request,
                      event_plan=partial(plan, column=column_from_group(args.output.name)))
    print(json.dumps(result, indent=2))
