"""Capture two one-sided triplets around the historical Heart c[7] event."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

from m23_fresh_point0 import NATIVE_C
from m24_capture_c0 import base_plan, campaign
from native_cherchi_teacher import digest

# Rounded binary64 coordinates preregistered by W3. The middle abscissa in
# each triplet is the nearest float to the exact midpoint of its two endpoints.
ABSCISSAE = (
    ('left','coarse','0x1.24ac5f4eb5eaap-19'),
    ('left','mid',   '0x1.2ab68f7e53921p-19'),
    ('left','near',  '0x1.30c0bfadf1398p-19'),
    ('right','near', '0x1.336ff18a374b0p-19'),
    ('right','mid',  '0x1.397a21b9d4f27p-19'),
    ('right','coarse','0x1.3f8451e97299ep-19'),
)


def plan(group_id, modules, source_checkout):
    stencil = base_plan(group_id, modules, source_checkout)
    samples = []
    for index,(side, location, encoded) in enumerate(ABSCISSAE, 1):
        c = list(NATIVE_C)
        c[7] = float.fromhex(encoded)
        samples.append(dict(index=index, run_id=f'{group_id}-{index:03d}', column=7,
                            side=side, location=location, native_c7_hex=encoded, native_c=c,
                            native_c_sha256=digest(c),
                            actual_offset_exact=str(Fraction(c[7])-Fraction(NATIVE_C[7]))))
    return dict(stencil,schema='w2-m26-c7-plan-v1',column=7,samples=samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-request',type=Path,required=True)
    parser.add_argument('--module-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(campaign(args.output,args.module_manifest,args.prepared_request,event_plan=plan),indent=2))
