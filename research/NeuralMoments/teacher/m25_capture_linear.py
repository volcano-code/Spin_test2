"""Capture the two linear x and y coefficients on the original H8f abscissae."""
import argparse
import json
from pathlib import Path

from m24_capture_c0 import H, campaign

# Digests of the original H8f coefficient vectors, packets 009–012 and 017–020.
COLUMNS = {
    1: (
        (-H,   '710b87b25330055748280a9ca717748ed8bc964dc50daa5d5b25e0ac350e4ebc'),
        (-H/2, '93bb83aea96b6626f4da2749ffa3767f55583487e6877e3f45c6fe03b62bca75'),
        ( H/2, '1cc92b40b41cce4562b52708ae6bd0d042399bba19922a3d5445a9513d8e4025'),
        ( H,   'b0ce1437c2ec81a3462e2d3d4b0ac60a6be4c51217ac18b4512c0c241b855cdc'),
    ),
    2: (
        (-H,   '3050355b97a159bd3c93fa3e863d398d7984cf47437462e2221d5f7bb9b46f3e'),
        (-H/2, '5746b72e9582019be105cb7fae68db23bf35e54e9ab75b67e30bed3b981600ec'),
        ( H/2, '0f1b2df0f7262050cfc287757ea619110884e99057adc564b6986ea3939c790e'),
        ( H,   '47b339f0f6f2b2e3b12024781cba82bbc402b08f80711e02d64e8b0d4e6eb29b'),
    ),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-request', type=Path, required=True)
    parser.add_argument('--module-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(campaign(args.output, args.module_manifest, args.prepared_request, columns=COLUMNS),
                     indent=2))
