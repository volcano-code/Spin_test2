"""Cell-centre raster tests on closed spherical shells of calibrated thickness (millimetres)."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


FACE6 = ndimage.generate_binary_structure(3, 1)
FULL26 = ndimage.generate_binary_structure(3, 3)
DOMAIN_MM = 100.0
OUTER_RADIUS_MM = 3.91
WIDTHS_MM = (0.08, 0.12, 0.18, 0.24)
# Fractions of a grid cell along each axis; half-open phase interval.
PHASES = tuple(itertools.product((0., 0.25, 0.5), repeat=3))


def check(n, width, phase):
    h = DOMAIN_MM / n
    r = OUTER_RADIUS_MM
    inner = r - width
    margin = int(np.ceil(r / h)) + 3
    axis = h * np.arange(-margin, margin + 1)
    axes = [axis - p * h for p in phase]
    x, y, z = (u.reshape(shape) for u, shape in zip(axes, [(-1, 1, 1), (1, -1, 1), (1, 1, -1)]))
    rad2 = x*x + y*y + z*z
    shell = (rad2 >= inner**2) & (rad2 <= r*r)
    assert all(not np.take(shell, [0, -1], axis=k).any() for k in range(3))
    # A zero-padded region of interest with air on all faces is equivalent to the full domain.
    n_components = ndimage.label(shell, structure=FACE6)[1]
    center = tuple(int(np.argmin(abs(a))) for a in axes)
    assert not shell[center]  # Both probes below identify the component containing the centre.
    def sealed(structure):
        labels, _ = ndimage.label(~shell, structure=structure)
        return bool(labels[center] != labels[0, 0, 0])
    exact_volume = 4*np.pi/3*(r**3 - inner**3)
    exact_izz = 8*np.pi/15*(r**5 - inner**5)
    ijk = shell.nonzero()
    count = ijk[0].size
    izz = h**3 * np.sum(axes[0][ijk[0]]**2 + axes[1][ijk[1]]**2 + h*h/6)
    return dict(n=n, thickness_mm=width, cells_across=width/h, phase=list(phase),
                six_material_components=int(n_components), sealed_26_air=sealed(FULL26),
                sealed_6_air=sealed(FACE6), occupied_voxels=count,
                signed_volume_relerr=count*h**3/exact_volume - 1, signed_izz_relerr=izz/exact_izz - 1)


def run():
    rows = []
    summary = []
    for n in (256, 512, 1024):
        for width in WIDTHS_MM:
            subset = [check(n, width, phase) for phase in PHASES]
            rows += subset
            passed = [a['six_material_components'] == 1 and a['sealed_26_air'] for a in subset]
            failures = [a for a, p in zip(subset, passed) if not p]
            summary.append(dict(n=n, width_mm=width, cells_across=subset[0]['cells_across'],
                                nphases=len(subset), pass_material6_and_sealed26=sum(passed),
                                single_material6=sum(a['six_material_components'] == 1 for a in subset),
                                sealed_air6=sum(a['sealed_6_air'] for a in subset),
                                sealed_air26=sum(a['sealed_26_air'] for a in subset),
                                max_abs_izz_relerr=max(abs(a['signed_izz_relerr']) for a in subset),
                                smallest_abs_izz_relerr_among_failures=min((abs(a['signed_izz_relerr']) for a in failures), default=None)))
    return dict(schema='manufactured-spherical-shell-v1', length_unit='mm', domain_side_mm=DOMAIN_MM,
                outer_radius_mm=OUTER_RADIUS_MM, exact_material_components=1, analytic_cavity_sealed=True,
                criterion='face-connected material and centre air separated from exterior under 26-neighbour connectivity',
                center_sampling='shell centre displacement in grid spacings, rotations immaterial for the analytic sphere',
                phases_note='27 deterministic shifts in [0,.5]^3; symmetry-related shifts included; no probabilistic interpretation',
                summaries=summary, rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('shell_check.json'))
    args = p.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for s in result['summaries']:
        print(s)
