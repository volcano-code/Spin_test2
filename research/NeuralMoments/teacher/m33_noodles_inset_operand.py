"""Build one exact E-prime operand for the registered M32 CSG chain."""
import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

from m16_native_acceptance import require
from m19_noodles_fda_audit import PAIR_ORDER, fraction_signed_ten, oriented_topology
from m31_candidate_contact_screen import NOODLES, NOODLES_SHA, noodles_sources, read
from m32_noodles_inset_wall import SCALE


M32_SHA = '1ba4ffc573bbc3a8a6f83b548a5b73b8b85e92e50eadc49f092634588e97a74b'


def rational_token(x):
    return f'{x.numerator}/{x.denominator}'


def affine_moments(raw, center, scale):
    """Push all ten unit-density mass moments through x -> scale*x + shift."""
    shift = [c * (1 - scale) for c in center]
    result = [scale**3 * raw[0]]
    for j in range(3):
        result.append(scale**3 * (scale * raw[j+1] + shift[j] * raw[0]))
    for index, (i, j) in enumerate(PAIR_ORDER, 4):
        value = (scale**2 * raw[index] + scale * (shift[i]*raw[j+1] + shift[j]*raw[i+1])
                 + shift[i]*shift[j]*raw[0])
        result.append(scale**3 * value)
    return result


def make_operand(vertices, faces, center, scale):
    """Produce the first of the two native exact_input_boolean.cpp input records."""
    require(0 < scale < 1, 'INVALID_SCALE')
    source = [tuple(map(Fraction, row)) for row in vertices]
    faces = [tuple(map(int, face)) for face in faces]
    center = tuple(map(Fraction, center))
    inner = [tuple(c + scale*(x-c) for c, x in zip(center, pt)) for pt in source]
    edges = oriented_topology(inner, faces)
    raw = fraction_signed_ten(source, faces)
    inner_raw = fraction_signed_ten(inner, faces)
    require(inner_raw == affine_moments(raw, center, scale), 'INSET_MOMENTS_MISMATCH')

    lines = [f'{len(inner)} {len(faces)}']
    lines.extend(' '.join(map(rational_token, row)) for row in inner)
    lines.extend(' '.join(map(str, face)) for face in faces)
    payload = ('\n'.join(lines) + '\n').encode('ascii')
    drift = [abs(x - Fraction.from_float(float(x))) for pt in inner for x in pt]
    return payload, dict(
        operand_sha256=sha256(payload).hexdigest(), bytes=len(payload), vertex_count=len(inner),
        face_count=len(faces), paired_edges=edges, scale=str(scale),
        center=[rational_token(x) for x in center], non_binary64_coordinates=sum(x != 0 for x in drift),
        coordinate_count=len(drift), maximum_abs_binary64_drift_model=float(max(drift)),
        raw_order=['1','x','y','z','xx','yy','zz','yz','xz','xy'],
        source_ten=list(map(rational_token, raw)), inset_ten=list(map(rational_token, inner_raw)),
        affine_moment_check_passed=True,
        fixed_envelope_layer_volume_fraction=str(1-inner_raw[0]/raw[0]),
        fixed_envelope_layer_Iz_fraction=str(1-(inner_raw[4]+inner_raw[5])/(raw[4]+raw[5])))


def build(main_tree):
    manifest = json.loads(read(main_tree, NOODLES/'frozen-parts-v7c/manifest.json',
                               NOODLES_SHA['frozen-parts-v7c/manifest.json']))
    m32 = json.loads(read(Path(__file__).parent, 'M32_NOODLES_INSET_WALL.json', M32_SHA))
    center = manifest['spec']['ellipsoid_center_model']
    require(m32['source_manifest_sha256'] == NOODLES_SHA['frozen-parts-v7c/manifest.json']
            and list(map(Fraction, m32['center_exact_model'])) == list(map(Fraction, center))
            and m32['scale_exact'] == str(SCALE), 'M32_OPERAND_REGISTRATION_DRIFT')
    vertices, faces, fixed_boxes, sources = noodles_sources(main_tree, manifest)
    payload, report = make_operand(vertices, faces, center, SCALE)
    report.update(schema='w2-m33-noodles-inset-operand-v1', m32_registration_sha256=M32_SHA, operand='Eprime, not Dprime or Fprime',
                  frame='model coordinates about (0,0,0); unit density; axial inertia about the registered z axis',
                  source_manifest_sha256=NOODLES_SHA['frozen-parts-v7c/manifest.json'],
                  source_envelope_obj_sha256=manifest['parts']['outer-ellipsoid']['sha256_model'],
                  checked_source_obj_count=1+len(fixed_boxes), generators=sources, native_boolean_calls=0)
    return payload, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-tree', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New native operand text file')
    args = parser.parse_args()
    payload, report = build(args.main_tree)
    with args.output.open('xb') as stream:
        stream.write(payload)
    print(json.dumps(report, indent=2, sort_keys=True))
