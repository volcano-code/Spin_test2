"""Exact transverse first-moment capacity of the frozen P19 material."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from p17_boundary_certificate import read_binary64_obj
from p20_physical_contract import P19_MATERIAL_SHA256, P19_MANIFEST_SHA256, P19_REPORT_SHA256, fraction_from_payload

HERE = Path(__file__).resolve().parent
FRAME_SHA256 = 'eaba6b60753fcb2141be425081897a8e891821349c166f5daff143d3ce9ea5fc'
REQUEST_SHA256 = '74d9cfa64417dd2c82d196ab27094651976954742a4c692667a1b19b1032de62'


def bound_json(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'Frozen input changed: {path.name}')
    return json.loads(data)


def half_triangle(triangle, axis, origin, sign):
    polygon = []
    for p, q in zip(triangle, triangle[1:] + triangle[:1]):
        u, v = p[axis] - origin, q[axis] - origin
        if sign*u >= 0:
            polygon.append(p)
        if u*v < 0:
            polygon.append(tuple(a + u*(b-a)/(u-v) for a,b in zip(p,q)))
    return polygon


def first_moment(triangles, axis, origin, side=0):
    """Flux of e_i (x_i-origin)^2/2. The cut-plane cap has zero flux."""
    j, k = (axis + 1) % 3, (axis + 2) % 3
    total = Fraction()
    for t in triangles:
        polygon = half_triangle(t, axis, origin, side) if side else t
        for idx in range(1, len(polygon) - 1):
            a, b, c = polygon[0], polygon[idx], polygon[idx+1]
            normal = (b[j]-a[j])*(c[k]-a[k]) - (b[k]-a[k])*(c[j]-a[j])
            x, y, z = a[axis]-origin, b[axis]-origin, c[axis]-origin
            total += normal*(x*x+y*y+z*z+x*y+x*z+y*z)/24
    return total


def number(q):
    return {'exact': str(q), 'decimal': float(q)}


def replay(root=HERE):
    case = root/'P19_material_case'
    manifest = bound_json(case/'manifest.json', P19_MANIFEST_SHA256)
    canonical = bound_json(root/'P19_CANONICAL_SOURCE_COMPLETE_REPORT.json', P19_REPORT_SHA256)['physical_proposal']
    frame = bound_json(root/'P20_EXACT_FRAME_AND_PHYSICS_REPORT.json', FRAME_SHA256)['actual_frame']
    request = bound_json(root/'P20_native_request.json', REQUEST_SHA256)
    if (request['target']['axis'] != manifest['construction']['spin_axis_reference'] or
            request['target']['axis'] != [0,0,1] or request['target']['origin'] != [0,0,0] or
            manifest['construction']['original_tip_reference'] != [0,0,0]):
        raise ValueError('Frozen target axis or origin changed')
    meshes = {}
    for name in ('F','D'):
        path = manifest['material_files'][name]
        mesh = read_binary64_obj(case/path)
        if mesh.sha256 != P19_MATERIAL_SHA256[name] or mesh.sha256 != manifest['files_sha256'][path]:
            raise ValueError('Frozen material changed: ' + name)
        meshes[name] = mesh
    recorded = {name: list(map(fraction_from_payload, canonical['materials'][name]['source_raw_ten_exact']))
                for name in meshes}
    triangles = {name:mesh.triangles for name,mesh in meshes.items()}
    results = {}
    for i, coord in enumerate(('x', 'y')):
        axes = {}
        for tip, origin in [('requested', Fraction()), ('represented', fraction_from_payload(frame['represented_tip_reference_exact'][i]))]:
            full = {name:first_moment(tris, i, origin) for name,tris in triangles.items()}
            for name, value in full.items():
                if value != recorded[name][i+1] - origin*recorded[name][0]:
                    raise ValueError('P19 independent full first moment differs: ' + name)
            low = first_moment(triangles['D'], i, origin, -1)
            high = first_moment(triangles['D'], i, origin, +1)
            if low > 0 or high < 0 or low+high != full['D']:
                raise ValueError('Signed half-domain decomposition failed')
            # Independent of the clipped integrals: all of D is at or above its leftmost vertex.
            support = min(Fraction(), min(p[i] for p in meshes['D'].vertices)-origin)
            coarse = recorded['F'][i+1] - origin*recorded['F'][0] + support*recorded['D'][0]
            minimum, maximum = full['F']+low, full['F']+high
            if coarse > minimum:
                raise ValueError('Independent volume and vertex bound differs')
            axes[tip] = dict(fixed=number(full['F']), design_negative=number(low), design_positive=number(high),
                             total_min=number(minimum), total_max=number(maximum), conservative_total_min=number(coarse),
                             zero_in_projection=(minimum <= 0 <= maximum))
        results[coord] = axes
    return dict(schema='W4-P25-frozen-transverse-capacity-v1', source_w4='624721bb44302ea4f7d84b3e9b0e57f654541d18',
                region='P19 saved binary64 F plus any measurable retained subset of P19 saved D',
                units='reference length to the fourth at unit density',
                prescribed_axis=request['target']['axis'], material_sha256={n:m.sha256 for n,m in meshes.items()},
                manifest_sha256=P19_MANIFEST_SHA256, canonical_sha256=P19_REPORT_SHA256, frame_sha256=FRAME_SHA256,
                request_sha256=REQUEST_SHA256, results=results, native_boolean_calls=0, formal_M6_executed=False)


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2, allow_nan=False))
