"""Exact feasibility bound for every retained subset of the frozen Heart design."""
import argparse
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path

from .real_h8e_archive_m8 import FROZEN_REQUEST_SHA256, check_closed_indexed_surface, require

# The W1 exact self-intersection check on this connected, outward design boundary
# tested all 218939 candidate pairs. Reconstruct its exact coordinate/index bytes.
DESIGN_BOUNDARY_SHA = 'eff414f5daaf52f8ae847bb851d3cbb5cfd604bce0f88a5a56f99ed6b1efb9ad'


def clip(triangle, sign):
    """Oriented polygon clipped by sign*x >= 0. A point on the plane occurs once."""
    poly = []
    for a, b in zip(triangle, (*triangle[1:], triangle[0])):
        if sign*a[0] >= 0:
            poly.append(a)
        if a[0]*b[0] < 0:
            t = a[0]/(a[0]-b[0])
            poly.append(tuple(a[k] + t*(b[k]-a[k]) for k in range(3)))
    return [(poly[0], poly[i], poly[i+1]) for i in range(1, len(poly)-1)]


def flux(triangle):
    """Two exact surface integrals for first x moment. Planar cap flux is zero."""
    a,b,c = triangle
    ux,uy,uz = (b[i]-a[i] for i in range(3))
    vx,vy,vz = (c[i]-a[i] for i in range(3))
    nx,ny = uy*vz-uz*vy, uz*vx-ux*vz  # oriented double area components
    sx = a[0]+b[0]+c[0]
    sy = a[1]+b[1]+c[1]
    return (nx*(sx*sx+sum(p[0]**2 for p in triangle))/48,
            ny*(sx*sy+sum(p[0]*p[1] for p in triangle))/24)


def integrate(vertices, faces):
    neg, check, pos, whole, volume = (F(0),)*5
    crossing = 0
    for i,j,k in faces:
        triangle = a,b,c = (vertices[i],vertices[j],vertices[k])
        xs = [a[0],b[0],c[0]]
        six_volume = sum(a[d]*(b[(d+1)%3]*c[(d+2)%3]-b[(d+2)%3]*c[(d+1)%3]) for d in range(3))
        volume += six_volume/6
        whole += six_volume*sum(xs)/24  # independent signed tetra moment
        if min(xs) == max(xs) == 0:
            continue
        if min(xs) < 0:
            negative = [triangle] if max(xs)<=0 else clip(triangle, -1)
            for piece in negative:
                primary, alternate = flux(piece)
                neg += primary
                check += alternate
        if max(xs) > 0:
            positive = [triangle] if min(xs)>=0 else clip(triangle, 1)
            pos += sum(flux(piece)[0] for piece in positive)
        crossing += min(xs)<0<max(xs)
    require(neg==check and neg+pos==whole, 'halfspace flux identity failed')
    require(volume>0 and neg<=0<=pos, 'invalid solid orientation or volume')
    return neg, pos, whole, volume, crossing


def interval(q, places=15):
    """Outward decimal rationals, sufficient for verification without giant output integers."""
    scale = 10**places
    def fmt(n):
        return ('-' if n<0 else '') + str(abs(n)//scale) + '.' + str(abs(n)%scale).zfill(places)
    return [fmt(q*scale//1), fmt(-((-q*scale)//1))]


def certificate(path):
    data = path.read_bytes()
    require(sha256(data).hexdigest()==FROZEN_REQUEST_SHA256, 'changed prepared request')
    packet = json.loads(data)
    r = packet['request']
    mm_per_unit = F(packet['evidence']['normalized_scale_mm_per_unit'])
    require(mm_per_unit==25, 'unsupported mesh length scale')
    require(r['target_axes']==[[1,0,0],[0,1,0],[0,0,1]] and r['target_origin'][:2]==[0,0]
            and r['material_density']==1 and r['mode']=='SPINNING_TOP', 'unsupported frame or material')
    v = [tuple(map(F,p)) for p in r['design_vertices']]
    faces = r['design_faces']
    check_closed_indexed_surface(faces, len(v))
    lines = [f'{len(v)} {len(faces)}', *(' '.join(map(str,p)) for p in v), *(' '.join(map(str,f)) for f in faces)]
    mesh_sha = sha256(('\n'.join(lines)+'\n').encode()).hexdigest()
    require(mesh_sha==DESIGN_BOUNDARY_SHA, 'different embedded design boundary')
    lo, hi, full, mass, crossing = integrate(v,faces)
    # The frozen shell moments are specified in the model frame. With identity axes
    # and zero target x/y origin, the first x moment is exactly invariant.
    fm,fx,fy,fz,fxx,fyy,fzz,fyz,fxz,fxy = map(F,r['fixed_shell_moments'])
    oz = F(r['target_origin'][2])
    fixed = (fm,fx,fy,fz-oz*fm,fxx,fyy,fzz-2*oz*fz+oz*oz*fm,fyz-oz*fy,fxz-oz*fx,fxy)
    bound = fx+lo
    require(bound>0, 'constraint has no strictly positive certified lower bound')
    # Each total target moment is bounded by |fixed| + mass(design)*sup|monomial|.
    # The positive mass and bounding box of the design bound all occupations.
    x,y,z = [max(abs(p[k]-(oz if k==2 else 0)) for p in v) for k in range(3)]
    monomials = (1,x,y,z,x*x,y*y,z*z,y*z,x*z,x*y)
    norm_bound = max(1, sum(abs(m)+mass*b for m,b in zip(fixed,monomials)))
    relative = bound/norm_bound
    total_mass = fm+mass
    offset = bound/total_mass
    require(bound>F(r['settings']['equation_tolerance']) and relative>F(r['settings']['physical_relative_tolerance']),
            'insufficient lower bound to exclude strict tolerances')
    return dict(schema='w0-heart-frozen-axis-feasibility-v1', prepared_request_sha256=FROZEN_REQUEST_SHA256,
            design_boundary_sha256=mesh_sha, design_vertices=len(v), design_faces=len(faces), crossing_faces=crossing,
            target_origin=r['target_origin'], target_axes=r['target_axes'], material_density=r['material_density'],
            units='normalized target frame, first moment = material mass times normalized length',
            fixed_shell_first_x_exact=str(fx), fixed_shell_first_x_interval=interval(fx),
            minimum_variable_first_x_interval=interval(lo), maximum_variable_first_x_interval=interval(hi),
            full_design_first_x_interval=interval(full), full_design_mass_exact=str(mass),
            raw_equation_residual_lower_bound_interval=interval(bound),
            total_mass_upper_bound_interval=interval(total_mass),
            com_x_normalized_lower_bound_interval=interval(offset),
            normalized_scale_mm_per_unit=int(mm_per_unit), com_x_mm_lower_bound_interval=interval(offset*mm_per_unit),
            physical_normalizer_upper_bound_interval=interval(norm_bound),
            physical_relative_residual_lower_bound_interval=interval(relative),
            equation_tolerance=r['settings']['equation_tolerance'], physical_tolerance=r['settings']['physical_relative_tolerance'],
            negative_half_x_flux_equals_y_flux_exactly=True,
            negative_and_positive_half_fluxes_sum_to_signed_tetra_moment_exactly=True,
            scope='fixed locked shell, registered axis, uniform density, variable support a subset of the frozen design solid',
            feasible_strict_terminal_under_frozen_contract=False, native_calls=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request',type=Path)
    print(json.dumps(certificate(parser.parse_args().request),indent=2,allow_nan=False))
