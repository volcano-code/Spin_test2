"""Exact D1 references for minimum mass at 90% initial axial inertia.

The initial body is a unit-radius, unit-height cylinder of unit density
centered at the origin. Its concentric carveable cylinder has height 5/8;
the remaining end caps and outer wall are fixed. All mass moments below are
divided by pi. These analytic tests cover moment constraints only.
"""
from fractions import Fraction as Q
import json

from .moment_coordinates import RAW_R_ORDER

TAU = Q(9, 10)
HEIGHT = Q(5, 8)


def cylinder_moments(radius_squared, height):
    mass = radius_squared * height
    transverse = mass * radius_squared / 4
    return dict(zip(RAW_R_ORDER, (mass, 0, 0, 0, transverse, transverse,
                                 mass * height**2 / 12, 0, 0, 0), strict=True))


def axial_inertia(raw, x_axis=0):
    """Moment about the fixed line (x_axis,0,z), including its offset."""
    return raw['QXX'] + raw['QYY'] - 2*x_axis*raw['LX'] + x_axis**2*raw['M']


def centered_case(design_radius_squared, cavity_radius_squared, multiplier):
    b, t, eta = design_radius_squared, cavity_radius_squared, multiplier
    if not (0 <= t <= b < 1 and eta >= 0):
        raise ValueError("invalid cavity, design domain or dual multiplier")
    full = cylinder_moments(Q(1), Q(1))
    design = cylinder_moments(b, HEIGHT)
    cavity = cylinder_moments(t, HEIGHT)
    retained = {k: full[k] - cavity[k] for k in RAW_R_ORDER}
    reference = axial_inertia(full)
    target = TAU * reference
    fixed_mass = full['M'] - design['M']
    fixed_inertia = reference - axial_inertia(design)

    # Lower bound for any retained subset, including nonsymmetric cuts:
    # mass + eta*(target - inertia) is at most feasible mass. Its infimum
    # over D selects 1-eta*s < 0, with s=r^2 and d(volume)/pi=HEIGHT ds.
    edge = min(b, 1 / eta) if eta else b
    integral = HEIGHT * ((b-edge) - eta*(b**2-edge**2)/2)
    lower = fixed_mass + eta*(target-fixed_inertia) + integral
    inertia = axial_inertia(retained)
    mass = retained['M']
    if inertia < target or mass != lower:
        raise ValueError("primal feasibility and matching lower bound required")
    return dict(design_radius_squared=b, cavity_radius_squared=t,
                reference_inertia_over_pi=reference, retained_mass_over_pi=mass,
                retained_volume_over_pi=mass, inertia_ratio=inertia/reference,
                inertia_slack_over_pi=inertia-target, multiplier=eta,
                dual_mass_lower_over_pi=lower, total_raw_over_pi=retained)


def references():
    full = cylinder_moments(Q(1), Q(1))
    axis = Q(1, 4)
    radius = Q(1, 2)
    max_removed_mass = cylinder_moments(radius**2, HEIGHT)['M']
    # The entire full body's first moment about the displaced line is -axis.
    # Each removed mass element has (x-axis) >= -(radius+axis).
    upper = full['LX'] - axis*full['M'] + (radius+axis)*max_removed_mass
    return dict(
        schema='W0_ANALYTIC_MASS90_CYLINDERS_V1', inertia_fraction=TAU,
        active=centered_case(Q(4, 5), Q(2, 5), Q(5, 2)),
        slack=centered_case(radius**2, radius**2, Q(0)),
        infeasible=dict(axis_x=axis, design_radius_squared=radius**2,
                        reference_inertia_over_pi=axial_inertia(full, axis),
                        first_x_about_axis_upper_over_pi=upper,
                        balance_excluded=upper < 0),
    )


if __name__ == '__main__':
    print(json.dumps(references(), default=str, indent=2))
