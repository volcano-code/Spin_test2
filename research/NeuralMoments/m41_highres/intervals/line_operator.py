"""Positive-side ten moments and the simple-root derivative, in native raw-q order."""
import numpy as np

B = np.array([1.] * 7 + [2.] * 3)
PAIRS = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]


def basis(points):
    return np.column_stack((np.ones(len(points)), *points.T,
                            *(points[:, i]*points[:, j] for i,j in PAIRS)))


def evaluate(bank, axis, q):
    """Integrate on each straight interval; flag unresolved line derivatives separately."""
    base, lo, hi, area = bank
    q = np.asarray(q, dtype=float)
    scale = np.linalg.norm(q)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("expected a finite nonzero quadric")
    n = q/scale
    hess_half = np.array([[n[4],n[9],n[8]], [n[9],n[5],n[7]], [n[8],n[7],n[6]]])
    a = hess_half[axis, axis]
    b = n[1+axis] + 2 * (base @ hess_half[axis])
    c = basis(base) @ (n*B)
    roots = np.full((len(lo), 2), np.inf)
    singular = np.zeros(len(lo), dtype=bool)
    tol = 1e-12
    if abs(a) < 1e-15:
        linear = abs(b) > tol
        roots[linear, 0] = -c[linear]/b[linear]
        singular |= ~linear & (abs(c) <= tol)
    else:
        disc = b*b - 4*a*c
        midpoint = -b/(2*a)
        singular |= (abs(disc) <= tol) & (midpoint >= lo-tol) & (midpoint <= hi+tol)
        repeated = disc == 0
        roots[repeated, :] = midpoint[repeated, None]
        real = disc > 0
        s = -.5*(b[real] + np.copysign(np.sqrt(disc[real]), b[real]))
        roots[real, 0] = s/a
        roots[real, 1] = c[real]/s
    singular |= np.any((abs(roots-lo[:,None]) < tol) | (abs(roots-hi[:,None]) < tol), axis=1)
    breaks = np.sort(np.column_stack((lo, np.clip(roots, lo[:,None], hi[:,None]), hi)), axis=1)
    moments = np.zeros(10)
    jac = np.zeros((10,10))
    for l, r in zip(breaks.T[:-1], breaks.T[1:]):
        mid, length = (l+r)*.5, r-l
        take = (length > 0) & (((a*mid+b)*mid+c) > 0)
        x = base[take].copy()
        x[:, axis] = mid[take]
        avg = basis(x)
        avg[:, 4+axis] += length[take]**2/12
        moments += (area*length[take]) @ avg
    roots_used = 0
    for t in roots.T:
        take = (t > lo) & (t < hi) & ~singular
        x = base[take].copy()
        x[:, axis] = t[take]
        phi = basis(x)
        weights = area/(scale*abs(2*a*t[take]+b[take]))
        jac += (phi.T*weights) @ (phi*B)
        roots_used += len(x)
    return moments, jac, {"regular_roots":roots_used, "guarded_intervals":int(singular.sum()),
                          "derivative_supported":bool(roots_used and not singular.any())}
