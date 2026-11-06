"""Check the unavoidable tangent symmetric-head mismatch on the represented discrete map."""
from fractions import Fraction as Q
from math import isqrt

from crossing_flux import SCALE,dot


def sqrt_interval(s,places=24):
    f=10**places
    k=isqrt(s.numerator*f*f//s.denominator)
    return Q(k,f),Q(k+1,f)


def tangent_skew(flux):
    lo,hi=flux['lower_tip'],flux['upper_tip']
    b=[1]*7+[2]*3
    p=[Q(x)/w for w,x in zip(b,flux['q'])]
    P=[[int(i==j)-p[i]*p[j]/dot(p,p) for j in range(10)] for i in range(10)]
    J=[[Q(a+z,2*SCALE*c) for a,z,c in zip(x,y,b)] for x,y in zip(lo,hi)]
    def matmul(A,B):
        return [[dot(r,c) for c in zip(*B)] for r in A]
    T=matmul(matmul(P,J),P)
    # ||P E P||_F <= ||E||_F <= 10 * max radius (B^{-1} also contracts).
    radius=10*max(Q(z-a,2*SCALE) for x,y in zip(lo,hi) for a,z in zip(x,y))
    norm_hi=sqrt_interval(sum(v*v for row in T for v in row))[1]
    skew_lo=sqrt_interval(sum(((T[i][j]-T[j][i])/2)**2 for i in range(10) for j in range(10)))[0]
    bound=max(Q(0),(skew_lo-radius)/(norm_hi+radius))
    return dict(frobenius_tangent_Jp_upper_approx=float(norm_hi+radius),
                frobenius_skew_tangent_Jp_lower_approx=float(skew_lo-radius),
                certified_relative_distance_to_any_symmetric_tangent_head_at_least=f'{int(bound*10**12)}/1000000000000',
                scope='tip-frame Jp=Jc*inverse(B), projected onto the tangent of the native logistic normalization (normal inverse(B)*c); represented source motion at this sample')
