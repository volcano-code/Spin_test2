"""Certify one continuous ellipsoidal cavity for the registered single-density M32 body."""
import argparse
from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
import json
import math
from pathlib import Path



A_TEN = '5912e9f1dc44378919f5c00690983d284189edadc857ea3b354f09c4b86a1986'
D_PRIME = 'c3633c74bc74c2be488eb22de108454276f28941f0477cfee5837f57db99d622'
# Fixed decimal rationals chosen by a preliminary search of ellipsoid shapes.
X, Z, AXY, H, YAW_HALF_TAN = map(Q, ('0.486743','0.957761','0.007523','0.128712','1.062001'))


@dataclass(frozen=True)
class I:
    lo: Q
    hi: Q

    @staticmethod
    def exact(x): return I(Q(x),Q(x))
    def __add__(self,b):
        if not isinstance(b,I): b=I.exact(b)
        return I(self.lo+b.lo,self.hi+b.hi)
    __radd__=__add__
    def __neg__(self):return I(-self.hi,-self.lo)
    def __sub__(self,b):return self+(-b)
    def __rsub__(self,b):return (-self)+b
    def __mul__(self,b):
        if not isinstance(b,I): b=I.exact(b)
        vals=[x*y for x in (self.lo,self.hi) for y in (b.lo,b.hi)]
        return I(min(vals),max(vals))
    __rmul__=__mul__
    def reciprocal(self):
        if self.lo<=0<=self.hi:raise ValueError("division by interval including zero")
        return I(1/self.hi,1/self.lo)
    def __truediv__(self,b):return self*(b.reciprocal() if isinstance(b,I) else 1/Q(b))
    def __rtruediv__(self,b):return self.reciprocal()*b


def enclose_hex(value):
    a,b=(int(x,0) for x in value.split('/'))
    q,r=divmod(a*2**128,b)
    return I(Q(q,2**128),Q(q+bool(r),2**128))


def pi_interval():
    def atan_inv(q):
        s=sum(Q((-1)**k,(2*k+1)*q**(2*k+1)) for k in range(24))
        return I(s,s+Q(1,49*q**49))
    return 16*atan_inv(5)-4*atan_inv(239)


def parameters(m):
    # B's columns: a*u, b*v in xy, followed by (sx,sy,h); true unit ball image.
    t=YAW_HALF_TAN; co=(1-t*t)/(1+t*t); si=2*t/(1+t*t)
    mass=m[1]/X; y=m[2]/mass
    b=3*mass/(4*pi_interval()*AXY*H)
    sx=5*(m[8]/mass-X*Z)/H
    sy=5*(m[7]/mass-y*Z)/H
    center=[I.exact(X),y,I.exact(Z)]
    matrix=[[I.exact(AXY*co),-b*si,sx],[I.exact(AXY*si),b*co,sy],[I.exact(0),I.exact(0),I.exact(H)]]
    if min(mass.lo,b.lo)<=0: raise ValueError("invalid ellipsoid")
    iz_removed=mass*(X*X+y*y+(AXY*AXY+b*b+sx*sx+sy*sy)/5)
    return center,matrix,mass,1-iz_removed/(m[4]+m[5]),b


def coefficient_hint(center,matrix,b, *, tip_z):
    """Binary64 approximation of (p-c)ᵀB⁻ᵀB⁻¹(p-c)-1, with origin at the tip."""
    mid=lambda i:float((i.lo+i.hi)/2)
    t=float(YAW_HALF_TAN);co=(1-t*t)/(1+t*t);si=2*t/(1+t*t)
    sx,sy=mid(matrix[0][2]),mid(matrix[1][2]);a,h=float(AXY),float(H);b=mid(b)
    inv=[[co/a,si/a,-(co*sx+si*sy)/(a*h)],[-si/b,co/b,(si*sx-co*sy)/(b*h)],[0,0,1/h]]
    cs=list(map(mid,center));cs[2]-=tip_z
    cols=list(zip(*inv));k=[[dot(x,y) for y in cols] for x in cols]
    kc=[dot(row,cs) for row in k]
    q=[dot(kc,cs)-1,*(-2*v for v in kc),k[0][0],k[1][1],k[2][2],k[1][2],k[0][2],k[0][1]]
    norm=math.sqrt(dot(q,q))
    return [v/norm for v in q]


def nominal(value):
    if value.lo==value.hi:return value.lo
    return Q(round((value.lo+value.hi)*2**39),2**40)


def distance_from(interval,x):
    return max(abs(x-interval.lo),abs(x-interval.hi))


def dot(a,b):return sum(x*y for x,y in zip(a,b))


def winding(points,triangles,p):
    x,y,z=p;hits=[];total=0
    for face in triangles:
        a,b,c=(points[i] for i in face)
        if any(v<min(a[j],b[j],c[j]) or v>max(a[j],b[j],c[j]) for j,v in enumerate((x,y))):continue
        ax,ay=a[0]-c[0],a[1]-c[1];bx,by=b[0]-c[0],b[1]-c[1];vx,vy=x-c[0],y-c[1]
        det=ax*by-ay*bx
        if det==0:
            if vx*by-vy*bx==0 and ax*vy-ay*vx==0:raise ValueError("ray overlaps a projected triangle")
            continue
        u=(vx*by-vy*bx)/det;v=(ax*vy-ay*vx)/det;w=1-u-v
        if min(u,v,w)<0:continue
        if min(u,v,w)==0:raise ValueError("ambiguous ray edge")
        boundary=u*a[2]+v*b[2]+w*c[2]
        if boundary==z:raise ValueError("point on boundary")
        if boundary>z:
            s=1 if det>0 else -1
            total+=s;hits.append(s)
    return total,hits


def certify(operand,ten):
    dp=operand.read_bytes();ap=ten.read_bytes()
    if sha256(dp).hexdigest()!=D_PRIME or sha256(ap).hexdigest()!=A_TEN:raise ValueError("source hash mismatch")
    m=list(map(enclose_hex,ap.decode().splitlines()))
    center,matrix,mass,izratio,b=parameters(m)
    c=list(map(nominal,center));B=[list(map(nominal,row)) for row in matrix]
    # The entire true ellipsoid lies in c+B*ball plus a delta box componentwise.
    delta=max(distance_from(center[i],c[i])+sum(distance_from(matrix[i][k],B[i][k]) for k in range(3)) for i in range(3))
    if distance_from(center[1],c[1])>=min(AXY,b.lo):raise ValueError("interior center probe unverified")
    ext=[sum(map(abs,row))+delta for row in B]
    low=[x-r for x,r in zip(c,ext)];high=[x+r for x,r in zip(c,ext)]
    raw=dp.decode().splitlines();nv,nf=map(int,raw[0].split())
    points=[tuple(map(Q,line.split())) for line in raw[1:1+nv]]
    triangles=[tuple(map(int,line.split())) for line in raw[1+nv:]]
    if len(triangles)!=nf:raise ValueError("invalid mesh length")
    # Preclassify vertices by exact position relative to the enclosing box.
    codes=[]
    for v in points:
        code=0
        for k in range(3):
            if v[k]<low[k]:code|=1<<(2*k)
            if v[k]>high[k]:code|=2<<(2*k)
        codes.append(code)
    near=[(i,f) for i,f in enumerate(triangles) if not (codes[f[0]] & codes[f[1]] & codes[f[2]])]
    separator_path=Path(__file__).with_name("M32_ELLIPSOID_SEPARATORS.txt")
    # Normals merely propose separating directions. Source triangle selection,
    # every support test and the winding ray are checked with rational arithmetic.
    hints=[tuple(map(int,r.split())) for r in separator_path.read_text().splitlines()]
    if len(hints)!=len(near):raise ValueError("missing separating plane")
    worst=None
    for (i,face),row in zip(near,hints):
        if len(row)!=4 or row[0]!=i or not any(row[1:]):raise ValueError("invalid separating plane")
        n=row[1:]
        lower=min(dot(n,points[k]) for k in face)-dot(n,c)-delta*sum(map(abs,n))
        radius2=sum(dot(n,col)**2 for col in zip(*B))
        if lower<=0 or lower*lower<=radius2:raise ValueError(f"strict rational separator failed on {face}")
        ratio=lower*lower/radius2
        worst=ratio if worst is None else min(worst,ratio)
    wind,hits=winding(points,triangles,c)
    if wind!=1 or izratio.lo<Q(9,10):raise ValueError("winding or inertia gate failed")
    # Gershgorin upper bound on the largest transverse centroidal principal inertia.
    # The two cross terms involving z vanish exactly by construction.
    def removed_second(i,j):
        return mass*(center[i]*center[j]+dot(matrix[i],matrix[j])/5)
    qxx=m[4]-removed_second(0,0);qyy=m[5]-removed_second(1,1);qxy=m[9]-removed_second(0,1)
    rem_z=m[3]-mass*Z
    qzz_cent=m[6]-removed_second(2,2)-rem_z*rem_z/(m[0]-mass)
    margin=(qxx+qyy).lo-max((qxx+qzz_cent).hi,(qyy+qzz_cent).hi)-max(abs(qxy.lo),abs(qxy.hi))
    if margin<=0:raise ValueError("centroidal target-axis inertia not certified dominant")
    mr=1-mass/m[0]
    def bounds(v,d=15):
        s=10**d
        return [str(Q(v.lo*s//1,s)),str(Q(-(-v.hi*s//1),s))]
    return dict(schema="W4_M32_INTERIOR_SINGLE_ELLIPSOID_V1",design_operand_sha256=D_PRIME,
        whole_body_exact_ten_payload_sha256=A_TEN,vertices=nv,faces=nf,exact_aabb_separated=nf-len(near),
        rational_separators=len(near),min_support_ratio_squared_floor=str(Q(worst*10**10//1,10**10)),
        nominal_center_model=list(map(str,c)),nominal_matrix=[[str(x) for x in row] for row in B],
        parametric_error_box_sup_model_upper=str(Q(-(-delta*10**18//1),10**18)),ray_winding=wind,positive_z_hits=hits,
        x=str(X),z=str(Z),xy_semiaxis=str(AXY),height_semiaxis=str(H),yaw_tan_half=str(YAW_HALF_TAN),
        removed_volume_model_interval=bounds(mass),retained_mass_ratio_interval=bounds(mr),axis_inertia_ratio_interval=bounds(izratio),
        centroidal_dominant_Izz_gap_lower=str(Q(margin*10**15//1,10**15)),
        coefficient_hint_model_normalized_native_order=coefficient_hint(center,matrix,b,tip_z=0.),
        coefficient_hint_tip_frame_model_unit_normalized_native_order=coefficient_hint(center,matrix,b,tip_z=-0.16),
        native_coefficient_order=["constant","x","y","z","xx","yy","zz","2yz","2xz","2xy"],
        retained_side="outside ellipsoid: beta >= 0; binary64 hints are not feasibility certificates",
        analytic_zero_total_moments=["x","y","xz","yz"],unit_density=True,certified_positive_volume_inside_exact_design=True,
        quadric_kind="continuous bounded ellipsoid fully contained in Dprime",physical_prototype_validated=False,
        represented_native_coefficient_vector_validated=False,strict_newton_terminal=False,objective_optimality_claimed=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('Dprime',type=Path);p.add_argument('A_ten',type=Path);a=p.parse_args()
    print(json.dumps(certify(a.Dprime,a.A_ten),indent=2,sort_keys=True))
