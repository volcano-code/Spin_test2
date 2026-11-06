"""Oriented moving-face flux for a sampled native discrete quadric clipped against fixed D'."""
from fractions import Fraction as Q

BITS = 140
SCALE = 1 << BITS
PAIRS = ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1))
# Symmetric degree-three Hammer rule. The weights average over the triangle.
RULE = ((Q(-27,48), (Q(1,3),)*3),) + tuple(
    (Q(25,48), tuple(Q(3 if i==j else 1,5) for i in range(3))) for j in range(3))


def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def mix(coords,weights):
    return tuple(dot(col,weights) for col in zip(*coords))


def monomials(p):
    return (1,*p,*(p[i]*p[j] for i,j in PAIRS))


def barycentric(parent, p, normal):
    axis=max(range(3),key=lambda i:abs(normal[i]))
    if not normal[axis]:
        raise ValueError('degenerate cutter parent')
    d=sub(p,parent[0])
    if dot(d,normal):
        raise ValueError('output point outside parent plane')
    b=cross(d,sub(parent[2],parent[0]))[axis]/normal[axis]
    c=cross(sub(parent[1],parent[0]),d)[axis]/normal[axis]
    a=1-b-c
    if min(a,b,c)<0:
        raise ValueError('output point outside parent triangle')
    return a,b,c


def face_flux(triangle,parent,motions,same_orientation):
    """Integrate m(x) * v.nout on this output triangle; motions are 3 x ncol per parent vertex."""
    outnormal=cross(sub(triangle[1],triangle[0]),sub(triangle[2],triangle[0]))
    pn=cross(sub(parent[1],parent[0]),sub(parent[2],parent[0]))
    orient=dot(outnormal,pn)
    if not orient or (orient>0)!=same_orientation:
        raise ValueError('incorrect retained-side orientation')
    interp=[barycentric(parent,p,pn) for p in triangle]
    parent_speed=[[dot(outnormal,col)/2 for col in zip(*v)] for v in motions]
    # Each table entry is the integral of a moment monomial times one affine
    # shape function of the parent. The oriented area vector enters parent_speed.
    weights=[[Q(0)]*3 for _ in range(10)]
    for weight,b in RULE:
        p=mix(triangle,b)
        shape=mix(interp,b)
        for row,monomial in zip(weights,monomials(p)):
            factor=monomial*weight
            for i in range(3):
                row[i]+=factor*shape[i]
    return [[dot(row,col) for col in zip(*parent_speed)] for row in weights]


def integrate(vertices,faces,owners,cutter_vertices,cutter_faces,motions,n_fixed_faces,operation):
    if operation not in ('intersection','difference'):
        raise ValueError('unsupported operation')
    if len(faces)!=len(owners) or len(motions)!=len(cutter_vertices):
        raise ValueError('incomplete output/source map')
    ncol=len(motions[0][0])
    if any(len(v)!=3 or any(len(line)!=ncol for line in v) for v in motions):
        raise ValueError('inconsistent motion dimensions')
    lower=[[0]*ncol for _ in range(10)]
    upper=[[0]*ncol for _ in range(10)]
    nmove=0
    for face,birth in zip(faces,owners):
        if not 0<=birth<n_fixed_faces+len(cutter_faces):
            raise ValueError('undefined face owner')
        if birth<n_fixed_faces:
            continue
        if any(i<0 or i>=len(vertices) for i in face):
            raise ValueError('invalid output index')
        f=cutter_faces[birth-n_fixed_faces]
        tri=[vertices[i] for i in face]
        parent=[cutter_vertices[i] for i in f]
        values=face_flux(tri,parent,[motions[i] for i in f],operation=='intersection')
        for j,line in enumerate(values):
            for k,value in enumerate(line):
                lo,r=divmod(value.numerator*SCALE,value.denominator)
                lower[j][k]+=lo
                upper[j][k]+=lo+bool(r)
        nmove+=1
    return lower,upper,nmove
