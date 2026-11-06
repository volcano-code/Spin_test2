"""H9: independently certify an *interior* edge/edge event of an analytic surrogate.

All archival snapshots are original, hash-checked, pinned Cherchi materials.
The interval proof is ONLY for the frozen, fixed-normal inverse-Gauss
surrogate; no fresh native Boolean, production smooth stratum, or Newton claim.

Dependency-free (Python 3.11+). Never accepts a replacement or fake ZIP.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as Q
import gzip
from hashlib import sha256
from math import isqrt
import json
from pathlib import Path
import zipfile

ZIP_SHA = '99a050f4604109e0323c0110a029aa5d7601d729d755e6956ad48ed8c77ccda9'
REQUEST_SHA = '1d2b06063c5b2c7010fcb2764518c05c9707f013d6914bc4e20298f8516e061c'
SOURCE_IDS=(24,90)
DESIGN_IDS=(5517,5518)
SAMPLES=(0,57,58,59,60,61,62,63,64)
LOW=Q('1.5e-6')
HIGH=Q('3e-6')
UNIT=10**40


def need(ok,reason):
    if not ok: raise ValueError(reason)


def sha(data):return sha256(data).hexdigest()


def file_sha(path):
    digest=sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def numerator_ceil(a,b):return -((-a)//b)


@dataclass(frozen=True)
class I:
    lo:int
    hi:int

    def __post_init__(self):need(self.lo<=self.hi,'EMPTY_INTERVAL')

    @classmethod
    def of(cls,value):
        n=Q.from_float(value) if isinstance(value,float) else Q(value)
        top=n.numerator*UNIT
        return cls(top//n.denominator,numerator_ceil(top,n.denominator))

    @classmethod
    def between(cls,a,b):
        need(a<=b,'REVERSED_INTERVAL')
        return cls(cls.of(a).lo,cls.of(b).hi)

    def __neg__(self):return I(-self.hi,-self.lo)
    def __add__(self,other):
        y=as_i(other);return I(self.lo+y.lo,self.hi+y.hi)
    __radd__=__add__
    def __sub__(self,other):return self+-as_i(other)
    def __rsub__(self,other):return as_i(other)+-self
    def __mul__(self,other):
        y=as_i(other)
        choices=[a*b for a in (self.lo,self.hi) for b in (y.lo,y.hi)]
        return I(min(choices)//UNIT,numerator_ceil(max(choices),UNIT))
    __rmul__=__mul__
    def __truediv__(self,other):
        y=as_i(other);need(y.lo>0 or y.hi<0,'INTERVAL_DIVISION_BY_ZERO')
        choices=[(a*UNIT,b) for a in (self.lo,self.hi) for b in (y.lo,y.hi)]
        return I(min(a//b for a,b in choices),max(numerator_ceil(a,b) for a,b in choices))
    def __rtruediv__(self,other):return as_i(other)/self
    def sqrt(self):
        need(self.lo>=0,'INTERVAL_SQRT_OF_NEGATIVE')
        lo=isqrt(self.lo*UNIT)
        hi=isqrt(self.hi*UNIT)
        if hi*hi!=self.hi*UNIT:hi+=1
        return I(lo,hi)
    def sign(self):
        if self.lo>0:return 1
        if self.hi<0:return -1
        if self.lo==self.hi==0:return 0
        return None
    def strictly_unit(self):return self.lo>0 and self.hi<UNIT
    def printable(self):return {'lower':float(Q(self.lo,UNIT)),'upper':float(Q(self.hi,UNIT))}


def as_i(v):return v if isinstance(v,I) else I.of(v)


@dataclass(frozen=True)
class J:
    v:I
    d:I
    @classmethod
    def of(cls,value):return cls(I.of(value),I.of(0))
    @classmethod
    def variable(cls,lo,hi):return cls(I.between(lo,hi),I.of(1))
    def __neg__(self):return J(-self.v,-self.d)
    def __add__(self,other):
        y=as_j(other);return J(self.v+y.v,self.d+y.d)
    __radd__=__add__
    def __sub__(self,other):return self+-as_j(other)
    def __rsub__(self,other):return as_j(other)+-self
    def __mul__(self,other):
        y=as_j(other);return J(self.v*y.v,self.d*y.v+self.v*y.d)
    __rmul__=__mul__
    def __truediv__(self,other):
        y=as_j(other)
        return J(self.v/y.v,(self.d*y.v-self.v*y.d)/(y.v*y.v))
    def __rtruediv__(self,other):return as_j(other)/self
    def sqrt(self):
        root=self.v.sqrt();need(root.lo>0,'DERIVATIVE_SQRT_ZERO')
        return J(root,self.d/(2*root))


def as_j(v):return v if isinstance(v,J) else J.of(v)


def vec(values):return [J.of(Q.from_float(x) if isinstance(x,float) else Q(x)) for x in values]

def vsub(a,b):return [x-y for x,y in zip(a,b)]

def vadd(a,b):return [x+y for x,y in zip(a,b)]

def dot(a,b):return sum((x*y for x,y in zip(a,b)),J.of(0))

def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def center(q,t):
    a,b,g=q[4:7]
    det=b*g-t*t
    need(det.v.lo>0 and a.v.lo>0,'NON_ELLIPSOID_SURROGATE')
    return [-q[1]/(2*a),-(g*q[2]-t*q[3])/(2*det),-(-t*q[2]+b*q[3])/(2*det)]


def gauss(q,t,n):
    a,b,g=q[4:7]
    determinant=b*g-t*t
    c=center(q,t)
    direction=[n[0]/a,(g*n[1]-t*n[2])/determinant,
               (-t*n[1]+b*n[2])/determinant]
    energy=-q[0]+(q[1]*q[1]/a+(g*q[2]*q[2]-2*t*q[2]*q[3]+b*q[3]*q[3])/determinant)/4
    need(energy.v.lo>0,'INVALID_SURROGATE_ELLIPSOID_LEVEL')
    scalar=(energy/dot(n,direction)).sqrt()
    return vadd(c,[scalar*e for e in direction])


def incident(faces,edge):
    return {tuple(sorted(f)) for f in faces if set(edge).issubset(f)}


def archival_data(archive, sample_indices=SAMPLES):
    need(file_sha(archive)==ZIP_SHA,'UNTRUSTED_ARCHIVE_SHA256')
    with zipfile.ZipFile(archive) as z:
        request_bytes=z.read('artifacts/h8-request/prepared-request.json')
        need(sha(request_bytes)==REQUEST_SHA,'REQUEST_SHA256_MISMATCH')
        request=json.loads(request_bytes)
        report_bytes=z.read('artifacts/h8f-all81-points/report.json')
        report=json.loads(report_bytes)
        classifications=json.loads(z.read('artifacts/h8f-81-exact-construction-classification.json'))
        need(request['schema']=='spin-heart-h8-faster-preparation-v1'
             and request['evidence']['scope']=='H8_RECONSTRUCTED_CAD_ONLY_NOT_NATIVE_CERTIFIED_NOT_SOLVED'
             and report['completed_samples']==81 and report['all_numerical_checks_passed'] is True
             and report['qualified_branch_neighborhood'] is False
             and report['formal_newton_converged'] is False
             and len(report['numeric_checks'])==30 and all(r['numerical_passed'] for r in report['numeric_checks'])
             and report['input_request_sha256']==REQUEST_SHA
             and classifications['source_report_sha256']==sha(report_bytes)
             and classifications['classifications']=={
                 'CONSTRUCTION_FEATURE_CHANGE_OR_COUNT_CHANGE':1,
                 'SYMBOLIC_PATCH_AND_TRIANGLE_IDENTITY_STABLE_NOT_CERTIFIED':79},
             'INVALID_H8F_FAILED_HISTORICAL_PROTOCOL')
        loaded={}
        for i in sample_indices:
            path=f'artifacts/h8f-all81-points/point-{i:03d}'
            point=json.loads(z.read(path+'.json'))
            raw=z.read(path+'-material.json.gz')
            need(sha(raw)==point['exact_same_call_material_archive_sha256'],f'POINT_{i}_ARCHIVE_TAMPER')
            material=json.loads(gzip.decompress(raw))
            need(point['index']==i and point['coefficients']==material['quadric_coefficients']
                 and point['feature_branch_hash']==material['topology_hash']
                 and point['material_geometry_hash']==material['geometry_hash']
                 and point['exact_native_material_boundary_passed'] is True
                 and material['material_proof']['schema']=='cherchi-same-call-material-v1'
                 and str(material['boolean_backend_identity']).startswith(
                     'cherchi-reference-7bd6c2697695f555f9a15d7e154e084d841316f0:'),
                 f'POINT_{i}_NOT_REAL_PINNED_CHERCHI')
            loaded[i]=(point,material)
        return request,report,loaded


def qf(x):return Q.from_float(float(x))


def qvec(x):return tuple(qf(v) for v in x)


def qsub(a,b):return tuple(x-y for x,y in zip(a,b))


def qdot(a,b):return sum((x*y for x,y in zip(a,b)),Q(0))


def qcross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def exact_line_witness(a,b,c,d):
    """Exact Q line closest parameters and oriented separation for binary64 input."""
    u,v,r=qsub(b,a),qsub(d,c),qsub(c,a)
    n=qcross(u,v)
    square=qdot(n,n)
    need(square>0,'PARALLEL_EXACT_INPUT_EDGES')
    s=qdot(qcross(r,v),n)/square
    t=qdot(qcross(r,u),n)/square
    determinant=qdot(qcross(u,r),v)
    closest_distance_squared=determinant*determinant/square
    return {'source_segment_parameter':s,'design_segment_parameter':t,
            'determinant':determinant,'infinite_line_min_squared_distance':closest_distance_squared,
            'cross_squared_norm':square}


def serialize_fraction(f):
    return {'numerator':str(f.numerator),'denominator':str(f.denominator),
            'approximate':float(f)}


def surrogate(q,origin,design,normals,t):
    a=vadd(origin,gauss(q,t,normals[SOURCE_IDS[0]]))
    b=vadd(origin,gauss(q,t,normals[SOURCE_IDS[1]]))
    c,d=design
    u,v,r=vsub(b,a),vsub(d,c),vsub(c,a)
    n=cross(u,v)
    square=dot(n,n)
    need(square.v.lo>0,'SURROGATE_SOURCE_AND_DESIGN_EDGES_PARALLEL')
    s=dot(cross(r,v),n)/square
    w=dot(cross(r,u),n)/square
    determinant=dot(cross(u,r),v)
    return {'source_parameter':s,'design_parameter':w,
            'cross_squared_norm':square,'determinant':determinant,
            'signed_line_separation_squared':determinant*determinant/square}


def seed_normals(q,origin,base):
    point,mat=base
    t=J.of(qf(point['coefficients'][7]))
    c=center(q,t)
    a,b,g=q[4:7]
    normals={}
    for j in SOURCE_IDS:
        src=vec(mat['quadric_source_vertices'][j])
        x=vsub(vsub(src,origin),c)
        normals[j]=[a*x[0],b*x[1]+t*x[2],t*x[1]+g*x[2]]
    return normals


def solve(artifact):
    request,report,archived=archival_data(artifact)
    req=request['request']
    need(req['target_axes']==[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],'ONLY_IDENTITY_TARGET_FRAME')
    base_q=tuple(map(qf,report['target_q']))
    need(base_q[7:]==(Q(0),Q(0),Q(0)) and all(base_q[j]>0 for j in (4,5,6)),
         'NOT_FROZEN_ELLIPSOID_QUADRIC')
    expected_s={(24,89,90),(24,90,151)}
    expected_d={(5516,5517,5518),(5517,5518,5523)}
    need(incident(req['design_faces'],DESIGN_IDS)==expected_d,
         'DESIGN_EDGE_FACET_PROVENANCE_CHANGED')
    base_source=archived[0][1]['quadric_source_faces']
    need(incident(base_source,SOURCE_IDS)==expected_s,'SOURCE_EDGE_FACET_PROVENANCE_CHANGED')
    for i in SAMPLES:
        item,mat=archived[i]
        need(all(qf(item['coefficients'][j])==base_q[j] for j in range(10) if j!=7),
             f'UNREGISTERED_Q_DIRECTION_POINT_{i}')
        need(mat['quadric_source_faces']==base_source,'SOURCE_FACE_CONNECTIVITY_NOT_FIXED')
    # Binary64 source vertices and design-domain coordinates are treated as
    # their exact rational values. This still does NOT bound production motion
    # between samples or authorize a Newton step.
    src_design=tuple(qvec(req['design_vertices'][i]) for i in DESIGN_IDS)
    sample_witnesses={}
    for i in (59,60):
        item,mat=archived[i]
        native_a,native_b=(qvec(mat['quadric_source_vertices'][j]) for j in SOURCE_IDS)
        facts=exact_line_witness(native_a,native_b,*src_design)
        need(0<facts['source_segment_parameter']<1 and 0<facts['design_segment_parameter']<1,
             f'NATIVE_ARCHIVE_{i}_PROJECTED_INTERSECTION_OUTSIDE_EDGES')
        expected_sign=1 if i==59 else -1
        need((facts['determinant']>0)-(facts['determinant']<0)==expected_sign,
             f'ARCHIVED_{i}_REAL_EVENT_SIGN_DRIFT')
        sample_witnesses[str(i)]={
            'exact_native_binary64_material_source':True,
            'historical_c7':item['coefficients'][7],
            'source_parameter':serialize_fraction(facts['source_segment_parameter']),
            'design_parameter':serialize_fraction(facts['design_segment_parameter']),
            'signed_determinant':serialize_fraction(facts['determinant']),
            'infinite_line_min_squared_distance':serialize_fraction(facts['infinite_line_min_squared_distance'])}
    q=vec(base_q)
    origin=vec(req['target_origin'])
    design=[vec(req['design_vertices'][i]) for i in DESIGN_IDS]
    normals=seed_normals(q,origin,archived[0])
    whole=surrogate(q,origin,design,normals,J.variable(LOW,HIGH))
    need(whole['determinant'].d.hi<0,'FULL_BRACKET_SURROGATE_DERIVATIVE_NOT_NEGATIVE')
    need(whole['source_parameter'].v.strictly_unit() and
         whole['design_parameter'].v.strictly_unit(),
         'FULL_BRACKET_SEGMENT_INTERIOR_NOT_CERTIFIED')
    lower,upper=LOW,HIGH
    need(surrogate(q,origin,design,normals,J.of(lower))['determinant'].v.sign()==1 and
         surrogate(q,origin,design,normals,J.of(upper))['determinant'].v.sign()==-1,
         'NO_CONDITIONAL_ANALYTIC_EVENT_SIGN_BRACKET')
    for _ in range(44):
        midpoint=(lower+upper)/2
        orientation=surrogate(q,origin,design,normals,J.of(midpoint))['determinant'].v.sign()
        need(orientation in (1,-1),'SIGN_AT_MIDPOINT_IS_INDETERMINATE')
        if orientation>0:lower=midpoint
        else:upper=midpoint
    root=surrogate(q,origin,design,normals,J.variable(lower,upper))
    need(root['source_parameter'].v.strictly_unit() and
         root['design_parameter'].v.strictly_unit() and
         root['cross_squared_norm'].v.lo>0 and
         root['determinant'].v.lo<=0<=root['determinant'].v.hi,
         'ROOT_EVENT_INTERIOR_OR_COPLANARITY_NOT_CERTIFIED')
    return {
        'schema':'heart-h9-native-endpoints-and-conditional-interior-event-v1',
        'scope':'REAL_NATIVE_BINARY64_FINITE_ENDPOINTS_AND_FIXED_NORMAL_ANALYTIC_SURROGATE_ONLY',
        'historical_run_id':39123384402,'historical_run_conclusion':'failure',
        'artifact_id':12135528290,'artifact_sha256':ZIP_SHA,'request_sha256':REQUEST_SHA,
        'real_native_endpoint_indices':[59,60],
        'verified_material_sample_indices':list(SAMPLES),
        'input_source_edge':list(SOURCE_IDS),'input_design_edge':list(DESIGN_IDS),
        'exact_native_binary64_projected_line_witnesses':sample_witnesses,
        'surrogate_assumptions':['fixed inverse-Gauss normal per source vertex',
                                 'unchanged source triangulation and continuous fixed-normal motion',
                                 'roundoff of unknown in-between native source not strictly bounded'],
        'surrogate_root_exact_interval':[str(lower),str(upper)],
        'surrogate_root_interval_width':float(upper-lower),
        'surrogate_root_midpoint_display':float((lower+upper)/2),
        'full_coefficient_interval':[float(LOW),float(HIGH)],
        'whole_interval_source_parameter_bound':whole['source_parameter'].v.printable(),
        'whole_interval_design_parameter_bound':whole['design_parameter'].v.printable(),
        'whole_interval_cross_squared_bound':whole['cross_squared_norm'].v.printable(),
        'whole_interval_event_derivative_bound':whole['determinant'].d.printable(),
        'at_surrogate_root_source_parameter_bound':root['source_parameter'].v.printable(),
        'at_surrogate_root_design_parameter_bound':root['design_parameter'].v.printable(),
        'at_surrogate_root_cross_squared_bound':root['cross_squared_norm'].v.printable(),
        'at_surrogate_root_det_bound':root['determinant'].v.printable(),
        'conditional_analytic_interior_edge_crossing_certified':True,
        'actual_in_between_native_boolean_samples_evaluated':0,
        'native_production_continuity_or_topology_certified':False,
        'actual_native_cherchi_newton_accepted':False,
        'newton_admitted':False,'final_heart_obj_produced':False,
        'final_independently_integrated_metrics_available':False,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=solve(args.artifact)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    print('H9_REAL_NATIVE_EDGE_ENDPOINTS_AND_SURROGATE_INTERIOR_CROSSING',json.dumps({
        'native_endpoints':result['real_native_endpoint_indices'],
        'root':result['surrogate_root_midpoint_display'],
        'source_param':result['at_surrogate_root_source_parameter_bound'],
        'design_param':result['at_surrogate_root_design_parameter_bound'],
        'new_native_calls':result['actual_in_between_native_boolean_samples_evaluated'],
        'newton_admitted':result['newton_admitted']},sort_keys=True))


if __name__=='__main__':main()
