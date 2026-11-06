"""W2-M19: independently verify authentic Noodles V7 inputs and F/D/A CSG handoff.

Read-only with respect to historical artifacts. Source part evidence never stands
in for the six missing rational CGAL Boolean packets. This module does not
run Cherchi/CGAL, admit derivative J_D, or accept the final optimized OBJ as
frozen teacher input.

Usage:
  python m19_noodles_fda_audit.py --source-bundle W2_M18_Noodles_V7_94_Source_Recovery.zip \\
    --output /new/evidence.json [--csg-dir /actual/historical/boolean-packets]

The six historic packets require identical original bytes and receipt hashes;
newly recomputed packets may not silently impersonate the original artifacts.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from zipfile import ZipFile

SOURCE_MANIFEST_GIT_BLOB = 'f476a8c3e87bd30ef186b24442f641c2602c32ec'
SOURCE_MANIFEST_SHA256 = '67364abba55c74ccd377eef6174b8c96ee04e4af1aadbd7e505466bafabd2a3a'
CSG_RECEIPT_GIT_BLOB = '1e824c72d5ee13cc484bbf60a0be9b79be86d787'
CSG_RECEIPT_SHA256 = '35f74fbdb432ee4eccebf21d7e4f1f07408f63420c659025857f1d83d15abfd0'
HISTORIC_CSG_BACKEND_SHA256 = 'd6803f184047f6172b177b9f436b16f22cdb6a949126d1df88a5d084f3dd9df3'
CORPUS_DIGEST_188 = '6b2da86949e4426986e2460167f1255f0c44d6e708be87bfec17fb78ce4a69c6'
BUNDLE_MANIFEST = 'UpstreamPinned/frozen-parts-v7c/manifest.json'
BUNDLE_CSG_RECEIPT = 'UpstreamPinned/frozen-inputs/author-rational/construction.json'
PARTS_COUNT = 94
FIXED_SOURCE_PARTS = 93
PACKETS = ('design','envelope','fixed','fixed-inside-envelope','fixed-outside-envelope','initial')
PAIR_ORDER = ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1))
MAX_MEMBER_BYTES = 200000


class GateError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise GateError(code)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def json_strict(data: bytes):
    def reject_duplicates(pairs):
        out={}
        for k,v in pairs:
            require(k not in out, 'DUPLICATE_JSON_KEY')
            out[k]=v
        return out
    def reject_number(x):
        raise GateError('UNSUPPORTED_JSON_NUMBER:'+x)
    return json.loads(data, object_pairs_hook=reject_duplicates, parse_constant=reject_number)


def pinned_json(data: bytes, sha256: str, blob: str, code: str):
    require(sha256_bytes(data)==sha256, code+'_SHA256_MISMATCH')
    require(git_blob_sha1(data)==blob, code+'_GIT_BLOB_MISMATCH')
    return json_strict(data)


def zip_member(z: ZipFile, name: str, *, size_limit=MAX_MEMBER_BYTES):
    try:
        info=z.getinfo(name)
    except KeyError as e:
        raise GateError('MISSING_BUNDLE_MEMBER:'+name) from e
    require(info.file_size<=size_limit, 'OVERSIZED_SOURCE_MEMBER:'+name)
    require(not info.is_dir(), 'NOT_SOURCE_FILE:'+name)
    raw=z.read(info)
    require(len(raw)==info.file_size, 'TRUNCATED_SOURCE_FILE:'+name)
    return raw


def read_static_triangles(raw: bytes, name: str):
    """Parse original .17g model/SI OBJ, rejecting recoded/ambiguous triangles."""
    try:
        text=raw.decode('utf-8-sig')
    except UnicodeError as exc:
        raise GateError('OBJ_INVALID_UTF8:'+name) from exc
    vertices=[];faces=[];began_faces=False
    for lineno,line in enumerate(text.splitlines(),1):
        line=line.split('#',1)[0].strip()
        if not line:continue
        pieces=line.split()
        if pieces[0]=='v':
            require(not began_faces and len(pieces)==4,'OBJ_VERTEX_ORDER_OR_ARITY:'+name)
            try: xyz=tuple(float(k) for k in pieces[1:])
            except ValueError as exc: raise GateError('OBJ_BAD_VERTEX:'+name) from exc
            require(all(math.isfinite(k) for k in xyz),'OBJ_NONFINITE_VERTEX:'+name)
            vertices.append(xyz)
        elif pieces[0]=='f':
            began_faces=True
            require(len(pieces)==4 and all(t.isdecimal() for t in pieces[1:]),
                    'OBJ_NOT_ABSOLUTE_TRIANGLE:'+name)
            face=tuple(int(t)-1 for t in pieces[1:])
            require(min(face)>=0 and max(face)<len(vertices) and len(set(face))==3,
                    'OBJ_INVALID_INDEX:'+name)
            faces.append(face)
        else:
            raise GateError('OBJ_UNKNOWN_RECORD:'+name+':'+str(lineno))
    require(len(vertices)>3 and len(faces)>3,'OBJ_EMPTY:'+name)
    return vertices,faces


def oriented_topology(vertices, faces):
    """Necessary directed-edge manifold closure; NOT self-intersection proof."""
    edges=Counter()
    for i,j,k in faces:
        edges[i,j]+=1;edges[j,k]+=1;edges[k,i]+=1
    require(all(n==1 and edges[(v,u)]==1 for (u,v),n in edges.items()),
            'ORIENTED_TWO_MANIFOLD_EDGE_PAIRING_FAILED')
    return len(edges)//2


def exact_signed_ten(vertices, faces, *, target_origin=(0.,0.,0.)):
    """Independent exact binary64→Fraction signed-tetra raw mass moments.

    Fraction.from_float preserves actual source binary64 coordinates; an input
    object's primitive moments cannot simply be summed to produce F because
    all 93 fixed components overlap. No Boolean/embedding proof is claimed.
    """
    require(len(target_origin)==3 and all(math.isfinite(float(q)) for q in target_origin),
            'INVALID_EXACT_TARGET')
    origin=tuple(Fraction.from_float(float(o)) for o in target_origin)
    xyz=[tuple(Fraction.from_float(float(v))-origin[i] for i,v in enumerate(row))
         for row in vertices]
    return fraction_signed_ten(xyz, faces)


def fraction_signed_ten(xyz, faces):
    """Integrate the oriented raw moments of coordinates already converted to Fraction."""
    m=[Fraction(0) for _ in range(10)]
    for ai,bi,ci in faces:
        a,b,c=xyz[ai],xyz[bi],xyz[ci]
        det=(a[0]*(b[1]*c[2]-b[2]*c[1])+
             a[1]*(b[2]*c[0]-b[0]*c[2])+
             a[2]*(b[0]*c[1]-b[1]*c[0]))
        vol=det/6
        m[0]+=vol
        su=[a[k]+b[k]+c[k] for k in range(3)]
        for k in range(3):m[1+k]+=vol*su[k]/4
        for j,(u,v) in enumerate(PAIR_ORDER):
            m[4+j]+=vol*(su[u]*su[v]+a[u]*a[v]+b[u]*b[v]+c[u]*c[v])/20
    require(m[0]>0,'NONPOSITIVE_SOURCE_EXACT_VOLUME')
    return m


def rational_ten_digest(tens):
    require(len(tens)==10 and all(isinstance(x,Fraction) for x in tens),
            'EXACT_TEN_FRACTION_REQUIRED')
    values=[f'{x.numerator}/{x.denominator}' for x in tens]
    return sha256_bytes(json.dumps(values,separators=(',',':'),allow_nan=False).encode())


def _digest_corpus(parts):
    # Byte-for-byte M18 frozen tab/newline corpus encoding (not JSON hashing).
    payload=''.join(f'{name}\t{parts[name]["sha256_model"]}\t{parts[name]["sha256_si"]}\t'
                    f'{parts[name]["audit"]["vertices"]}\t{parts[name]["audit"]["triangles"]}\n'
                    for name in sorted(parts))
    return sha256_bytes(payload.encode('ascii'))


def audit_source_bundle(bundle, *, exact_all_parts=True):
    """Verify the actual recovered 188 historical originals, never report CSG."""
    begin=time.monotonic()
    path=Path(bundle)
    require(path.is_file() and not path.is_symlink(), 'M18_SOURCE_ARCHIVE_MISSING')
    with ZipFile(path) as z:
        names=z.namelist()
        require(len(names)==len(set(names)), 'DUPLICATE_ZIP_MEMBER')
        manifest=pinned_json(zip_member(z,BUNDLE_MANIFEST),
                             SOURCE_MANIFEST_SHA256,SOURCE_MANIFEST_GIT_BLOB,'ORIGINAL_MANIFEST')
        receipt=pinned_json(zip_member(z,BUNDLE_CSG_RECEIPT),
                            CSG_RECEIPT_SHA256,CSG_RECEIPT_GIT_BLOB,'ORIGINAL_CSG_RECEIPT')
        parts=manifest['parts']; fixed_names=manifest['construction']['fixed_parts']
        require(len(parts)==PARTS_COUNT and len(fixed_names)==FIXED_SOURCE_PARTS and
                set(fixed_names)==set(parts)-{'outer-ellipsoid'} and
                len(set(fixed_names))==FIXED_SOURCE_PARTS,
                'FROZEN_94_PART_AND_93_FIXED_RECIPE_DRIFT')
        require(manifest['status']=='SOURCE_PARTS_ONLY_EXACT_BOOLEAN_AND_SOLVER_NOT_RUN' and
                manifest['spec']['author_original'] is False and
                manifest['construction']['fixed']=='EXACT_UNION(ALL fixed_parts)' and
                manifest['construction']['variable']=='EXACT_DIFFERENCE(outer-ellipsoid, fixed)' and
                manifest['construction']['initial']=='EXACT_UNION(outer-ellipsoid, fixed)',
                'NOODLES_PHYSICAL_DOMAIN_RECIPE_CHANGED')
        require(_digest_corpus(parts)==CORPUS_DIGEST_188,
                'FROZEN_188_CORPUS_IDENTITY_CHANGED')
        require(receipt['manifest_sha']==SOURCE_MANIFEST_SHA256 and
                receipt['backend_sha']==HISTORIC_CSG_BACKEND_SHA256 and
                receipt['csg_executed_current_process']==96 and
                receipt['independent_partition_verified'] is False and
                set(receipt['outputs'])==set(PACKETS),
                'UNAUTHENTICATED_HISTORIC_CSG_RECEIPT')
        pairs=0;tot_faces=0;independent_moments={};max_error=0.0;global_source_sha=[]
        target=manifest['spec']['target_origin_model']
        require(target==[0.0,0.0,-.16], 'FROZEN_TARGET_DRIFT')
        for name,p in parts.items():
            require(p['model_obj']==name+'-model.obj' and
                    p['si_obj']==name+'-si.obj', 'SOURCE_FILENAME_DRIFT:'+name)
            mr=zip_member(z,'OBJ/'+p['model_obj']);sr=zip_member(z,'OBJ/'+p['si_obj'])
            require(sha256_bytes(mr)==p['sha256_model'] and sha256_bytes(sr)==p['sha256_si'],
                    'FROZEN_SOURCE_SHA_DRIFT:'+name)
            v,f=read_static_triangles(mr,name+'-model.obj')
            sv,sf=read_static_triangles(sr,name+'-si.obj')
            require((len(v),len(f))==(p['audit']['vertices'],p['audit']['triangles']) and
                    sf==f and len(sv)==len(v), 'FROZEN_SOURCE_TOPOLOGY_OR_COUNT_DRIFT:'+name)
            require(all(tuple(float(x)*.025 for x in point)==observed
                        for point,observed in zip(v,sv)),
                    'SOURCE_MODEL_SI_BINARY64_SCALE_MISMATCH:'+name)
            oriented_topology(v,f)
            require(p['audit']['closed'] is True and
                    p['audit']['winding_consistent'] is True and
                    p['audit']['components']==1,
                    'ARCHIVED_SOURCE_AUDIT_NO_SINGLE_CLOSED_PART:'+name)
            tot_faces+=len(f);pairs+=2
            global_source_sha.append(p['sha256_model']+p['sha256_si'])
            if exact_all_parts or name=='outer-ellipsoid':
                # This is exact for the input binary64 mesh, not the unseen CGAL output.
                exact=exact_signed_ten(v,f,target_origin=target)
                relative=abs(float(exact[0])-p['audit']['signed_volume_model3'])
                require(relative<=1e-12,'SOURCE_EXACT_VOLUME_VS_FROZEN_REFERENCE:'+name)
                max_error=max(max_error,relative)
                independent_moments[name]={
                   'volume_model3':float(exact[0]),
                   'r10_normalized_target_frame':[float(x) for x in exact],
                   'source_exact_rational_integral':True,
                   'exact_rational_ten_sha256':rational_ten_digest(exact),
                   'is_union_of_fixed_components':False,
                }
        require(pairs==188 and len(global_source_sha)==94,'INCOMPLETE_188_SOURCE_SET')
        return {
         'schema':'w2-m19-noodles-94-exact-source-moments-v1',
         'status':'REAL_ORIGINAL_94_SOURCES_SHA_AND_EXACT_TEN_MOMENTS_VERIFIED_NO_CSG',
         'scope':'BINARY64_TRIANGULATED_INPUTS_NOT_AUTHOR_CAD_NOT_BOOLEAN_SOLIDS',
         'bundle_sha256':sha256_bytes(path.read_bytes()),
         'original_source_manifest_sha256':SOURCE_MANIFEST_SHA256,
         'original_manifest_git_blob':SOURCE_MANIFEST_GIT_BLOB,
         'original_CSG_receipt_sha256':CSG_RECEIPT_SHA256,
         'source_corpus_188_digest':CORPUS_DIGEST_188,
         'original_fixed_parts_count':len(fixed_names),
         'source_parts_identity_checked':PARTS_COUNT,
         'original_obj_sha256_verified':pairs,
         'source_triangles_total':tot_faces,
         'independently_reintegrated_source_meshes':len(independent_moments),
         'independently_reintegrated_ten_moment_components':len(independent_moments)*10,
         'max_volume_error_relative_to_frozen_f64_source_audit':max_error,
         'source_moment_snapshots':independent_moments,
         'exact_rational_all_source_ten_digest':sha256_bytes(json.dumps(
             [[name,independent_moments[name]['exact_rational_ten_sha256']]
              for name in sorted(independent_moments)],
             separators=(',',':'),allow_nan=False).encode()),
         'source_mass_moments_must_not_be_summed_to_fixed_F':True,
         'original_historic_exact_CSG_calls':96,
         'new_exact_CSG_calls':0,
         'historical_CSG_independent_partition_certified':False,
         'new_Cherchi_calls':0,
         'J_D':None,'training_admitted':0,
         'wall_seconds':time.monotonic()-begin,
        }


def historical_csg_gate(directory, receipt, *, expected_original_source_sha=SOURCE_MANIFEST_SHA256):
    """Only an original byte-exact 6-packet handoff; never certifies solid F/D/A.

    Matching historical SHA is necessary for importing six old CSG packets,
    NOT sufficient for independently verifying their Boolean semantics, legal
    volume partition, or a smooth Jacobian neighborhood.
    """
    r=pinned_json(Path(receipt).read_bytes(),CSG_RECEIPT_SHA256,
                  CSG_RECEIPT_GIT_BLOB,'HISTORIC_CSG_RECEIPT')
    require(r['manifest_sha']==expected_original_source_sha and
            r['backend_sha']==HISTORIC_CSG_BACKEND_SHA256 and
            r['csg_executed_current_process']==96,
            'HISTORIC_CSG_RUN_IDENTITY_MISMATCH')
    root=Path(directory);missing=[];matched=[]
    require(root.is_dir(),'CSG_PACKET_DIRECTORY_NOT_FOUND')
    for name in PACKETS:
        record=r['outputs'][name]
        require(int(record['faces'])>0 and record['path'].endswith('/'+name+'-rational.json'),
                'HISTORIC_CSG_PACKET_PATH_OR_FACE_COUNT_CHANGED:'+name)
        packet=root/(name+'-rational.json')
        if not packet.is_file():missing.append(name);continue
        require(not packet.is_symlink(), 'CSG_PACKET_SYMLINK_FORBIDDEN:'+name)
        require(sha256_bytes(packet.read_bytes())==record['sha'],
                'HISTORIC_CSG_PACKET_SHA_CHANGED:'+name)
        matched.append(name)
    return {'historical_csg_packets_original_sha_match':matched,
            'historical_csg_packets_missing':missing,
            'all_original_six_packets_present':len(matched)==6,
            'historical_CSG_independent_partition_certified':False,
            'new_Cherchi_calls':0,
            'training_admitted':0}


def _atomic_exclusive_json(path, record):
    path=Path(path);require(not os.path.lexists(path),'REFUSE_OVERWRITE_EXISTING_EVIDENCE')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
             prefix='.'+path.name+'.',delete=False) as f:
            tmp=Path(f.name);json.dump(record,f,sort_keys=True,indent=2,allow_nan=False)
            f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
        dfd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(dfd)
        finally:os.close(dfd)
    finally:
        if tmp is not None:tmp.unlink(missing_ok=True)


def run(bundle,output,*,csg_dir=None,exact_all_parts=True):
    result=audit_source_bundle(bundle,exact_all_parts=exact_all_parts)
    with ZipFile(bundle) as z:
        receipt=zip_member(z,BUNDLE_CSG_RECEIPT)
    if csg_dir is None:
        csg={'historical_csg_packets_original_sha_match':[],
             'historical_csg_packets_missing':list(PACKETS),
             'all_original_six_packets_present':False,
             'historical_CSG_independent_partition_certified':False,
             'new_Cherchi_calls':0,'training_admitted':0}
    else:
        # Original receipt bytes must be staged separately and match Git blob.
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'construction.json';p.write_bytes(receipt)
            csg=historical_csg_gate(csg_dir,p)
    result['historic_CSG_packet_evidence']=csg
    result['status']=('SOURCE_MOMENTS_VERIFIED_SIX_ORIGINAL_CSG_PACKETS_IMPORTABLE_BUT_UNCERTIFIED'
                      if csg['all_original_six_packets_present'] else
                      'SOURCE_MOMENTS_VERIFIED_CSG_PACKETS_MISSING_NATIVE_TEACHER_BLOCKED')
    result['native_teacher_request_admitted']=False
    _atomic_exclusive_json(output,result)
    return result


def main(args=None):
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--source-bundle',required=True)
    a.add_argument('--csg-dir')
    a.add_argument('--output',required=True)
    a.add_argument('--envelope-only',action='store_true',
                   help='Quick preflight: one independently exact integrated envelope, not all 94')
    opts=a.parse_args(args)
    try:
        result=run(opts.source_bundle,opts.output,csg_dir=opts.csg_dir,
                   exact_all_parts=not opts.envelope_only)
        print(json.dumps({'status':result['status'],
          'source_parts':result['source_parts_identity_checked'],
          'exact_source_ten_moment_components':result['independently_reintegrated_ten_moment_components'],
          'historical_csg_packets_imported':len(result['historic_CSG_packet_evidence']['historical_csg_packets_original_sha_match']),
          'new_cherchi_calls':0,'training_admitted':0},sort_keys=True))
        return 2 if not result['native_teacher_request_admitted'] else 0
    except (GateError,FileNotFoundError,ValueError) as exc:
        print(json.dumps({'status':'REJECTED_BEFORE_TEACHER','error':str(exc),
                          'new_cherchi_calls':0,'training_admitted':0}),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
