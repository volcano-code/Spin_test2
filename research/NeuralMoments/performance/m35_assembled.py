"""Validate an intermediate whole-object assembly against exact component moments/chains."""
import argparse
import json
from pathlib import Path

from m34_noodles_partition import Q, partition_chain, point_tip, sha, surface_data
from m35_bounded_ten import integrate

FIXED_SHA = 'aecc5cad813acd05bbe210a7bed8c0933d066b9eaa557959077f08c1dafbde97'
FIXED_TEN_SHA = '213e485af9604156e156426a1a302e8bceec4bf23bfd7cbdbc3af1844b2ca6ed'


def rational_inside(raw,off,l,r):
    na,da=(int(k,0) for k in raw.split('/'))
    if da<=0 or l>r:
        raise ValueError('INVALID_RATIONAL_OR_ENCLOSURE')
    lo,hi=l-off,r-off
    return lo.numerator*da<=na*lo.denominator and na*hi.denominator<=hi.numerator*da


def audit(fixed, fixed_ten, clip, clip_ten, clip_w1, assembled):
    f_sha,fv,ff=surface_data(fixed);c_sha,cv,cf=surface_data(clip);u_sha,uv,uf=surface_data(assembled)
    fdata=fixed_ten.read_bytes();cdata=clip_ten.read_bytes()
    cproof=json.loads(clip_w1.read_text())
    if (f_sha!=FIXED_SHA or sha(fdata)!=FIXED_TEN_SHA or not cproof['regularized_intersection_verified']
        or cproof['output_sha256']!=c_sha or cproof['exact_ten_sha256']!=sha(cdata)):
        raise ValueError('OPERAND_OR_SOURCE_MISMATCH')
    nums=integrate(uv,uf)
    rows=fdata.decode().splitlines();off=[Q(int(x.split('/')[0],0),int(x.split('/')[1],0))
                                          for x in cdata.decode().splitlines()]
    if len(rows)!=10 or len(off)!=10 or not all(rational_inside(row,x,l,h) for row,x,(l,h) in zip(rows,off,nums)):
        raise ValueError('WHOLE_TEN_OUTSIDE_INDEPENDENT_ENCLOSURE')
    chain=partition_chain([(fv,ff,1),(cv,cf,1),(uv,uf,-1)])
    if not chain['oriented_boundary_chain_additive']:raise ValueError('WHOLE_BOUNDARY_CHAIN_MISMATCH')
    tip=point_tip([tuple(map(Q,p)) for p in uv])
    if not tip['exact_fixed_tip']:raise ValueError('FIXED_PIVOT_CHANGED')
    return dict(schema='w1-m35-accepted-intermediate-assembly-v1',fixed_sha=f_sha,
        variable_sha=c_sha,union_source_sha=u_sha,faces=len(uf),vertices=len(uv),tip=tip,
        exact_oriented_boundary=chain,all_ten_enclose_exact_component_sums=True,
        total_ten_model=[float((l+r)/2) for l,r in nums],
        total_ten_bounds=[[str(l),str(r)] for l,r in nums],terminal_certified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('fixed','fixed_ten','clip','clip_ten','clip_w1','assembled'):
        p.add_argument(arg,type=Path)
    a=p.parse_args(); print(json.dumps(audit(**vars(a)),indent=2))
