"""Run the hash-pinned M26 predicates with disjoint outer-face indices and sum the receipts."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

SOURCE_SHA = '2536f7783ff6552ac930c294879e69b8430fc8e527873859b3ec1187d39c6ea0'


def audit(source, geometry, output, shards, compiler, flags):
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA:
        raise ValueError("unexpected predicate source")
    s=original.decode()
    old='std::vector<Elem>possible;rtree.query('
    if s.count(old)!=1:raise ValueError('unexpected traversal')
    # Every worker builds the complete incremental R-tree and runs all topology checks.
    # Only the query on face i is partitioned, so no cross-shard pair is omitted.
    s=s.replace(old,'std::vector<Elem>possible;if(i % shard_count == shard_index) rtree.query(')
    s=s.replace('int max_seconds=120;', '''size_t shard_index=std::stoul(std::getenv("M36_SHARD_INDEX"));
  size_t shard_count=std::stoul(std::getenv("M36_SHARD_COUNT"));
  if (!shard_count || shard_index>=shard_count) throw std::runtime_error("INVALID_SHARD");
  int max_seconds=120;''')
    s=s.replace('status=EXACT_POST_BOOLEAN_SELF_INTERSECTION_FREE','status=EXACT_SHARD_COMPLETE')
    raw=geometry.read_bytes()
    mesh=json.loads(raw); v=mesh['vertices']; f=mesh['faces']
    with tempfile.TemporaryDirectory() as td:
        base=Path(td);cpp=base/'sharded.cpp';cpp.write_text(s)
        binary=base/'checker';packet=base/'mesh.txt'
        with packet.open('w') as handle:
            handle.write(f'{len(v)} {len(f)}\n')
            for point in v:handle.write(' '.join(point)+'\n')
            for face in f:handle.write(' '.join(map(str,face))+'\n')
        subprocess.run([compiler,'-std=c++17','-O2',str(cpp),*flags,'-o',str(binary)],check=True)
        def work(i):
            env=dict(os.environ,M36_SHARD_COUNT=str(shards), M36_SHARD_INDEX=str(i))
            with packet.open() as inp:
                run=subprocess.run([str(binary),'--max-seconds','1800'],env=env,stdin=inp,
                                   capture_output=True,text=True,check=False)
            rec=dict(line.split('=',1) for line in run.stdout.splitlines() if '=' in line)
            if (run.returncode or rec.get('status')!='EXACT_SHARD_COMPLETE' or
                  int(rec['vertices'])!=len(v) or int(rec['faces'])!=len(f) or
                  int(rec['candidate_pairs'])!=int(rec['exact_pairs_checked']) or
                  int(rec['candidate_pairs'])!=int(rec['adjacent_pairs'])+int(rec['nonadjacent_pairs'])):
                raise ValueError(f'failed shard {i}: {rec}')
            return dict(shard=i,**rec)
        with concurrent.futures.ThreadPoolExecutor(max_workers=shards) as pool:
            receipts=list(pool.map(work,range(shards)))
        all_counts={k:sum(int(row[k]) for row in receipts) for k in ('candidate_pairs','exact_pairs_checked','adjacent_pairs','nonadjacent_pairs')}
        report=dict(status='EXACT_POST_BOOLEAN_SELF_INTERSECTION_FREE',vertices=len(v),faces=len(f),
          input_sha256=hashlib.sha256(raw).hexdigest(),m26_sha256=SOURCE_SHA,
          generated_source_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),shards=receipts,**all_counts)
        output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','mesh','output']:p.add_argument(name,type=Path)
    p.add_argument('--shards',type=int,default=6,choices=range(1,33))
    p.add_argument('--compiler',default='g++')
    p.add_argument('--flags',nargs=argparse.REMAINDER,default=['-lgmpxx','-lgmp'])
    a=p.parse_args();audit(a.source,a.mesh,a.output,a.shards,a.compiler,a.flags)
