"""Compute boundary genera for a finite closed orientable embedded triangular 3D design."""
import json,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1]); raw=p.read_bytes();mesh=json.loads(raw)
f=mesh['faces'];v=mesh['vertices'];p=list(range(len(v)))
def root(i):
 while i!=p[i]:p[i]=p[p[i]];i=p[i]
 return i
for a,b,c in f:
 i,j,k=map(root,(a,b,c));p[k]=i;p[j]=i
from collections import Counter
vs=Counter(root(i) for i in range(len(v)));fs=Counter(root(face[0]) for face in f)
out=[]
for i,n in vs.items():
 nf=fs[i];ne=3*nf//2;chi=n-ne+nf
 assert (2-chi)%2==0
 out.append(dict(vertices=n,faces=nf,edges=ne,chi=chi,boundary_genus=(2-chi)//2))
print(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),components=sorted(out,key=lambda r:-r['faces']),
  sum_boundary_genera=sum(r['boundary_genus'] for r in out)),indent=2))
