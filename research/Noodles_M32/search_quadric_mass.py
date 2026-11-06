"""Reproduce the floating-point candidate search before independent exact certification."""
import argparse
import ctypes as c
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.optimize import root as solve

from quadric_mass_bounds import D_PRIME, A_TEN, check_hash, enclose_hex, FOUR


def evaluator(tmp, tri):
    src=Path(__file__).with_name("quadric_proposal.cpp")
    out=tmp/"proposal.so"
    subprocess.run(["c++","-std=c++17","-fPIC","-O3","-shared",str(src),"-o",str(out)],check=True)
    data=tmp/"triangles.bin"
    np.asarray(tri,dtype=np.float64).tofile(data)
    lib=c.CDLL(str(out))
    if lib.init(str(data).encode())!=len(tri):raise ValueError("triangle input mismatch")
    lib.cut.argtypes=[c.c_int]+[c.c_double]*6+[np.ctypeslib.ndpointer(dtype="float64",ndim=1,flags="C")]
    def call(n,args):
        ans=np.empty(10)
        lib.cut(n,*args,ans)
        return ans
    return call


def fit(design, whole_ten):
    from fractions import Fraction as Q
    rows=check_hash(design,D_PRIME).splitlines()
    nv,nf=map(int,rows[0].split())
    vertices=np.array([[float(Q(t.decode())) for t in line.split()] for line in rows[1:nv+1]])
    faces=np.array([list(map(int,line.split())) for line in rows[nv+1:]])
    raw=check_hash(whole_ten,A_TEN).splitlines()
    m=np.array([float(enclose_hex(line.decode()).lo) for line in raw])
    height=m[3]/m[0]-.0002
    result={}
    with tempfile.TemporaryDirectory() as td:
        cut=evaluator(Path(td),vertices[faces])
        guess=np.array([.53,0.,0.,0.,0.])
        for kind in ("cylinder","cone"):
            cylinder=kind=="cylinder"
            if not cylinder:guess=np.append(guess,.03)
            trajectory=[]
            for n in ((32,64,128,256,512) if cylinder else (64,128,256,512)):
                def residual(q):
                    removed=cut(n,np.append(q,0.) if cylinder else q)
                    kept=m-removed
                    budget=(.0999 if cylinder else .09985)*(m[4]+m[5])
                    equations=[*kept[list(FOUR)],budget-removed[4]-removed[5]]
                    if not cylinder:equations.append(kept[3]-height*kept[0])
                    return np.asarray(equations)
                ret=solve(residual,guess,options={"eps":1e-10 if cylinder else 1e-9,"xtol":1e-9})
                err=float(np.max(np.abs(residual(ret.x))))
                if not ret.success or err>1e-8:raise ValueError("proposal sweep failed")
                guess=ret.x
                trajectory.append(dict(sides=n,coefficients=guess.tolist(),residual_max=err))
            result[kind]=trajectory
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("design",type=Path);p.add_argument("whole_ten",type=Path)
    a=p.parse_args()
    print(json.dumps(fit(a.design,a.whole_ten),indent=2))
