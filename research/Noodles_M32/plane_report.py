"""Bind moment and Jacobian enclosures and finite differences to an M34 operand."""
import argparse
from fractions import Fraction as Q
from hashlib import sha256
import json
from pathlib import Path
import subprocess


def analyze(stdout):
    lines = stdout.splitlines()
    _, bits, nv, nf, full, empty, cut, margin = lines[0].split()
    bits, nv, nf, full, empty, cut = map(int, (bits, nv, nf, full, empty, cut))
    if full+empty+cut != nf or cut < 1 or bits != 160 or Q(margin) <= 0:
        raise ValueError("invalid enclosing-plane accounting")
    rows, jac = {}, {}
    def pairs(tokens):
        if len(tokens)!=20: raise ValueError("invalid tensor row")
        values = [Q(int(x),1<<bits) for x in tokens]
        out=list(zip(values[::2],values[1::2]))
        if any(l>h for l,h in out): raise ValueError("inverted interval")
        return out
    for line in lines[1:]:
        parts=line.split()
        if parts[0]=="r": rows[(int(parts[1]),Q(parts[2]))]=pairs(parts[3:])
        elif parts[0]=="j": jac[int(parts[1])]=pairs(parts[2:])
        else: raise ValueError("unknown result row")
    if len(rows)!=13 or set(jac)!=set(range(10)):raise ValueError("missing evaluations")
    checks=[]
    for col in (0,1,2):
        for e in (20,21):
            h=Q(1,2**e);plus=rows[(col,h)];minus=rows[(col,-h)]
            errors=[]
            for i,((pl,ph),(ml,mh)) in enumerate(zip(plus,minus)):
                jl,jh=jac[i][col]
                lo=(pl-mh)/(2*h)-jh;hi=(ph-ml)/(2*h)-jl
                errors.append(max(abs(lo),abs(hi)))
            maximum=max(errors)
            if maximum>Q(1,10**6):raise ValueError(f"directional check failed: c[{col}], 2^-{e}, {float(maximum)}")
            checks.append(dict(column=col,absolute_step=f"1/{2**e}",worst_component=errors.index(maximum),
                               max_abs_fd_minus_J_upper_exact=str(maximum),max_abs_fd_minus_J_approx=float(maximum)))
    def integers(row):return [[str(int(l*(1<<bits))),str(int(h*(1<<bits)))] for l,h in row]
    base=rows[(0,Q(0))]
    return dict(vertices=nv,faces=nf,fully_retained_source_faces=full,fully_removed_source_faces=empty,
                cut_faces=cut,vertex_sign_patterns_identical_across_all_13_evaluations=True,
                min_abs_base_beta_at_source_vertex_exact=str(Q(margin)),cut_orientation="beta >= 0",
                ambient_native_coefficients=["-3/5","0","0","4/5","0","0","0","0","0","0"],
                moments_relative_to="model origin; variable Dprime only; unit density",
                raw_order=["M","x","y","z","xx","yy","zz","yz","xz","xy"],interval_denominator=f"2^{bits}",
                raw_10_enclosure_numerators=integers(base),
                J_10x10_enclosure_numerators=[integers(jac[i]) for i in range(10)],
                raw_10_approx=[float((l+h)/2) for l,h in base],directional_checks=checks,
                frechet_scope="transverse cut of the continuous plane, ambient native coefficients at this plane",
                smooth_quadric_stencil_columns_checked=[0,1,2],fixed_material_included=False,training_admitted=False,
                native_newton_calls=0,full_quadric_newton_admitted=False)


def measure(operand, expected, binary):
    if sha256(operand.read_bytes()).hexdigest()!=expected:raise ValueError("wrong operand identity")
    cmd=subprocess.run([str(binary),str(operand)],check=True,stdout=subprocess.PIPE,timeout=600)
    return dict(schema="W3_M32_RATIONAL_PLANE_REFERENCE_V1",operand_sha256=expected,
                oriented_integration_output_sha256=sha256(cmd.stdout).hexdigest(),**analyze(cmd.stdout.decode("ascii")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("operand",type=Path)
    p.add_argument("expected_sha256");p.add_argument("binary",type=Path);a=p.parse_args()
    print(json.dumps(measure(a.operand,a.expected_sha256,a.binary),indent=2,sort_keys=True))
