"""Apply the frozen endpoint formula to five distinct earlier M37 Noodles states."""
import argparse
from pathlib import Path
import json

import numpy as np
from analytic_jacobian import endpoint_jacobian,source_velocities
from native_jv import checked,normals,parse_mesh,Raster,rel,stencil,vertices,DESIGN_SHA

REPORTS={'exact':('194101f1fdb11660f3444235723deca6f1b67e404a2f2938da719dda668e1aef',[1,2,4]),
         'learned':('900d7b864c53df474a0f3f171107950ae2e78f46f3ccdb7f2e66f460bb65f2fb',[3]),
         'unit':('316c75178691df274653ae22efca8305e49931a742ff95185f42db502ad7af16',[3])}

RECEIPT_SHA = {('exact', 1): 'e0fa21f1a0be865edaf160d7e037f1c7bb77c218699a5f91c9c4b04b092b5a33', ('exact', 2): '3c861a93a063742248869e2a7bdb2711788419dbe8b719bebd835beb81c84573', ('exact', 4): 'b8961a2655d4c762000474a65be0aa57d7de806790276e5e319a18575734fb53', ('learned', 3): 'a548c2f13edb606a023cfa5140f804b4b311f65c5b3c71d72040e680c4880b3f', ('unit', 3): '3895b29cfa6e6617ef48d5b7af2cf40149970b0ddef3716e1614b2c99bf700f0'}


def run(runs,design,sizes,out):
    mesh,_=parse_mesh(design)
    states=[]
    for arm,(digest,indices) in REPORTS.items():
        report=checked(runs/arm/'report.json',digest)
        for index in indices:
            base=runs/arm/f'call-{index:03d}'
            row=report['evaluations'][index-1]
            rec=checked(base/'receipt.json',RECEIPT_SHA[arm,index])
            source=checked(base/'source.json',rec['source_sha256'])
            if row['q']!=source['raw_q'] or row['result_sha256']!=rec['result_sha256'] or rec['design_sha256']!=DESIGN_SHA or rec['cap_output_faces']!=0:
                raise ValueError(f'native binding mismatch: {arm}/{index}')
            states.append((arm,index,row,rec,source))
    common=normals(np.array(states[0][2]['q']),np.array(states[0][4]['vertices_tip']))
    result={'schema':'m37-five-state-crosscheck-v1','design_sha256':DESIGN_SHA,
            'scope':'five deduplicated M37 development states on Noodles; ellipsoid sources reconstructed from q; complement hyperboloids evaluated using archived native source vertex positions and their geometric coefficient velocities; native J_tip withheld from estimator','rows':[]}
    for n in sizes:
        rast=Raster(stencil(mesh,n),n)
        for arm,index,row,rec,source in states:
            q=np.array(row['q'])
            geom=np.asarray(source['vertices_tip'])
            vel=np.asarray(source['native_jacobian'])
            if source['surface_type']=='ELLIPSOID':
                reconstructed=vertices(q,common)
                if np.max(abs(reconstructed-geom))>1e-11 or rel(source_velocities(q,common),vel)>1e-8:
                    raise ValueError('independent finite ellipsoid source differs from archived packet')
                geom,vel=reconstructed,source_velocities(q,common)
            mat,active=endpoint_jacobian(rast,geom,np.array(source['faces']),vel)
            if rec['operation']=='difference':
                mat=-mat
            elif rec['operation']!='intersection':
                raise ValueError('unsupported native operation')
            ref=np.array(row['J_tip'])
            record=dict(arm=arm,index=index,n=n,source_type=source['surface_type'],op=rec['operation'],
                        source_sha256=rec['source_sha256'],ref_report_sha256=REPORTS[arm][0],native_cap_output_faces=rec['cap_output_faces'],
                        active_endpoints=active,relative_frobenius=rel(mat,ref),relative_izz_gradient=rel(mat[4]+mat[5],ref[4]+ref[5]),
                        J_estimated=mat.tolist(),J_native=ref.tolist())
            result['rows'].append(record)
            print(n,arm,index,record['relative_frobenius'],record['relative_izz_gradient'],flush=True)
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',required=True,type=Path)
    p.add_argument('--design',required=True,type=Path)
    p.add_argument('--sizes',type=int,nargs='+',default=[512,1024])
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    run(a.runs,a.design,a.sizes,a.output)
