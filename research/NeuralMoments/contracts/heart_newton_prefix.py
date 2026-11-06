"""Offline replay of a bounded original Faster solve on the frozen Heart request."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .heart_c0_column import get_json, POINT0_REQUEST, POINT0_SETTINGS
from .native_teacher_replay import affine_target_from_physical, frac
from .real_h8e_archive_m8 import (canonical_digest as digest, check_closed_indexed_surface,
    exact_rational_ten_moments, read_json, require, target_fixed_moments)

ELF = {"make_it_stand._make_it_stand":"e4ed1f1fe2a37ed6effcfb4626ba2dbc0f042a8be6236d1fbcb93bf17e78082a",
       "_cherchi_core_boolean":"792f9419615078df06f48e5570de8773beed8008802013f3d2b35029a53bce4d",
       "_cherchi_rational_moments":"ed05436e2796b190ddebb024f67318e2e88bdc46d96b01e9d7db71a6af080dca"}
TRIAL_REASONS = {"residual_decreased","topology_transition_relinearized"}
REQUEST_FILE = "1d2b06063c5b2c7010fcb2764518c05c9707f013d6914bc4e20298f8516e061c"
TOL = 1e-9


def near(actual, expected, name, tol=TOL):
    a,b = np.asarray(actual,dtype=float), np.asarray(expected,dtype=float)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and
            np.all(np.abs(a-b) <= tol*np.maximum(1.,np.abs(b))), name+" mismatch")


def deterministic_starts(seed, count):
    """Replay the frozen uint64 SplitMix sequence and spinning-top c[6] rule."""
    mask=(1<<64)-1
    state=seed ^ 0x5350494E49544641
    for _ in range(count):
        q=[]
        for _ in range(10):
            state=(state+0x9e3779b97f4a7c15)&mask
            v=((state^(state>>30))*0xbf58476d1ce4e5b9)&mask
            v=((v^(v>>27))*0x94d049bb133111eb)&mask
            q.append(2*((v^(v>>31))>>11)/2**53-1.)
        q[6]=-abs(q[6])-.1
        yield np.asarray(q)/np.linalg.norm(q)


def physics(r):
    m,x,y,z,xx,yy,zz,yz,xz,xy=r
    a,b,c=yy+zz-(y*y+z*z)/m,xx+zz-(x*x+z*z)/m,xx+yy-(x*x+y*y)/m
    d=-xy+x*y/m
    return np.asarray([(a*a+b*b+2*d*d)/(c*c),x,y,-xz+x*z/m,-yz+y*z/m])


def state(q, r):
    r=np.asarray(r,dtype=float); p=np.asarray(q,dtype=float).copy()
    require(p.shape==r.shape==(10,) and r[0]>0, 'invalid state')
    p[7:]*=2
    p/=np.linalg.norm(p)
    value=physics(r)
    D=np.column_stack([physics(r+1e-24j*e).imag/1e-24 for e in np.eye(10)])
    g,G=D[0],D[1:]; cons=value[1:]
    M=np.column_stack((-G.T,p))
    lm,_,rank,_=np.linalg.lstsq(M,-g,rcond=None)
    opt=g+M@lm
    raw=float(np.linalg.norm(np.r_[opt,cons]))
    phy=float(np.linalg.norm(cons)/max(1,np.linalg.norm(r)))
    return dict(objective=float(value[0]), constraints=cons.tolist(), constraint_jacobian=G,
                lambda_=lm[:-1].tolist(), multiplier_mu=float(lm[-1]), projection_rank=int(rank),
                solver_coordinates=p, optimality=opt, raw_equation_residual=raw,
                normalized_equation_residual=raw/max(1,np.linalg.norm(g),np.linalg.norm(G)),
                physical_relative_residual=phy)


def exact_endpoint(m, request):
    v=m['variable_rational_vertices'];fs=m['variable_faces']
    source=m['material_proof']['exact_vertices']
    require(digest(dict(vertices=source,faces=fs))==m['geometry_hash'],'saved geometry hash')
    cutter=dict(vertices=m['quadric_source_vertices'],faces=m['quadric_source_faces'],coefficients=m['quadric_coefficients'])
    require(digest(cutter)==m['quadric_source_identity'], 'saved source hash')
    transformed=affine_target_from_physical((tuple(map(frac,row)) for row in source),
        origin=(0,0,0),length=1,target_origin=request['target_origin'],target_axes=request['target_axes'])
    require(transformed==tuple(tuple(map(F,row)) for row in v), 'saved target transform')
    if v or fs:
        check_closed_indexed_surface(fs,len(v))
        moments=exact_rational_ten_moments(v,fs,request['material_density'])
    else:
        moments=[F(0)]*10
    stored=m['variable_raw_moments']
    err=max(abs(x-F(y)) for x,y in zip(moments,stored))
    require(len(stored)==10 and err <= F(1e-10), 'exact endpoint moment mismatch')
    return dict(vertices=len(v), faces=len(fs), moment_hash=digest(list(map(str,moments))),
                max_abs_exact_error=float(err), variable_mass=float(moments[0]), empty=(not fs))


def lagrangian_gradient(r, lam):
    """Closed first derivatives are complex-safe for an independent Hessian."""
    basis=np.eye(10,dtype=r.dtype)
    def central(k,i,j):
        return (r[k]-r[i]*r[j]/r[0],
                basis[k]-(r[i]*basis[j]+r[j]*basis[i])/r[0]+r[i]*r[j]*basis[0]/r[0]**2)
    xx,dx=central(4,1,1); yy,dy=central(5,2,2); zz,dz=central(6,3,3)
    xy,dxy=central(9,1,2); _,dxz=central(8,1,3); _,dyz=central(7,2,3)
    a,b,c=yy+zz,xx+zz,xx+yy
    df=(2*a*(dy+dz)+2*b*(dx+dz)+4*xy*dxy)/c**2 - 2*(a*a+b*b+2*xy*xy)*(dx+dy)/c**3
    return df-lam[0]*basis[1]-lam[1]*basis[2]+lam[2]*dxz+lam[3]*dyz


def check_system(previous,payload):
    p=previous['solver_coordinates']; A=np.asarray(payload['matrix']); b=np.asarray(payload['right_hand_side'])
    require(A.shape==(15,15) and b.shape==(15,), 'invalid full-space system')
    r=np.asarray(previous['raw_moments']); lam=previous['lambda_']
    H=np.column_stack([lagrangian_gradient(r+1e-24j*axis,lam).imag/1e-24 for axis in np.eye(10)])
    near(A[:10,:10],H@previous['solver_jacobian']+previous['multiplier_mu']*np.eye(10),'objective Hessian block')
    near(A[10:,10:],np.zeros((5,5)),'system zero block')
    near(b,np.r_[-previous['optimality'],-np.asarray(previous['constraints']),0], 'system rhs')
    near(A[:10,10:14],-previous['constraint_jacobian'].T,'multiplier block')
    near(A[:10,14],p,'direction normalization column');near(A[14,:10],2*p,'normalization row')
    near(A[10:14,:10],previous['constraint_jacobian']@previous['solver_jacobian'],'constraint block')
    near(payload['constraint_multipliers'],previous['lambda_'],'system lambda')
    near(payload['multiplier_mu'],previous['multiplier_mu'],'system mu')
    return A,b


def geodesic(previous, q, angle, payload):
    require(0<angle<=.5,'invalid full-space step')
    p=previous['solver_coordinates']; A,b=check_system(previous,payload)
    direction=np.linalg.solve(A,b)[:10]
    direction-=p*(p@direction)
    target=np.cos(angle)*p+np.sin(angle)*direction/np.linalg.norm(direction)
    target[7:]*=.5
    predicted=target/np.linalg.norm(target)
    near(q,predicted,'independent geodesic from captured full system')
    return dict(angle=float(angle), geodesic_max_abs_error=float(np.max(np.abs(q-predicted))),
                captured_system_condition=float(np.linalg.cond(A)))


def moment_reintegration_indices(trajectory, steps, accepted):
    if trajectory:
        return set(accepted)
    endpoints=set(steps[0]) if steps else set()
    if accepted:
        endpoints.update((accepted[0],accepted[-1]))
    return endpoints


def audit(folder, run_id, public_result_sha, *, trajectory=False, first_step_reference=None):
    def file_hash(name):
        with (folder/name).open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    result=get_json(folder/'result.json')
    require(file_hash('result.json')==public_result_sha and result['run_id']==run_id and
            result['schema']==('w2-m30-trajectory-v1' if trajectory else 'w2-m29-first-newton-step-v1'),'unbound run result')
    names={'request.json','module-manifest.json','resources.json','events.jsonl','evaluations.jsonl'}
    require(set(result['file_sha256'])==names and all(not (folder/n).is_symlink() and file_hash(n)==v
            for n,v in result['file_sha256'].items()),'saved archive file changed')
    resources=get_json(folder/'resources.json')
    pre,post=resources['before'],resources['after']
    require(all(r['memory_max']==9*2**30 and r['memory_high']==7*2**30 and
                r['host_swap']==r['swap_max']==0 for r in (pre,post)) and
            all(post['memory_events'][k]==pre['memory_events'][k] for k in ('max','oom','oom_kill')),
            'resource scope changed')
    req=get_json(folder/'request.json'); request=req['request']; settings=request['settings']
    require(result['file_sha256']['request.json']==REQUEST_FILE and digest(request)==POINT0_REQUEST
            and digest(req['evaluator_settings'])==POINT0_SETTINGS and result['explicit_seeds']==0
            and result['module_sha256']==ELF and {k:v['sha256'] for k,v in
                 get_json(folder/'module-manifest.json')['extensions'].items()}==ELF,'frozen contract drift')
    with (folder/'events.jsonl').open('rb') as f:
        events=[read_json(line) for line in f if line.strip()]
    ev=[e for e in events if e['type']=='evaluation']
    starts=[e for e in events if e['type']=='attempt_start']
    seeds=list(deterministic_starts(settings['deterministic_seed'],settings['maximum_restarts']))
    require(0<len(starts)<=len(seeds) and [s['restart'] for s in starts]==list(range(len(starts))),
            'unexpected default starts')
    for seed,start in zip(seeds,starts):
        near(start['quadric_coefficients'],seed,'deterministic initial coefficients',5e-15)
    require(len(ev)==result['evaluation_count'] and [e['evaluation_index'] for e in ev]==list(range(1,len(ev)+1)),
            'incomplete evaluation journal')
    # The trajectory protocol requires exact reintegration at every accepted state.
    previous={}; steps=[]; accepted_states=[]
    for event in ev:
        if event['outcome']!='accepted': continue
        i=event['evaluation_index']; restart=event['restart']
        if event['iteration']>0:
            require(restart in previous and event['reason'] in TRIAL_REASONS, 'unexpected accepted trial')
            steps.append((previous[restart],i))
        previous[restart]=i
        accepted_states.append(i)
    if trajectory:
        require(result['accepted_evaluation_indices']==[b for _,b in steps] and
                result['status'] in ('NATIVE_TERMINAL','BUDGET_EXHAUSTED','TIME_BUDGET_EXHAUSTED') and
                result['maximum_capture_evaluations']==60 and len(ev)<=60 and result['soft_seconds']==690.
                and (result['status']!='BUDGET_EXHAUSTED' or len(ev)==60)
                and (result['status']!='TIME_BUDGET_EXHAUSTED' or result['elapsed_capture_seconds']>=690.),
                'trajectory receipt mismatch')
    else:
        expected_index=steps[0][1] if steps else None
        require(len(steps)<=1 and result['accepted_evaluation_index']==expected_index and
                (result['status']=='FIRST_TRIAL_ACCEPTED')==bool(steps),'acceptance receipt mismatch')
    endpoints=moment_reintegration_indices(trajectory,steps,accepted_states)
    fixed=target_fixed_moments(request['fixed_shell_moments'],request['target_origin'],request['target_axes'])
    summaries=[]; states={};counts={'source':0,'boolean':0,'integration':0,'derivative':0}
    with (folder/'evaluations.jsonl').open('rb') as stream:
        for e in ev:
            row=read_json(stream.readline());i=e['evaluation_index']
            require(row['index']==i and e['quadric_coefficients']==row['native_c'], 'callback/event pairing')
            require(row['call_counts_before']==counts, 'native counter chain')
            counts=row['call_counts_after']
            summary=dict(index=i,restart=e['restart'],iteration=e['iteration'],step=e['step'],
                         outcome=e['outcome'],reason=e['reason'],material_returned=(row['status']=='returned'))
            summaries.append(summary)
            if e['iteration']==0:
                near(row['native_c'],seeds[e['restart']],'initial callback coefficients',5e-15)
            if row['status']=='exception':
                require(e['outcome']=='failed', 'failed callback marked accepted')
                summary['error']=row['error']
                continue
            require(row['status']=='returned','unknown callback state')
            mat=row['material']; mp=row['map']; q=mat['quadric_coefficients']
            require(digest(mat)==row['material_sha256'] and q==row['native_c'] and
                    all(mat[k]==mp[k] for k in ('raw_moments','variable_raw_moments','moment_jacobian',
                        'evaluation_map_identity','boolean_backend_identity','geometry_hash','topology_hash','derivative_defined'))
                    and mat['boolean_backend_identity']=='cherchi-reference-7bd6c2697695f555f9a15d7e154e084d841316f0:'+ELF['_cherchi_core_boolean']
                    and mat['rational_arithmetic_identity']['module_sha256']==ELF['_cherchi_rational_moments']
                    and mat['evaluation_map_identity']=='e8347819b347e54caacaadeb458afdea555615eae99491b19b8eba2b8aac42a7'
                    and counts['boolean']==row['call_counts_before'].get('boolean',0)+1,'unbound returned material')
            near(mat['raw_moments'],fixed+mat['variable_raw_moments'],'fixed plus variable',1e-13)
            summary.update(material_sha256=row['material_sha256'],geometry_hash=mat['geometry_hash'],
                           topology_hash=mat['topology_hash'])
            payload=e.get('evaluation_payload')
            require(payload is not None and payload['evaluation_map_identity']==mp['evaluation_map_identity'] and
                    payload['geometry_hash']==mp['geometry_hash'] and payload['topology_hash']==mp['topology_hash'] and
                    payload['solver_raw_moments']==mp['raw_moments'] and
                    payload['variable_raw_moments']==mp['variable_raw_moments'], 'solver payload/map mismatch')
            if e.get('raw_equation_residual') is None:
                require(e['outcome']=='failed', 'missing accepted-state residual')
                summary['detail']=e['detail']
                continue
            st=state(q,mp['raw_moments']); st['raw_moments']=mp['raw_moments']; states[i]=st
            for key in ('raw_equation_residual','normalized_equation_residual','physical_relative_residual','objective','multiplier_mu'):
                near(st[key],e[key],key)
                summary[key]=st[key]
            near(payload['solver_coordinates'],st['solver_coordinates'],'dual coordinates')
            p=st['solver_coordinates']; inv=np.diag([1.]*7+[.5]*3); raw=np.asarray(q)
            dcdp=(np.eye(10)-np.outer(raw,raw))@inv/np.linalg.norm(inv@p)
            st['solver_jacobian']=np.asarray(mp['moment_jacobian'])@dcdp
            near(payload['solver_moment_jacobian'],st['solver_jacobian'],'solver Jacobian')
            summary.update(constraints=st['constraints'],multiplier_lambda=st['lambda_'],projection_rank=st['projection_rank'])
            if e['iteration']==0:
                require((e['outcome']=='accepted')==(st['multiplier_mu']>0), 'initial multiplier decision')
            if trajectory:
                total=np.asarray(mp['raw_moments'])
                summary.update(total_mass=float(total[0]), variable_mass=float(mat['variable_raw_moments'][0]),
                    center_of_mass_axis_offset=float(np.linalg.norm(total[1:3])/total[0]),
                    axial_inertia_about_target_axis=float(total[4]+total[5]),
                    axial_inertia_about_parallel_centroidal_axis=float(total[4]+total[5]-(total[1:3]@total[1:3])/total[0]))
            if i in endpoints:
                summary['exact_material']=exact_endpoint(mat,request)
        require(not stream.readline() and counts==result['native_stage_counts']
                and result['native_boolean_calls']==counts.get('boolean',0), 'incomplete native counts')
    verified_systems=[]
    for e in events:
        if e.get('newton_system_payload') is not None:
            # Backtracking can leave the journal index at the latest rejected trial.
            applied = next(v for v in reversed(ev) if v['outcome']=='accepted' and
                           v['restart']==e['restart'] and v['evaluation_index']<=e['evaluation_index'])
            idx=applied['evaluation_index']
            near(e['quadric_coefficients'],applied['quadric_coefficients'],'Newton linearization point')
            check_system(states[idx],e['newton_system_payload'])
            verified_systems.append(dict(state_evaluation=idx,restart=e['restart'],iteration=e['iteration'],
                                         event_type=e['type'],reason=e['reason']))
    verified_steps=[]
    for before,after in steps:
        x,y=summaries[before-1],summaries[after-1]
        systems=[t for t in events if t['type']=='newton_linear_solve' and t['restart']==y['restart']
                 and t['iteration']==y['iteration'] and t['evaluation_index']<after]
        require(len(systems)==1,'missing unique solve for accepted trial')
        step=geodesic(states[before],np.asarray(ev[after-1]['quadric_coefficients']),y['step'],
                      systems[0]['newton_system_payload'])
        drop=x['raw_equation_residual']-y['raw_equation_residual']
        changed=x['topology_hash']!=y['topology_hash']
        require(drop>0 and (y['reason']=='topology_transition_relinearized')==changed,'trial acceptance rule mismatch')
        step.update(from_evaluation=before,to_evaluation=after,raw_residual_drop=float(drop),
                    drop_exceeds_replay_tolerance=bool(drop>TOL*max(1,x['raw_equation_residual'],y['raw_equation_residual'])),
                    topology_hash_changed=changed,objective_delta=y['objective']-x['objective'],
                    next_system_rebuilt_at_accepted_trial=any(s['state_evaluation']==after for s in verified_systems))
        verified_steps.append(step)
    output=dict(schema='w0-heart-first-strict-newton-prefix-v1', run_id=run_id,
                public_result_sha256=public_result_sha, source_commit=result['source_checkout'],
                native_status=result['status'], native_terminal=result.get('native_terminal'),
                verified_starts=len(starts), registered_starts=len(seeds), native_boolean_calls=result['native_boolean_calls'],
                evaluations=summaries, independently_verified_systems=verified_systems, terminations=[{'restart':e['restart'],'iteration':e['iteration'],'reason':e['reason']}
                        for e in events if e['type']=='attempt_terminal'], verified_positive_step=verified_steps[0] if steps else None,
                raw_terminal_packet_and_convergence_certified=False, whole_boolean_regions_certified=False,
                training_admitted=False, native_calls_in_replay=0)

    if trajectory:
        output.update(trajectory_report(result,ev,settings,steps,verified_steps,accepted_states,summaries))
        output['rejected_state_decisions_checked']=check_rejections(summaries,settings)
        exact=[row['index'] for row in summaries if 'exact_material' in row]
        require(exact==accepted_states, 'missing accepted-state exact moments')
        output['accepted_states_exactly_reintegrated']=exact
        if first_step_reference is not None:
            output['m29_numerical_prefix_comparison']=compare_m29(first_step_reference,folder)
    return output


def check_rejections(rows,settings):
    last={};checked=[]
    for row in rows:
        restart=row['restart']; reason=row['reason']
        if row['outcome']=='accepted':
            last[restart]=row
            continue
        if row['outcome']!='rejected':
            continue
        if row['iteration']==0:
            require(reason=='initial_multiplier_nonpositive' and row['multiplier_mu']<=0,'initial rejection mismatch')
        else:
            prior=last[restart]
            if reason=='stationary_multiplier_nonpositive':
                require(row['multiplier_mu']<=0 and row['raw_equation_residual']<=settings['equation_tolerance'] and
                        row['physical_relative_residual']<=settings['physical_relative_tolerance'],
                        'nonpositive stationary rejection mismatch')
            else:
                changed=row['topology_hash']!=prior['topology_hash']
                require(reason==('topology_transition' if changed else 'no_residual_decrease') and
                        row['raw_equation_residual']>=prior['raw_equation_residual'],
                        'rejected residual/topology decision mismatch')
        checked.append(row['index'])
    return checked


def trajectory_report(result, events, settings, steps, verified, accepted, rows):
    # Same ELF build and deterministic request must repeat M29 before tracking further.
    anchor_sha=('207f19d2c9c71e240f87650565625330466aec3e75c4ec63e326f1d7f1cfaf80',
                'd0ea04ff769c8a390c7eaf2f11670ad1ea4c987dc7c0d75221964d4799d31f39')
    hashes=tuple(r.get('material_sha256') for r in rows[:2])
    identical=steps[:1]==[(1,2)] and hashes==anchor_sha
    last=rows[accepted[-1]-1]
    summary=result.get('native_terminal_summary')
    require((result['status']=='NATIVE_TERMINAL')==(summary is not None), 'terminal summary mismatch')
    converged=False
    if summary is not None:
        require(result['native_terminal']==summary['status'] and summary['function_evaluations']==len(events),
                'native terminal status mismatch')
        if summary['status']=='CONVERGED':
            near(summary['quadric_coefficients'],events[last['index']-1]['quadric_coefficients'],'terminal coefficients')
            for key in ('raw_equation_residual','physical_relative_residual','multiplier_mu'):
                near(summary[key],last[key],'terminal '+key)
            require(last['raw_equation_residual']<=settings['equation_tolerance'] and
                    last['physical_relative_residual']<=settings['physical_relative_tolerance']
                    and last['multiplier_mu']>0 and 'exact_material' in last, 'terminal convergence replay failed')
            converged=True
    minima={}
    for idx in accepted:
        row=rows[idx-1]; key=str(row['restart'])
        if key not in minima or rows[minima[key]-1]['raw_equation_residual']>row['raw_equation_residual']:
            minima[key]=idx
    return dict(schema='w0-heart-strict-newton-trajectory-v1',
                m29_first_step_reproduced=identical, m29_full_material_digest_gate='PASS' if identical else 'FAIL',
                m29_reference_digests=anchor_sha, m30_observed_digests=hashes,
                verified_steps=verified, accepted_step_count=len(steps),
                last_accepted_evaluation=last['index'],last_observed_evaluation=rows[-1]['index'],
                minimum_raw_residual_accepted_indices_by_restart=minima,
                terminal_summary=summary, terminal_equation_and_physical_tolerances_verified=converged,
                raw_terminal_packet_and_convergence_certified=converged)


def compare_m29(reference, current):
    """Diagnostic additional to the preregistered two-digest gate."""
    public=reference/'result.json'
    require(hashlib.sha256(public.read_bytes()).hexdigest()=='d465b30768366dee0f10cf7fe8ec9415b7e5e6763ad1d3b3ac0684117982c2fb',
            'unbound M29 receipt')
    with (reference/'evaluations.jsonl').open('rb') as f:
        require(hashlib.file_digest(f,'sha256').hexdigest()==get_json(public)['file_sha256']['evaluations.jsonl'],
                'unbound M29 evaluations')
    comparison=[]
    with (reference/'evaluations.jsonl').open('rb') as old,(current/'evaluations.jsonl').open('rb') as new:
        for index in (1,2):
            a,b=read_json(old.readline())['material'],read_json(new.readline())['material']
            delta={}
            for field in ('quadric_coefficients','raw_moments','variable_raw_moments','moment_jacobian',
                          'quadric_source_vertices','variable_vertices'):
                x,y=np.asarray(a[field]),np.asarray(b[field])
                delta[field]=float(np.max(np.abs(x-y))) if x.shape==y.shape else None
            same_faces=all(a[k]==b[k] for k in ('variable_faces','quadric_source_faces'))
            same_topology=a['topology_hash']==b['topology_hash']
            comparison.append(dict(index=index,max_abs_difference=delta,ordered_faces_identical=same_faces,
                topology_hash_identical=same_topology,geometry_hash_identical=a['geometry_hash']==b['geometry_hash'],
                within_absolute_1e_9_and_same_connectivity=same_faces and same_topology and
                    all(v is not None and v<=TOL for v in delta.values())))
    return comparison


if __name__=='__main__':
    sys.set_int_max_str_digits(100000)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture_dir',type=Path)
    parser.add_argument('run_id')
    parser.add_argument('result_sha256')
    parser.add_argument('--trajectory',action='store_true',help='Verify the M30 multi-step run')
    parser.add_argument('--first-step-reference',type=Path,help='Compare additional numerical M29 prefix diagnostics')
    args=parser.parse_args()
    print(json.dumps(audit(args.capture_dir,args.run_id,args.result_sha256,trajectory=args.trajectory,
                           first_step_reference=args.first_step_reference),indent=2,allow_nan=False))
