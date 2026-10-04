"""Finite SISO AirComp/SISA study with fixed repetition and conditional zCDP."""
from pathlib import Path
import argparse, json, math, os, subprocess, time, traceback
import numpy as np
import torch
import experiment as E

ROOT=Path(__file__).resolve().parent
C=.1
DELTA=1e-5
T=160
GROUP_CACHE={}

def js(a,b):
    a=np.maximum(np.asarray(a,float),1e-30); b=np.maximum(np.asarray(b,float),1e-30)
    a=a/a.sum(-1,keepdims=True); b=b/b.sum(-1,keepdims=True); m=(a+b)/2
    return .5*np.sum(a*np.log(a/m)+b*np.log(b/m),axis=-1)

def eps(rho):
    return float(rho+2*np.sqrt(rho*np.log(1/DELTA)))

def rho_budget(epsilon):
    return (np.sqrt(np.log(1/DELTA)+epsilon)-np.sqrt(np.log(1/DELTA)))**2

def radio_data_score(groups,h,hist):
    source=[]; deletion=[]
    for group in groups:
        g=np.array(group); m=len(g)
        source.append(1/(m*m*np.min(h[g])**2))
        for i in g:
            retained=g[g!=i]
            deletion.append(1/((m-1)**2*np.min(h[retained])**2))
    radio=.5*np.mean(source)+.5*np.mean(deletion)
    divergence=np.mean([js(hist[g].mean(0),hist.mean(0)) for g in groups])
    return float(radio),float(divergence)

def routing(seed,K,h,hist):
    cachekey=(seed,K)
    if cachekey in GROUP_CACHE: return GROUP_CACHE[cachekey]
    start=time.perf_counter()
    random=[list(map(int,g)) for g in np.array_split(np.random.default_rng(E.key(seed,'routing')).permutation(20),K)]
    channel=[list(map(int,g)) for g in np.array_split(np.argsort(h,kind='stable'),K)]
    scales=np.maximum(radio_data_score(random,h,hist),1e-12)
    def objective(groups): return float(np.mean(np.array(radio_data_score(groups,h,hist))/scales))
    best=None
    for initial in [random,channel]:
        groups=[g.copy() for g in initial]; value=objective(groups)
        for _ in range(30):
            move=None; candidate_value=value
            for a in range(K):
                for b in range(a+1,K):
                    for ia in range(len(groups[a])):
                        for ib in range(len(groups[b])):
                            groups[a][ia],groups[b][ib]=groups[b][ib],groups[a][ia]
                            v=objective(groups)
                            groups[a][ia],groups[b][ib]=groups[b][ib],groups[a][ia]
                            if v<candidate_value-1e-10:
                                candidate_value=v; move=(a,b,ia,ib)
            if move is None: break
            a,b,ia,ib=move
            groups[a][ia],groups[b][ib]=groups[b][ib],groups[a][ia]
            value=candidate_value
        if best is None or value<best[0]: best=(value,[g.copy() for g in groups])
    result=dict(random=random,channel=channel,joint=best[1],construction_seconds=time.perf_counter()-start,
                objective={name:objective(g) for name,g in [('random',random),('channel',channel),('joint',best[1])]})
    assert result['objective']['joint']<=min(result['objective']['random'],result['objective']['channel'])+1e-10
    GROUP_CACHE[cachekey]=result
    return result

def metric(p,y):
    pred=p.argmax(1); recall=[]; f1=[]
    for c in range(10):
        tp=np.sum((pred==c)&(y==c)); fn=np.sum((pred!=c)&(y==c)); fp=np.sum((pred==c)&(y!=c))
        recall.append(float(tp/(tp+fn))); f1.append(float(2*tp/max(1,2*tp+fn+fp)))
    return dict(accuracy=float(np.mean(pred==y)),macro_f1=float(np.mean(f1)),worst_recall=min(recall),recall=recall)

class Sim(E.Engine):
    def __init__(self,data,seed,K,method,snr,repeats,cap,rounds=T):
        super().__init__(data,seed,K,method,rounds)
        self.snr,self.repeats,self.cap=snr,repeats,cap
        self.sigma2=10**(-snr/10)
        counts=np.stack([torch.bincount(data['y'][p],minlength=10).cpu().numpy() for p in data['pools']])
        self.hist=counts/counts.sum(1,keepdims=True)
        self.routes=routing(seed,K,self.base_h,self.hist)
        self.groups=self.routes[method]

    def physical(self,w,ids,c,t,branch):
        updates=[]; norms=[]; clipped_count=0
        for i in ids:
            E.assign(self.local,w); pool=self.data['pools'][i]
            for step in range(2):
                indices=pool[torch.randint(len(pool),(64,),device='cuda',generator=E.gen(E.key(self.seed,i,t,step,'batch')))]
                grad=E.gradient(self.local,self.data['x'][indices],self.data['y'][indices])
                E.assign(self.local,E.flat(self.local)-.05*grad)
            u=E.flat(self.local)-w; norm=float(u.norm())
            assert math.isfinite(norm)
            clipped_count+=int(norm>C)
            u=u*min(1.,C/max(norm,1e-30)); updates.append(u); norms.append(min(norm,C))
        n=len(ids); h=self.h[t,ids]
        beta=C/(n*np.min(h)*math.sqrt(self.D))
        sensitivity=2*C/n
        if self.cap is not None:
            perstep=rho_budget(self.cap)/(2*self.rounds)
            beta=max(beta,math.sqrt(sensitivity**2*self.repeats/(2*perstep*self.sigma2)))
        var=beta**2*self.sigma2/self.repeats
        powers=np.array(norms)**2/(n*n*h*h*beta**2*self.D)
        assert powers.max()<=1.00001
        avg=torch.stack(updates).mean(0)
        noise=torch.randn(self.D,device='cuda',generator=E.gen(E.key(self.seed,c,t,branch,'snr_noise')))*math.sqrt(var)
        assert torch.isfinite(avg+noise).all()
        expected=self.D*var; signal=float(avg.square().sum())
        control=64*n+64; pilots=8*n; dl=32*self.D; ul=self.repeats*self.D
        event=dict(t=t,c=c,ids=ids,active=n,beta=float(beta),variance=float(var),expected_mse=float(expected),
                   actual_noise_norm2=float(noise.square().sum()),aggregate_snr_db=float(10*np.log10(max(signal,1e-30)/expected)),
                   signal_norm2=signal,per_client_rho=float(sensitivity**2/(2*var)),max_power=float(powers.max()),
                   min_h=float(h.min()),clipped_clients=clipped_count,ul_reals=ul,dl_bits=dl,control_bits=control,
                   pilot_reals=pilots,total_re=ul+dl/2+control/2+pilots,local_calls=n,local_steps=2*n,
                   signal_energy=float(self.repeats*self.D*powers.sum()))
        return avg+noise,event

    @torch.no_grad()
    def evaluate(self,vectors):
        out={}
        for split,x,y in [('dev',self.data['x'][self.data['dev']],self.data['y'][self.data['dev']]),
                          ('test',self.data['tx'],self.data['ty'])]:
            probs=[]
            for w in vectors:
                E.assign(self.local,w); probs.append(E.predict(self.local,x).cpu().numpy())
            p=np.mean(probs,axis=0); out[split]=metric(p,y.cpu().numpy()); out[split+'_prob']=p
        return out

    def train(self,start,deleted=None,only=None,reference_snap=None,evaluate=True,label=''):
        states=[w.clone() for w in start]; events=[]; snapshots={}; curve=[]; totalrho=np.zeros(20)
        metadata=20*(10*32+32+32) if deleted is None else 0
        totals={k:0. for k in ['ul_reals','dl_bits','control_bits','pilot_reals','total_re','local_calls','local_steps','signal_energy']}
        totals['total_re']=metadata/2
        t0=time.perf_counter()
        affected=next((c for c,g in enumerate(self.groups) if deleted in g),None)
        for t in range(self.rounds):
            for c in (range(self.K) if only is None else [only]):
                ids=[i for i in self.groups[c] if i!=deleted]
                branch=0 if c!=affected else E.key(deleted,'deletion_noise')
                update,event=self.physical(states[c],ids,c,t,branch)
                states[c]=states[c]+update; events.append(event)
                totalrho[ids]+=event['per_client_rho']
                for k in totals: totals[k]+=event[k]
            if (t+1)%40==0 or t+1==self.rounds:
                current=[v.clone() for v in states]
                if reference_snap is not None:
                    for c in range(self.K):
                        if c!=only: current[c]=reference_snap[t+1][c].to('cuda').clone()
                snapshots[t+1]=[v.cpu().clone() for v in current]
                point=dict(round=t+1,cost=totals.copy(),rho_per_client=totalrho.tolist(),epsilon_max=eps(totalrho.max()))
                if evaluate:
                    pred=self.evaluate(current)
                    point.update(dev=pred['dev'],test=pred['test'])
                curve.append(point)
                if (t+1)%80==0 or self.rounds<40: E.log(event='training',phase=label,seed=self.seed,K=self.K,method=self.method,snr=self.snr,R=self.repeats,cap=self.cap,round=t+1)
        torch.cuda.synchronize()
        totals['seconds']=time.perf_counter()-t0
        totals['metadata_bits']=metadata
        return states,dict(events=events,cost=totals,curve=curve),snapshots

def physical_audit(out):
    rng=np.random.default_rng(10800); m,D=5,64; h=rng.uniform(.1,1,m)
    u=rng.normal(size=(m,D)); u*=C/np.maximum(C,np.linalg.norm(u,axis=1))[:,None]
    findings=[]
    for snr in [0,10,20]:
        sigma2=10**(-snr/10)
        for R in [1,4]:
            beta=C/(m*h.min()*np.sqrt(D)); x=u/(m*h[:,None]*beta)
            reconstructed=beta*np.sum(h[:,None]*x,axis=0)
            var=beta**2*sigma2/R; rho=(2*C/m)**2/(2*var)
            assert np.max(np.abs(reconstructed-u.mean(0)))<1e-12
            assert np.max(np.mean(x*x,axis=1))<=1+1e-12
            empirical=np.mean(np.sum((rng.normal(size=(20000,D))*np.sqrt(var))**2,axis=1))
            assert abs(empirical/(D*var)-1)<.04
            assert abs(rho/(2*D*h.min()**2*R/sigma2)-1)<1e-12
            findings.append(dict(snr=snr,R=R,mse_expected=D*var,mse_empirical=empirical,rho=rho))
    for cap in [8,100]: assert abs(eps(rho_budget(cap))-cap)<1e-10
    E.dump(out/'physical_privacy_audit.json',dict(passed=True,findings=findings,conditional_adjacency=True))

def run_case(out,seed,K,method,snr,R,cap,rounds=T):
    capname='natural' if cap is None else f'eps{cap}'
    name=f's{seed}_K{K}_{method}_snr{snr}_R{R}_{capname}'
    folder=out/name; folder.mkdir(exist_ok=False); began=time.perf_counter()
    E.log(event='case_start',case=name)
    data=E.load_data('FashionMNIST',.5,seed); sim=Sim(data,seed,K,method,snr,R,cap,rounds)
    np.savez_compressed(folder/'partition.npz',original=data['original_indices'],**{f'client_{i}':p for i,p in enumerate(data['original_pools'])})
    np.savez_compressed(folder/'labels.npz',dev=data['y'][data['dev']].cpu().numpy(),test=data['ty'].cpu().numpy())
    E.dump(folder/'config.json',dict(seed=seed,K=K,method=method,snr=snr,repeats=R,cap=cap,rounds=rounds,C=C,D=sim.D,
           groups=sim.groups,base_h=sim.base_h.tolist(),histogram=sim.hist.tolist(),counts=data['counts'],
           routes=sim.routes,conditional_privacy='client features; public fixed histograms/counts/CSI/routing',
           radio_data_score=radio_data_score(sim.groups,sim.base_h,sim.hist)))
    source,sl,ss=sim.train(sim.initial,label='source')
    E.save(folder/'source_models.pt',[w.cpu() for w in source]); E.save(folder/'source_snapshots.pt',ss)
    E.dump(folder/'source_ledger.json',sl)
    deletions=[0,7,14] if (seed==10801 and K==4 and snr==10 and R==1 and cap is None and rounds==T) else [0]
    results=[]
    for deleted in deletions:
        affected=next(c for c,g in enumerate(sim.groups) if deleted in g)
        ref,rl,rs=sim.train(sim.initial,deleted=deleted,label=f'reference_delete{deleted}')
        start=[w.clone() for w in source]; start[affected]=sim.initial[affected].clone()
        replay,ul,us=sim.train(start,deleted=deleted,only=affected,reference_snap=ss,evaluate=False,label=f'replay_delete{deleted}')
        assert all(torch.equal(a,b) for a,b in zip(ref,replay))
        assert all(all(torch.equal(a,b) for a,b in zip(rs[t],us[t])) for t in rs)
        assert all(torch.equal(source[c],ref[c]) for c in range(K) if c!=affected)
        assert all(deleted not in e['ids'] and e['c']==affected for e in ul['events'])
        combined=[]
        for src,reference,replayed in zip(sl['curve'],rl['curve'],ul['curve']):
            rho=np.array(src['rho_per_client'])+np.array(replayed['rho_per_client'])
            point=dict(round=src['round'],source=src,deleted=reference,delete_cost=replayed['cost'],
                       combined_total_re=src['cost']['total_re']+replayed['cost']['total_re'],
                       source_epsilon=src['epsilon_max'],combined_epsilon=eps(rho.max()),rho_per_client=rho.tolist())
            if cap is not None: assert point['combined_epsilon']<=cap+1e-7
            combined.append(point)
        pred=sim.evaluate(replay)
        assert abs(pred['test']['accuracy']-rl['curve'][-1]['test']['accuracy'])<1e-12
        np.savez_compressed(folder/f'delete{deleted}_probabilities.npz',dev=pred['dev_prob'],test=pred['test_prob'])
        E.save(folder/f'delete{deleted}_reference_models.pt',[w.cpu() for w in ref])
        E.save(folder/f'delete{deleted}_replay_models.pt',[w.cpu() for w in replay])
        E.dump(folder/f'delete{deleted}_reference_ledger.json',rl)
        E.dump(folder/f'delete{deleted}_replay_ledger.json',ul)
        row=dict(case=name,seed=seed,K=K,method=method,snr=snr,repeats=R,cap=cap,deleted_client=deleted,
                 source_cost=sl['cost'],reference_cost=rl['cost'],delete_cost=ul['cost'],curve=combined,
                 replay_exact=True,checkpoints_exact=True,affected=affected,
                 mean_js=radio_data_score(sim.groups,sim.base_h,sim.hist)[1],
                 mean_actual_snr_db=float(np.mean([e['aggregate_snr_db'] for e in sl['events']])),
                 mean_mse=float(np.mean([e['expected_mse'] for e in sl['events']])),
                 clip_fraction=sum(e['clipped_clients'] for e in sl['events'])/sl['cost']['local_calls'])
        E.dump(folder/f'delete{deleted}_result.json',row); results.append(row)
        E.log(event='deletion_complete',case=name,deleted=deleted,accuracy=pred['test']['accuracy'],epsilon=combined[-1]['combined_epsilon'],exact=True)
    E.dump(folder/'complete.json',dict(case=name,deletions=deletions,seconds=time.perf_counter()-began))
    del sim,data,source,ref,replay
    torch.cuda.empty_cache()
    return results

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--preflight',action='store_true'); args=ap.parse_args()
    out=ROOT/('snr_privacy_preflight_v1' if args.preflight else 'snr_privacy_v1'); out.mkdir(exist_ok=False)
    inventory=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True)
    foreign=[s for s in inventory.splitlines()[1:] if s.strip() and 'ChatGPT.exe' not in s and not s.strip().startswith(str(os.getpid())+',')]
    if foreign: E.dump(out/'blocked.json',dict(processes=foreign)); raise RuntimeError('Existing GPU work')
    E.dump(out/'environment.json',dict(pid=os.getpid(),inventory=inventory,torch=torch.__version__,code_sha256=E.sha(__file__),
           base_sha256=E.sha(ROOT/'experiment.py'),core_sha256=E.sha(E.BASE/'research_20260929_pdf_ota_unlearning/core.py'),
           prereg_sha256=E.sha(ROOT/'SNR_PRIVACY_PREREG.md')))
    jobs=[]
    if args.preflight: jobs=[(10800,4,'joint',10,1,8)]
    else:
        for seed in [10801,10802,10803]:
            for K in ([2,4,5] if seed==10801 else [4]):
                for snr in [0,10,20]:
                    for method in ['random','channel','joint']: jobs.append((seed,K,method,snr,1,None))
        for seed in [10802,10803]:
            for method in ['random','channel','joint']: jobs.append((seed,4,method,10,4,None))
        for cap in [8,100]:
            for method in ['random','channel','joint']: jobs.append((10801,4,method,10,1,cap))
    E.dump(out/'jobs.json',jobs)
    began=time.perf_counter(); rows=[]; error=None
    with E.Power() as meter:
        try:
            physical_audit(out)
            for job in jobs: rows.extend(run_case(out,*job,rounds=4 if args.preflight else T))
            E.dump(out/'results.json',rows)
        except BaseException:
            error=traceback.format_exc(); E.dump(out/'failure.json',dict(error=error,completed_deletions=len(rows)))
        finally: torch.cuda.synchronize()
    # Count existing ledgers, including a partial failed case, without duplicating cached sources.
    all_ledgers=list(out.glob('*/source_ledger.json'))+list(out.glob('*/delete*_reference_ledger.json'))+list(out.glob('*/delete*_replay_ledger.json'))
    calls=sum(json.loads(p.read_text())['cost']['local_calls'] for p in all_ledgers)
    E.dump(out/'cost_total.json',dict(seconds=time.perf_counter()-began,**meter.report(),completed_deletions=len(rows),
           source_cases=len(list(out.glob('*/source_ledger.json'))),local_calls=calls,error=error,ledger_count=len(all_ledgers)))
    if error: raise RuntimeError(error)
    E.dump(out/'complete.json',dict(source_cases=len(jobs),deletions=len(rows),all_exact=all(r['replay_exact'] for r in rows)))
    E.log(event='worker_complete',cases=len(jobs),deletions=len(rows),seconds=time.perf_counter()-began)

if __name__=='__main__': main()
