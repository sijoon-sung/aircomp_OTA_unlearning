from pathlib import Path
import sys, json, time, math, hashlib, traceback
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'aircomp_noise_control_20261003'))
import importlib.util
spec=importlib.util.spec_from_file_location('noise_engine',R.parent/'aircomp_noise_control_20261003/run.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
M=N.M;E=N.E
SEEDS=[64001,64002,64003];METHODS=['random','channel','data'];LEVELS=[('no_noise',0.),('nominal20dB',.01)]
CAP=200;TARGETS={'q60':.60,'q65':.65};COUNTS={'local_calls':0,'aggregation_rounds':0,'dev_evaluations':0,'test_evaluations':0}
def dump(name,v):E.dump(R/name,v)
def digest(v):return hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
def groups_for(seed,h,hist):
    glob=hist.mean(0);cache={}
    def feature(g):
        if g not in cache:cache[g]=(M.js(hist[list(g)].mean(0),glob),1/(25*min(h[list(g)])**2))
        return cache[g]
    random=M.canon(np.array_split(np.random.default_rng(E.key(seed,'routing')).permutation(20),4))
    sorted_h=M.canon(np.array_split(np.argsort(h,kind='stable'),4));result={'random':random};traces={}
    for method,j in [('channel',1),('data',0)]:
        score=lambda gs:float(np.mean([feature(g)[j] for g in gs]))
        best=None
        for initial in [random,sorted_h]:
            gs=initial;value=score(gs);trace=[value]
            for it in range(15):
                candidates=set()
                for a in range(4):
                    for b in range(a+1,4):
                        for u in gs[a]:
                            for v in gs[b]:
                                z=[list(g) for g in gs];z[a].remove(u);z[b].remove(v);z[a].append(v);z[b].append(u);candidates.add(M.canon(z))
                cand=min(candidates,key=lambda g:(score(g),g));val=score(cand)
                if val>=value-1e-12:break
                gs,value=cand,val;trace.append(value)
            if best is None or (value,gs)<(best[0],best[1]):best=(value,gs,trace)
        result[method]=best[1];traces[method]=best[2]
    return {m:[list(g) for g in gs] for m,gs in result.items()},traces
def physical():
    rng=np.random.default_rng(64000);rows=[]
    for n in [3,5,8]:
        for hmin in [.1,.3,1.]:
            D=32;h=np.linspace(hmin,max(hmin,1),n);u=np.ones((n,D))*.2/math.sqrt(D)
            beta=1/(n*hmin*math.sqrt(D));x=u/(n*h[:,None]*beta)
            error=float(abs(beta*(h[:,None]*x).sum(0)-u.mean(0)).max())
            noise=rng.normal(size=(50000,D))*beta*.1;actual=float(np.mean(np.sum(noise**2,axis=1)));expected=.01/(n*n*hmin*hmin)
            power=float(np.mean(x*x,axis=1).max());relative=abs(actual/expected-1)
            assert error<1e-10 and power<=1+1e-12 and relative<.03
            rows.append(dict(n=n,hmin=hmin,mse=actual,expected=expected,relative_error=relative,max_power=power,mean_error=error))
    dump('physical_audit.json',dict(passed=True,rows=rows))
@torch.no_grad()
def metrics(p,y):
    pred=p.argmax(1);acc=float((pred==y).float().mean());recall=[float((pred[y==i]==i).float().mean()) for i in range(10)]
    return dict(accuracy=acc,worst_recall=min(recall),class_recall=recall)
def fit(sim,ids,c,name,hist):
    assert ids and len(set(ids))==len(ids)
    w=sim.initial[c].clone();events=[];curve=[];last={q:False for q in TARGETS};stops={};states={};preds={}
    for t in range(CAP):
        w,e=sim.step(w,ids,c,t);events.append(e);COUNTS['local_calls']+=len(ids);COUNTS['aggregation_rounds']+=1
        if (t+1)%20==0:
            E.assign(sim.local,w);p=E.predict(sim.local,sim.data['x'][sim.data['dev']]);COUNTS['dev_evaluations']+=1
            met=metrics(p,sim.data['y'][sim.data['dev']]);curve.append(dict(round=t+1,**met))
            names=[]
            for q,target in TARGETS.items():
                hit=met['accuracy']>=target
                if q not in stops and hit and last[q]:stops[q]=t+1;names.append(q)
                last[q]=hit
            if t+1==160:names.append('fixed160')
            if t+1==CAP:names.append('cap200');names += [q for q in TARGETS if q not in stops]
            if names:
                tp=E.predict(sim.local,sim.data['tx']);COUNTS['test_evaluations']+=1
                for q in names:states[q]=w.cpu().clone();preds[q]=tp.cpu().clone()
    costs={q:M.total(events[:stops.get(q,CAP)]) for q in TARGETS}
    feat=dict(js=M.js(hist[ids].mean(0),hist.mean(0)),n=len(ids),min_h=float(sim.base_h[ids].min()),q=float(1/(len(ids)**2*sim.base_h[ids].min()**2)),sample_count=sum(sim.data['counts'][i] for i in ids))
    obj=dict(ids=ids,shard=c,features=feat,curve=curve,stops={q:stops.get(q) for q in TARGETS},costs=costs,cost_fixed160=M.total(events[:160]),cost_cap=M.total(events),mean_mse=float(np.mean([e['expected_mse'] for e in events])),models={q:digest(v) for q,v in states.items()},test={q:metrics(p,sim.data['ty'].cpu()) for q,p in preds.items()},events=events)
    dump(name+'.json',obj);E.save(R/(name+'_states.pt'),states)
    return obj,states,preds
def reference(sim,ids,c,obj,states):
    w=sim.initial[c].clone();maxerror=0.;last={q:False for q in TARGETS};stops={};errs={}
    for t in range(CAP):
        w,e=sim.step(w,ids,c,t);COUNTS['local_calls']+=len(ids);COUNTS['aggregation_rounds']+=1
        if (t+1)%20==0:
            E.assign(sim.local,w);p=E.predict(sim.local,sim.data['x'][sim.data['dev']]);COUNTS['dev_evaluations']+=1
            a=metrics(p,sim.data['y'][sim.data['dev']])['accuracy']
            for q,target in TARGETS.items():
                hit=a>=target
                if q not in stops and hit and last[q]:stops[q]=t+1
                last[q]=hit
            for q in ['fixed160',*TARGETS]:
                target_t=160 if q=='fixed160' else obj['stops'][q] or CAP
                if t+1==target_t:errs[q]=float((w.cpu()-states[q]).abs().max())
    assert {q:stops.get(q) for q in TARGETS}==obj['stops']
    assert max(errs.values())<=1e-6
    return dict(errors=errs,stops_equal=True)
def main():
    start=time.perf_counter();results=[];meter=E.Power();error=None
    dump('provenance.json',{str(p):E.sha(p) for p in [R/'run.py',R/'PREREG.txt',R.parent/'aircomp_noise_control_20261003/run.py',R.parent/'aircomp_deletion_partition_20261003/run.py',E.ROOT/'experiment.py',R.parent/'research_20260929_pdf_ota_unlearning/core.py']})
    physical()
    try:
        with meter:
            for seed in SEEDS:
                data=E.load_data('FashionMNIST',.5,seed);sim=N.Sim(data,seed,0.)
                hist=np.array([torch.bincount(data['y'][p],minlength=10).cpu().numpy() for p in data['pools']],dtype=float);hist/=hist.sum(1,keepdims=True)
                groups,traces=groups_for(seed,sim.base_h,hist)
                dump(f'{seed}_routing.json',dict(groups=groups,traces=traces,histograms=hist.tolist(),h=sim.base_h.tolist(),counts=data['counts'],D=sim.D,metadata_bits=7680))
                dump(f'{seed}_data.json',dict(original_indices=data['original_indices'].tolist(),pools=[p.tolist() for p in data['original_pools']],public_dev_original=data['original_indices'][12000:].tolist()))
                for method in METHODS:
                    gs=groups[method];assert all(len(g)==5 for g in gs)
                    for label,sigma2 in LEVELS:
                        sim.sigma2=sigma2;prefix=f'{seed}_{method}_{label}';source=[];sp=[];sourcehash=[]
                        for c,g in enumerate(gs):
                            o,s,p=fit(sim,g,c,prefix+f'_source_{c}',hist);source.append(o);sp.append(p);sourcehash.append({k:digest(v) for k,v in p.items()})
                        sy=data['ty'].cpu();source_metrics={q:metrics(sum(p[q] for p in sp)/4,sy) for q in [*TARGETS,'fixed160','cap200']}
                        source_success={q:all(o['stops'][q] is not None for o in source) for q in TARGETS};deletions=[]
                        E.log(phase='source',seed=seed,method=method,noise=label,success=source_success,acc160=source_metrics['fixed160']['accuracy'])
                        for u in range(20):
                            c=next(j for j,g in enumerate(gs) if u in g);ids=[i for i in gs[c] if i!=u]
                            o,s,p=fit(sim,ids,c,prefix+f'_delete_{u}',hist)
                            ref=reference(sim,ids,c,o,s) if u==0 else None
                            assert all(digest(v)==sourcehash[j][q] for j,pp in enumerate(sp) for q,v in pp.items())
                            ens={q:metrics((p[q]+sum(pp[q] for j,pp in enumerate(sp) if j!=c))/4,sy) for q in [*TARGETS,'fixed160','cap200']}
                            deletions.append(dict(client=u,shard=c,stops=o['stops'],costs=o['costs'],cost_fixed160=o['cost_fixed160'],test=ens,features=o['features'],mean_mse=o['mean_mse'],reference=ref,unaffected_prediction_hashes_unchanged=True))
                            if (u+1)%5==0:E.log(phase='delete',seed=seed,method=method,noise=label,done=u+1,elapsed_seconds=round(time.perf_counter()-start,1))
                        row=dict(seed=seed,method=method,noise=label,sigma2=sigma2,groups=gs,source_test=source_metrics,source_success=source_success,source_stops={q:[o['stops'][q] for o in source] for q in TARGETS},source_cost={q:{k:sum(o['costs'][q][k] for o in source) for k in source[0]['costs'][q]} for q in TARGETS},deletions=deletions)
                        dump(prefix+'_summary.json',row);results.append(row)
                        dump(prefix+'_resource.json',dict(seconds=time.perf_counter()-start,**COUNTS,**meter.report()))
            dump('results.json',results)
    except BaseException as e:
        error=traceback.format_exc();dump('failure.json',dict(error=error,completed_conditions=len(results)));raise
    finally:
        dump('resource.json',dict(seconds=time.perf_counter()-start,**COUNTS,**meter.report(),completed_conditions=len(results),failed=error is not None))
    E.log(phase='complete',conditions=len(results),seconds=time.perf_counter()-start)
if __name__=='__main__':main()
