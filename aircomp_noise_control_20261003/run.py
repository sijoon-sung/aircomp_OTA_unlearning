from pathlib import Path
import sys,time,json,math
import numpy as np
import torch
R=Path(__file__).resolve().parent;OLD=R.parent/'aircomp_deletion_partition_20261003'
sys.path.insert(0,str(OLD))
import importlib.util
spec=importlib.util.spec_from_file_location('partition_engine',OLD/'run.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
E=M.E
COUNT={'local_calls':0,'aggregate_rounds':0}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(name,v):E.dump(R/name,v)
class Sim(M.Sim):
    def __init__(self,data,seed,sigma2):super().__init__(data,seed);self.sigma2=sigma2
    def step(self,w,ids,c,t):
        updates=[];norms=[];clipped=0
        for i in ids:
            E.assign(self.local,w);pool=self.data['pools'][i]
            for step in range(2):
                batch=pool[torch.randint(len(pool),(64,),device='cuda',generator=E.gen(E.key(self.seed,i,t,step,'batch')))]
                grad=E.gradient(self.local,self.data['x'][batch],self.data['y'][batch]);E.assign(self.local,E.flat(self.local)-.05*grad)
            u=E.flat(self.local)-w;norm=float(u.norm());clipped+=int(norm>1)
            updates.append(u/max(1.,norm));norms.append(min(1.,norm))
        n=len(ids);h=self.h[t,ids];beta=1/(n*h.min()*math.sqrt(self.D));variance=beta**2*self.sigma2
        noise=torch.randn(self.D,device='cuda',generator=E.gen(E.key(self.seed,c,t,'fixed_budget_noise')))*math.sqrt(variance)
        power=np.array(norms)**2/(n*n*h*h*beta*beta*self.D);assert power.max()<=1.00001
        out=w+torch.stack(updates).mean(0)+noise;assert torch.isfinite(out).all()
        control=64*n+64;pilot=8*n;dl=32*self.D
        e={'round':t+1,'shard':c,'ids':ids,'n':n,'beta':float(beta),'min_h':float(h.min()),'expected_mse':float(self.D*variance),'max_power':float(power.max()),'repeats':1,
           'ul_reals':self.D,'dl_bits':dl,'pilot_reals':pilot,'control_bits':control,'total_re':self.D+pilot+(control+dl)/2,'total_re_dl6':self.D+pilot+control/2+dl/6,
           'local_calls':n,'local_steps':2*n,'signal_energy':float(self.D*power.sum()),'clipped':clipped}
        COUNT['local_calls']+=n;COUNT['aggregate_rounds']+=1
        return out,e
def main():
    start=time.perf_counter();records=[];states={};checks=[]
    provenance=[R/'run.py',R/'PREREG.txt',OLD/'run.py',OLD/'results.json',OLD/'fixed160_results.json',E.ROOT/'experiment.py',R.parent/'research_20260929_pdf_ota_unlearning/core.py']
    provenance += [OLD/f'{seed}_{suffix}.json' for seed in [62001,62002,62003] for suffix in ['routing','data']]
    dump('provenance_resume.json',{str(p):E.sha(p) for p in provenance})
    with E.Power() as meter:
        for seed in [62001,62002,62003]:
            data=E.load_data('FashionMNIST',.5,seed);olddata=read(OLD/f'{seed}_data.json');route=read(OLD/f'{seed}_routing.json')
            assert data['original_indices'].tolist()==olddata['original_indices']
            assert [p.tolist() for p in data['original_pools']]==olddata['pools']
            legacy=M.Sim(data,seed);sim=Sim(data,seed,.01);assert sim.base_h.tolist()==route['h']
            for method in ['channel','data','delete_joint']:
                sim.sigma2=.01
                groups=route['methods'][method]['groups'];ids=groups[0]
                oldstep,_=legacy.step(legacy.initial[0],ids,0,0);newstep,_=sim.step(sim.initial[0],ids,0,0)
                err=float((oldstep-newstep).abs().max());assert err==0
                checks.append({'seed':seed,'method':method,'legacy_step_error':err,'data_and_channel_equal':True})
                for label,sigma2 in [('no_noise',0.),('nominal0dB',1.)]:
                    completed=R/f'{seed}_{method}_{label}_summary.json'
                    if completed.exists():
                        records.append(read(completed));E.log(phase='reuse_completed',seed=seed,method=method,noise=label);continue
                    sim.sigma2=sigma2;sim.groups=groups;vectors=[];events=[];pp=[]
                    for c,g in enumerate(groups):
                        w=sim.initial[c].clone()
                        for t in range(160):w,e=sim.step(w,g,c,t);events.append(e)
                        vectors.append(w);pp.append(sim.probs(w))
                    source=sim.metric([sum(p[k] for p in pp)/4 for k in range(2)]);sourcecost=M.total(events)
                    dump(f'{seed}_{method}_{label}_source.json',{'metrics':source,'cost':sourcecost,'groups':groups,'events':events,'sigma2':sigma2})
                    E.log(phase='source',seed=seed,method=method,noise=label,test=source['test'])
                    deleted=[]
                    for u in range(20):
                        c=next(c for c,g in enumerate(groups) if u in g);ids=[i for i in groups[c] if i!=u];other=[sum(p[k] for j,p in enumerate(pp) if j!=c) for k in range(2)]
                        w=sim.initial[c].clone();ev=[]
                        for t in range(160):w,e=sim.step(w,ids,c,t);ev.append(e)
                        p=sim.probs(w);metrics=sim.metric([(p[k]+other[k])/4 for k in range(2)]);err=None
                        if u==0:
                            ref=sim.initial[c].clone()
                            for t in range(160):ref,_=sim.step(ref,ids,c,t)
                            err=float((ref-w).abs().max());assert err<=1e-6
                        d={'client':u,'shard':c,'metrics':metrics,'cost':M.total(ev),'reference_maxabs':err,'mean_expected_mse':float(np.mean([e['expected_mse'] for e in ev]))}
                        deleted.append(d);dump(f'{seed}_{method}_{label}_delete_{u}.json',{'summary':d,'events':ev});states[f'{seed}_{method}_{label}_delete_{u}']=w.cpu()
                        if (u+1)%5==0:E.log(phase='deletion',seed=seed,method=method,noise=label,completed=u+1)
                    row={'seed':seed,'method':method,'noise':label,'sigma2':sigma2,'groups':groups,'source':source,'source_cost':sourcecost,'deletions':deleted,'mean_delete_test':float(np.mean([d['metrics']['test'] for d in deleted]))}
                    records.append(row);dump(f'{seed}_{method}_{label}_summary.json',row);states[f'{seed}_{method}_{label}_source']=[w.cpu() for w in vectors]
                    dump(f'{seed}_{method}_{label}_resource_cumulative.json',{'seconds':time.perf_counter()-start,**COUNT,**meter.report()})
    resource={'seconds':time.perf_counter()-start,**COUNT,'legacy_check_calls':M.COUNT,**meter.report()}
    dump('results.json',records);dump('control_checks.json',checks);dump('resource.json',resource);E.save(R/'states.pt',states);E.log(phase='complete',seconds=resource['seconds'])
if __name__=='__main__':main()
