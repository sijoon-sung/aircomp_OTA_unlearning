from pathlib import Path
import sys,time,json,math
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'research_20261003_sisa_aircomp'))
import experiment as E
METHODS=['random','channel','data','train_joint','delete_joint']
COUNT={'local_calls':0,'aggregation_rounds':0}
def dump(name,v):E.dump(R/name,v)
def js(a,b):
    a=np.maximum(a,1e-15);b=np.maximum(b,1e-15);m=(a+b)/2
    return float(.5*np.sum(a*np.log(a/m)+b*np.log(b/m)))
def canon(groups):return tuple(sorted(tuple(sorted(map(int,g))) for g in groups))
def routing(seed,h,hist):
    start=time.perf_counter();globalhist=hist.mean(0);cache={}
    def group(g):
        if g not in cache:
            n=len(g);a=np.array(g);q=1/(n*n*min(h[a])**2)
            dq=sum(1/((n-1)**2*min(h[[i for i in g if i!=u]])**2) for u in g)
            cache[g]=(js(hist[a].mean(0),globalhist),q,dq)
        return cache[g]
    def raw(groups):
        v=np.array([group(g) for g in groups]);return np.array([v[:,0].mean(),v[:,1].mean(),v[:,2].sum()/20])
    random=canon(np.array_split(np.random.default_rng(E.key(seed,'routing')).permutation(20),4))
    channel=canon(np.array_split(np.argsort(h),4));scale=np.maximum(raw(random),1e-12)
    def objective(g,method):
        d,q,dq=raw(g)/scale
        return {'channel':q,'data':d,'train_joint':.5*(d+q),'delete_joint':.5*(d+dq)}[method]
    output={'random':{'groups':random,'score':raw(random).tolist(),'objective':None,'trace':[]}}
    for method in METHODS[1:]:
        best=None
        for initial in [random,channel]:
            g=initial;value=objective(g,method);trace=[value]
            for it in range(15):
                candidates=set()
                for a in range(4):
                    for b in range(4):
                        if a==b:continue
                        if len(g[a])>3 and len(g[b])<8:
                            for u in g[a]:
                                x=[list(t) for t in g];x[a].remove(u);x[b].append(u);candidates.add(canon(x))
                        if a<b:
                            for u in g[a]:
                                for v in g[b]:
                                    x=[list(t) for t in g];x[a].remove(u);x[b].remove(v);x[a].append(v);x[b].append(u);candidates.add(canon(x))
                candidate=min(candidates,key=lambda z:(objective(z,method),z));v=objective(candidate,method)
                if v>=value-1e-12:break
                g,value=candidate,v;trace.append(value)
            if best is None or (value,g)<(best[0],best[1]):best=(value,g,trace)
        output[method]={'groups':best[1],'score':raw(best[1]).tolist(),'objective':best[0],'trace':best[2]}
    for obj in output.values():
        assert sorted(i for g in obj['groups'] for i in g)==list(range(20))
        assert all(3<=len(g)<=8 for g in obj['groups'])
    return {'methods':output,'scales':scale.tolist(),'seconds':time.perf_counter()-start,'h':h.tolist(),'histograms':hist.tolist(),'cache_groups':len(cache)}
def total(events):
    keys=['ul_reals','dl_bits','pilot_reals','control_bits','total_re','total_re_dl6','local_calls','local_steps','signal_energy']
    return {k:float(sum(e[k] for e in events)) for k in keys}
class Sim(E.Engine):
    def __init__(self,data,seed):
        super().__init__(data,seed,4,'random',240)
        self.initial=[self.initial[0].clone() for _ in range(4)]
    def step(self,w,ids,c,t):
        updates=[];norms=[];clipped=0
        for i in ids:
            E.assign(self.local,w);pool=self.data['pools'][i]
            for step in range(2):
                batch=pool[torch.randint(len(pool),(64,),device='cuda',generator=E.gen(E.key(self.seed,i,t,step,'batch')))]
                grad=E.gradient(self.local,self.data['x'][batch],self.data['y'][batch]);E.assign(self.local,E.flat(self.local)-.05*grad)
            u=E.flat(self.local)-w;norm=float(u.norm());clipped+=int(norm>1)
            updates.append(u/max(1.,norm));norms.append(min(1.,norm))
        n=len(ids);h=self.h[t,ids];beta=1/(n*h.min()*math.sqrt(self.D));variance=beta**2*.01
        noise=torch.randn(self.D,device='cuda',generator=E.gen(E.key(self.seed,c,t,'fixed_budget_noise')))*math.sqrt(variance)
        power=np.array(norms)**2/(n*n*h*h*beta*beta*self.D);assert power.max()<=1.00001
        out=w+torch.stack(updates).mean(0)+noise;assert torch.isfinite(out).all()
        control=64*n+64;pilot=8*n;dl=32*self.D
        e={'round':t+1,'shard':c,'ids':ids,'n':n,'beta':float(beta),'min_h':float(h.min()),'expected_mse':float(self.D*variance),'max_power':float(power.max()),'repeats':1,
           'ul_reals':self.D,'dl_bits':dl,'pilot_reals':pilot,'control_bits':control,'total_re':self.D+pilot+(control+dl)/2,'total_re_dl6':self.D+pilot+control/2+dl/6,
           'local_calls':n,'local_steps':2*n,'signal_energy':float(self.D*power.sum()),'clipped':clipped}
        COUNT['local_calls']+=n;COUNT['aggregation_rounds']+=1
        return out,e
    @torch.no_grad()
    def probs(self,w):
        E.assign(self.local,w)
        return [E.predict(self.local,self.data['x'][self.data['dev']]),E.predict(self.local,self.data['tx'])]
    def metric(self,probs):
        out={}
        for name,p,y in zip(['dev','test'],probs,[self.data['y'][self.data['dev']],self.data['ty']]):
            pred=p.argmax(1);recall=[float((pred[y==c]==c).float().mean()) for c in range(10)]
            out[name]=float((pred==y).float().mean());out[name+'_worst_recall']=min(recall)
        return out
def physical_audit():
    rng=np.random.default_rng(62000);errors=[]
    for n in [2,3,5,8]:
        D=32;h=rng.uniform(.1,1,n);u=rng.normal(size=(n,D));u/=np.maximum(1,np.linalg.norm(u,axis=1))[:,None]
        beta=1/(n*h.min()*np.sqrt(D));x=u/(n*h[:,None]*beta)
        error=float(abs(beta*(h[:,None]*x).sum(0)-u.mean(0)).max())
        assert error<1e-12 and np.mean(x*x,axis=1).max()<=1+1e-12
        errors.append(error)
    dump('physical_audit.json',{'passed':True,'errors':errors,'R':1})
def run():
    start=time.perf_counter();records=[];states={};physical_audit()
    dump('provenance.json',{str(p):E.sha(p) for p in [R/'run.py',R/'PREREG.txt',E.ROOT/'experiment.py',R.parent/'research_20260929_pdf_ota_unlearning/core.py']})
    with E.Power() as meter:
        for seed in [62001,62002,62003]:
            data=E.load_data('FashionMNIST',.5,seed);sim=Sim(data,seed)
            counts=np.array([torch.bincount(data['y'][p],minlength=10).cpu().numpy() for p in data['pools']]);hist=counts/counts.sum(1,keepdims=True)
            route=routing(seed,sim.base_h,hist);dump(f'{seed}_routing.json',route)
            dump(f'{seed}_data.json',{'original_indices':data['original_indices'].tolist(),'pools':[p.tolist() for p in data['original_pools']]})
            targets={}
            for method in METHODS:
                sim.groups=[list(g) for g in route['methods'][method]['groups']];vectors=[];source_events=[];source_probs=[]
                for c,g in enumerate(sim.groups):
                    w=sim.initial[c].clone()
                    for t in range(160):w,e=sim.step(w,g,c,t);source_events.append(e)
                    vectors.append(w);source_probs.append(sim.probs(w));E.log(phase='source',seed=seed,method=method,shard=c)
                metrics=sim.metric([sum(p[k] for p in source_probs)/4 for k in range(2)])
                sourcecost=total(source_events);sourcecost['setup_bits']=7680;sourcecost['total_re']+=3840;sourcecost['total_re_dl6']+=3840
                dump(f'{seed}_{method}_source.json',{'groups':sim.groups,'metrics':metrics,'cost':sourcecost,'events':source_events})
                unchanged=[v.clone() for v in vectors];dels=[]
                for u in range(20):
                    c=next(c for c,g in enumerate(sim.groups) if u in g);ids=[i for i in sim.groups[c] if i!=u]
                    other=[sum(p[k] for j,p in enumerate(source_probs) if j!=c) for k in range(2)]
                    events=[];curve=[];saved={};w=sim.initial[c].clone();limit=160 if method=='random' else 240
                    for t in range(limit):
                        w,e=sim.step(w,ids,c,t);events.append(e)
                        if (t+1)%40==0:
                            p=sim.probs(w);m=sim.metric([(p[k]+other[k])/4 for k in range(2)])
                            curve.append({'round':t+1,**m,'cost':total(events)});saved[t+1]=w.clone()
                            if method!='random' and m['dev']>=targets[u]:break
                    if method=='random':targets[u]=curve[-1]['dev']-.01
                    eligible=[v for v in curve if v['dev']>=targets[u]];selected=eligible[0] if eligible else curve[-1]
                    chosen=saved[selected['round']];error=None
                    if u==0:
                        ref=sim.initial[c].clone()
                        for t in range(selected['round']):ref,_=sim.step(ref,ids,c,t)
                        error=float((ref-chosen).abs().max());assert error<=1e-6
                    assert all(torch.equal(a,b) for a,b in zip(vectors,unchanged))
                    row={'client':u,'shard':c,'size_before':len(sim.groups[c]),'matched':bool(eligible),'target_dev':targets[u],'selected':selected,'reference_maxabs':error,'unchanged_source':True}
                    dump(f'{seed}_{method}_delete_{u}.json',{'summary':row,'curve':curve,'events':events})
                    dels.append(row);states[f'{seed}_{method}_delete_{u}']=chosen.cpu()
                    if (u+1)%5==0:E.log(phase='deletion',seed=seed,method=method,completed=u+1)
                record={'seed':seed,'method':method,'groups':sim.groups,'sizes':list(map(len,sim.groups)),'routing_score':route['methods'][method]['score'],'source_metrics':metrics,'source_cost':sourcecost,'deletions':dels,'D':sim.D,'model_and_initial_bytes':5*4*sim.D}
                records.append(record);dump(f'{seed}_{method}_summary.json',record);states[f'{seed}_{method}_source']=[w.cpu() for w in vectors]
                E.log(phase='condition_complete',seed=seed,method=method,source_test=metrics['test'],matched=sum(d['matched'] for d in dels))
    torch.cuda.synchronize();resource={'seconds':time.perf_counter()-start,**COUNT,**meter.report()}
    dump('results.json',records);dump('resource.json',resource);E.save(R/'states.pt',states);E.log(phase='complete',seconds=resource['seconds'])
if __name__=='__main__':run()
