from pathlib import Path
import time,json
import numpy as np
import torch
import run as M
R=M.R;E=M.E
def read(name):return json.loads((R/name).read_text(encoding='utf-8'))
def main():
    start=time.perf_counter();records=read('results.json');saved=torch.load(R/'states.pt',map_location='cpu',weights_only=False);out=[];audit=[]
    with E.Power() as meter:
        for seed in [62001,62002,62003]:
            data=E.load_data('FashionMNIST',.5,seed);sim=M.Sim(data,seed)
            for record in [r for r in records if r['seed']==seed]:
                method=record['method'];sim.groups=record['groups'];source=[v.to('cuda') for v in saved[f'{seed}_{method}_source']];pp=[sim.probs(v) for v in source]
                for d in record['deletions']:
                    u=d['client'];c=d['shard'];ids=[i for i in sim.groups[c] if i!=u];original=read(f'{seed}_{method}_delete_{u}.json');existing=next((p for p in original['curve'] if p['round']==160),None)
                    other=[sum(p[k] for j,p in enumerate(pp) if j!=c) for k in range(2)];w=None
                    if existing is None:
                        t0=d['selected']['round'];assert t0<160
                        w=saved[f'{seed}_{method}_delete_{u}'].to('cuda').clone();events=original['events'].copy();assert len(events)==t0
                        for t in range(t0,160):w,e=sim.step(w,ids,c,t);events.append(e)
                        probs=sim.probs(w);metrics=sim.metric([(probs[k]+other[k])/4 for k in range(2)])
                        result={'round':160,**metrics,'cost':M.total(events)}
                        M.dump(f'{seed}_{method}_fixed160_extension_{u}.json',{'from_round':t0,'events':events[t0:],'result':result})
                    else:result=existing
                    if u==0:
                        ref=sim.initial[c].clone()
                        for t in range(160):ref,_=sim.step(ref,ids,c,t)
                        prob=sim.probs(ref);mm=sim.metric([(prob[k]+other[k])/4 for k in range(2)])
                        err=max(abs(mm[k]-result[k]) for k in mm);assert err<=1e-7
                        pe=float((w-ref).abs().max()) if w is not None else None
                        if pe is not None:assert pe<=1e-6
                        audit.append({'seed':seed,'method':method,'client':u,'metric_error':err,'parameter_error_if_extended':pe})
                    out.append({'seed':seed,'method':method,'client':u,'size_before':len(sim.groups[c]),'metrics':result,'extended':existing is None})
                E.log(phase='fixed160',seed=seed,method=method)
    M.dump('fixed160_results.json',out);M.dump('fixed160_audit.json',audit);M.dump('fixed160_resource.json',{'seconds':time.perf_counter()-start,**M.COUNT,**meter.report()})
if __name__=='__main__':main()
