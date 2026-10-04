from pathlib import Path
import sys,time,json,hashlib,html
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'research_20261003_sisa_aircomp'))
import experiment as E

POL={'all':[0,0,0,0,0],'mild':[0,0,20,40,60],'late':[0,0,40,80,120]}
COUNTERS={'local_calls':0,'aggregate_rounds':0}
def dump(name,obj): E.dump(ROOT/name,obj)
def cost(events):
    keys=['total_re','total_re_dl6','ul_reals','dl_bits','pilot_reals','control_bits','local_calls','local_steps','ul_signal_energy']
    return {k:float(sum(e[k] for e in events)) for k in keys}
def step(sim,w,ids,c,t):
    u,event=sim.physical_receive(w,ids,c,t,0)
    COUNTERS['local_calls']+=len(ids);COUNTERS['aggregate_rounds']+=1
    return w+u,event
@torch.no_grad()
def evaluate(sim,states):
    out={}
    for name,x,y in [('dev',sim.data['x'][sim.data['dev']],sim.data['y'][sim.data['dev']]),('test',sim.data['tx'],sim.data['ty'])]:
        p=0
        for w in states:
            E.assign(sim.local,w);p=p+E.predict(sim.local,x)/4
        out[name]=float((p.argmax(1)==y).float().mean())
    return out
def train(sim,name):
    entry=POL[name];T=160 if name=='all' else 240
    states=[w.clone() for w in sim.initial];snaps={0:[w.clone() for w in states]};events=[];curve=[]
    for t in range(T):
        if t in entry:snaps[t]=[w.clone() for w in states]
        for c,g in enumerate(sim.groups):
            ids=[i for j,i in enumerate(g) if t>=entry[j]]
            states[c],event=step(sim,states[c],ids,c,t);events.append(event)
        if t+1>=160 and (t+1)%20==0:
            snaps[t+1]=[w.clone() for w in states]
            curve.append({'round':t+1,**evaluate(sim,states),'cost':cost(events)})
        if (t+1)%40==0:E.log(phase='source',seed=sim.seed,policy=name,round=t+1)
    dump(f'{sim.seed}_{name}_source.json',{'entry':entry,'groups':sim.groups,'base_h':sim.base_h.tolist(),'events':events,'curve':curve})
    return snaps,curve
def deletion(sim,name,snaps,T,j):
    entry=POL[name];deleted=sim.groups[0][j];start=entry[j]
    def replay(w,begin):
        ev=[]
        for t in range(begin,T):
            ids=[i for k,i in enumerate(sim.groups[0]) if i!=deleted and t>=entry[k]]
            w,e=step(sim,w,ids,0,t);ev.append(e)
        return w,ev
    w,events=replay(snaps[start][0].clone(),start)
    reference,refevents=replay(sim.initial[0].clone(),0)
    error=float((w-reference).abs().max());assert error<=1e-6,error
    final=[v.clone() for v in snaps[T]];final[0]=w
    metrics=evaluate(sim,final)
    row={'deleted':deleted,'entry_round':start,'replay_rounds':T-start,'maxabs_reference':error,**metrics,'cost':cost(events),'reference_cost':cost(refevents)}
    dump(f'{sim.seed}_{name}_delete_{deleted}.json',{'summary':row,'events':events})
    E.log(phase='deletion',seed=sim.seed,policy=name,client=deleted,error=error)
    return row,w.cpu()
def main():
    started=time.perf_counter();records=[];saved={}
    dump('provenance.json',{str(p):E.sha(p) for p in [ROOT/'run.py',ROOT/'PREREG.txt',E.ROOT/'experiment.py',ROOT.parent/'research_20260929_pdf_ota_unlearning/core.py']})
    with E.Power() as meter:
        for seed in [61001,61002,61003]:
            data=E.load_data('FashionMNIST',.5,seed);sim=E.Engine(data,seed,4,'random_fixed',240)
            dump(f'{seed}_partition.json',{'original_indices':data['original_indices'].tolist(),'pools':[p.tolist() for p in data['original_pools']],'groups':sim.groups})
            target=None
            for name in POL:
                snaps,curve=train(sim,name)
                if name=='all':target=curve[0]['dev']-.01
                eligible=[p for p in curve if p['dev']>=target]
                selected=eligible[0] if eligible else curve[-1];T=selected['round']
                dels=[]
                for j in range(5):
                    row,w=deletion(sim,name,snaps,T,j);dels.append(row);saved[f'{seed}_{name}_deleted_{j}']=w
                record={'seed':seed,'policy':name,'entry':POL[name],'matched':bool(eligible),'target_dev':target,'selected':selected,'at160':curve[0],
                    'deletions':dels,'expected_delete_re':float(np.mean([r['cost']['total_re'] for r in dels])),
                    'mean_deleted_test':float(np.mean([r['test'] for r in dels])),
                    'checkpoint_bytes':4*sim.D*4*len(set(POL[name])),'final_model_bytes':4*sim.D*4,'D':sim.D}
                records.append(record)
                saved[f'{seed}_{name}']={t:[v.cpu() for v in vs] for t,vs in snaps.items()}
                dump(f'{seed}_{name}_summary.json',record)
    resource={'seconds':time.perf_counter()-started,**COUNTERS,**meter.report()}
    dump('results.json',records);dump('resource.json',resource);E.save(ROOT/'states.pt',saved)
    E.log(phase='complete',seconds=resource['seconds'],calls=COUNTERS['local_calls'])
if __name__=='__main__':main()
