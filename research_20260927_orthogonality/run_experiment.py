"""Frozen projection ablation. One CUDA worker; no previous artifacts overwritten."""
from pathlib import Path
import sys, json, math, time, hashlib, argparse
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'research_20260926_versioned'))
import benchmark as b
torch, F, cm, rel=b.torch,b.F,b.cm,b.rel
OUT=ROOT/'results'; CP=ROOT/'checkpoints'
OUT.mkdir(exist_ok=True);CP.mkdir(exist_ok=True)
SEEDS=[381,382,383]; BETAS=[0.,.5,1.]; CLIENTS=list(range(1,10))+[0]
save=cm.write; log=cm.log; rng=cm.rng


class Ledger(b.Ledger):
    def report(self,snr=20):
        r=super().report();rate=.5*math.log2(1+10**(snr/10))
        r['total_real_uses_at_20db']=r['total_real_uses']
        r['total_real_uses']=1.125*(self.v['ul_real']+self.v['pilot_real']+
            (self.v['dl_bits']+self.v['digital_ul_bits']+self.v['metadata_bits'])/rate)
        r['rate_bits_per_real_use']=rate;r['cost_snr_db']=snr
        return r


def math_checks():
    dtype=torch.float64;device='cuda'
    rows=torch.randn(5,31,device=device,dtype=dtype,generator=rng(21))
    unit,_=rel.normalized_rows(rows);w=rel.relation_coefficients(unit)
    projection=unit[-1]-w@unit
    errors=[]
    for beta in BETAS:
        v=unit[-1]-beta*projection
        errors.append(float((unit[:-1]@v-(1-beta)*(unit[:-1]@unit[-1])).abs().max()))
    assert max(errors)<1e-12
    g_a=torch.tensor([-9.,0.],device=device,dtype=dtype)
    g_r=torch.tensor([1.,0.],device=device,dtype=dtype)
    source=torch.tensor([1.,0.],device=device,dtype=dtype)
    toy=[]
    for beta in BETAS:
        direction=g_a-beta*torch.dot(g_a,g_r)/g_r.square().sum()*g_r
        new=source+direction/9
        toy.append(dict(beta=beta,direction=direction.tolist(),new_model=new.tolist(),
                        squared_deletion_error_ratio=float(new.square().sum())))
    received,power=b.transmit(list(unit),w.tolist(),list(range(5)),21,snr=None)
    mac_error=float((received-w@unit).abs().max());assert mac_error<1e-12
    normal=rel.rescale_fedosd(w@unit,torch.tensor(3.,device=device,dtype=dtype))
    assert abs(float(normal.norm())-3)<1e-12
    report=dict(passed=True,projection_identity_max_errors=errors,mac_max_error=mac_error,
                toy_quadratic=toy,power=power,interpretation='CE stationary counterexample, not a theorem for UCE')
    save(OUT/'math_checks.json',report);log(stage='math',**report)


def training(data,seed,tag,exclude=False,offset=0):
    path=CP/f'seed{seed}_{tag}.pt';meta=OUT/f'seed{seed}_{tag}_training.json'
    if path.exists() and meta.exists():return cm.load_model(path),json.loads(meta.read_text())
    model,cost,seconds=cm.train_fl(data,seed,150,ortho=False,exclude=exclude,sampling_offset=offset)
    cm.save_model(path,model)
    info=dict(cost=cost,seconds=seconds,parameter_sha256=cm.vec_hash(cm.flat(model)),
              sampling_offset=offset,exclude=exclude)
    save(meta,info);return model,info


class Evaluator:
    def __init__(self,source,refs,data):
        self.data=data;self.source=cm.flat(source);self.refs=[cm.flat(r) for r in refs]
        self.sets={'forget':(data['x'][data['forget']],data['y'][data['forget']]),
                   'retain':(data['x'][data['retained'][::4]],data['y'][data['retained'][::4]]),
                   'test':(data['tx'],data['ty'])}
        self.ref_probs=[{name:cm.probabilities(r,x) for name,(x,y) in self.sets.items()} for r in refs]
        self.src_probs={name:cm.probabilities(source,x) for name,(x,y) in self.sets.items()}
        self.denoms=[{name:float(cm.js(self.src_probs[name],rp[name]).mean()) for name in self.sets} for rp in self.ref_probs]
        self.reference_difference={name:float(cm.js(self.ref_probs[0][name],self.ref_probs[1][name]).mean()) for name in self.sets}

    @torch.no_grad()
    def evaluate(self,model):
        probs={name:cm.probabilities(model,x) for name,(x,y) in self.sets.items()}
        result={name+'_accuracy':float((probs[name].argmax(1)==y).float().mean()) for name,(x,y) in self.sets.items()}
        for i,rp in enumerate(self.ref_probs):
            for name in self.sets:
                js=float(cm.js(probs[name],rp[name]).mean())
                result[f'{name}_js_ref{i}']=js
                result[f'{name}_normalized_js_ref{i}']=js/max(self.denoms[i][name],1e-12)
            result[f'parameter_ratio_ref{i}']=float((cm.flat(model)-self.refs[i]).square().sum()/(self.source-self.refs[i]).square().sum())
        yf=self.sets['forget'][1];p=probs['forget'];ix=torch.arange(len(yf),device='cuda')
        nm=self.data['nonmember'];qn=cm.probabilities(model,self.data['tx'][nm]);yn=self.data['ty'][nm]
        result['loss_mia_auc']=cm.auc_score(p[ix,yf].clamp_min(1e-12).log(),qn[ix,yn].clamp_min(1e-12).log())
        result['parameter_change_norm']=float((cm.flat(model)-self.source).norm())
        return result


def full_local_grad(model,data,client,target=False):
    ids=data['pools'][client];grad=torch.zeros_like(cm.flat(model))
    for part in ids.split(256):
        grad+=b.model_grad(model,data['x'][part],data['y'][part],target=target)*(len(part)/len(ids))
    return grad


def geometry(model,data):
    gs=torch.stack([full_local_grad(model,data,i) for i in range(10)])
    gu=full_local_grad(model,data,0,target=True);gr=gs[1:].mean(0)
    def cos(x,y):return float(F.cosine_similarity(x[None],y[None]))
    out=dict(full_ce_stationarity_ratio=float((.1*gs[0]+.9*gr).norm()/(.1*gs[0].norm()+.9*gr.norm())),
             ce_A_vs_retained_mean_cosine=cos(gs[0],gr),uce_A_vs_retained_mean_cosine=cos(gu,gr))
    for name,g in [('ce',gs[0]),('uce',gu)]:
        u,n=rel.normalized_rows(torch.cat((gs[1:],g[None])))
        w=rel.relation_coefficients(u)
        out[name+'_span_residual_fraction']=float((w@u).norm())
        out[name+'_aggregate_residual_fraction']=float((g-torch.dot(g,gr)/gr.square().sum()*gr).norm()/g.norm())
    return out


def run_case(source,data,evaluator,seed,beta,mode,noise=-1,raw=False):
    model=cm.clone_model(source);dim=cm.flat(model).numel();ledger=Ledger()
    streams={i:rng(seed*8100+i) for i in range(10)};history=[];rows=[]
    cost_snr=40 if mode=='ota40' else 20
    if mode!='exact':ledger.dl(32)  # fixed public SRHT seed throughout this transcript
    start=time.perf_counter()
    for step in range(10):
        ledger.dl(dim*32)
        grads=[b.model_grad(model,*cm.batch(data,i,streams[i],256)) for i in range(1,10)]
        grads.append(b.model_grad(model,*cm.batch(data,0,streams[0],256),target=True))
        unit,norms=rel.normalized_rows(torch.stack(grads))
        # Full-vector calculations below are oracle diagnostics, never server inputs for sketch modes.
        ew=rel.relation_coefficients(unit);ew[:-1]*=beta;exact=ew@unit
        if mode=='exact':
            weights=ew;ideal=exact;got=ideal
            ledger.v['digital_ul_bits']+=10*dim*32;ledger.v['pilot_real']+=80
            power=dict(variance=0.,max_average_power=0.,max_peak_power=0.)
            mac_error=0.
        else:
            sketches=rel.quantize_rows(rel.srht(unit,2048,seed*100),16)
            weights=rel.relation_coefficients(sketches);weights[:-1]*=beta
            ideal=weights@unit
            snr=None if mode=='sketch' else int(mode[3:])
            got,power=b.transmit(list(unit),weights.tolist(),CLIENTS,seed*20000+max(noise,0)*100+step,
                                 snr=snr,ledger=ledger)
            check,_=b.transmit(list(unit),weights.tolist(),CLIENTS,0,snr=None)
            mac_error=float((check-ideal).abs().max())
            assert mac_error<2e-6
            ledger.v['digital_ul_bits']+=10*(2048*16+32) # quantizer scale per row
            ledger.v['metadata_bits']+=10*32+32 # peak/RMS scale metadata plus target norm
            ledger.dl(10*32+32) # broadcast weights plus physical receive scale
        ledger.v['gradient_samples']+=2560
        correction=got*norms[-1] if raw else rel.rescale_fedosd(got,norms[-1])
        unit_got=got/got.norm().clamp_min(1e-20)
        history.append(dict(step=step+1,ideal_residual_norm=float(ideal.norm()),exact_residual_norm=float(exact.norm()),
            max_abs_retained_cosine=float((unit[:-1]@unit_got).abs().max()),
            received_vs_ideal_cosine=float(F.cosine_similarity(got[None],ideal[None])),
            sketch_vs_exact_cosine=float(F.cosine_similarity(ideal[None],exact[None])),
            noise_to_signal_norm=float((got-ideal).norm()/ideal.norm().clamp_min(1e-20)),
            target_norm=float(norms[-1]),step_norm=float(.01*correction.norm()),
            coefficient_max=float(weights.abs().max()),active_clients=int((weights.abs()>1e-12).sum()),
            power=power,mac_max_error=mac_error))
        cm.assign(model,cm.flat(model)-.01*correction)
        if step in [0,9]:
            led=Ledger();led.v=ledger.v.copy();led.dl(dim*32) # deliver resulting model
            rows.append(dict(seed=seed,beta=beta,mode=mode,noise=noise,normalization='raw' if raw else 'target_norm',
                step=step+1,metrics=evaluator.evaluate(model),cost=led.report(cost_snr),
                first_geometry=history[0],last_geometry=history[-1],
                privacy_status='full individual input comparator' if mode=='exact' else
                    ('A-only direction exposed' if beta==0 else 'sketch and aggregate disclosure; no privacy proof')))
    cm.sync();elapsed=time.perf_counter()-start
    tag=f'seed{seed}_{mode}_beta{beta:g}_noise{noise}_'+('raw' if raw else 'targetnorm')
    cm.save_model(CP/(tag+'.pt'),model)
    save(OUT/(tag+'.json'),dict(complete=True,rows=rows,history=history,seconds=elapsed))
    log(stage='case',seed=seed,beta=beta,mode=mode,noise=noise,raw=raw,seconds=round(elapsed,2))
    return rows


def main():
    assert torch.cuda.is_available()
    math_checks()
    env=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),one_worker=True,seeds=SEEDS,
         protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL_KO.md').read_bytes()).hexdigest(),
         code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(OUT/'environment.json',env)
    for seed in SEEDS:
        out=OUT/f'seed{seed}_complete.json'
        if out.exists():continue
        data=cm.load_data(seed);source,prep=training(data,seed,'source')
        ref0,r0=training(data,seed,'reference0',True)
        ref1,r1=training(data,seed,'reference1',True,900000)
        evaluator=Evaluator(source,[ref0,ref1],data)
        baseline=dict(source=evaluator.evaluate(source),reference0=evaluator.evaluate(ref0),
                      reference1=evaluator.evaluate(ref1),reference_difference=evaluator.reference_difference,
                      full_data_geometry=geometry(source,data))
        save(OUT/f'seed{seed}_baseline.json',baseline)
        rows=[]
        for mode in ['exact','sketch','ota20','ota40']:
            for beta in BETAS:
                for noise in (range(4) if mode.startswith('ota') else [-1]):
                    tag=f'seed{seed}_{mode}_beta{beta:g}_noise{noise}_targetnorm.json';path=OUT/tag
                    if path.exists():rows+=json.loads(path.read_text())['rows']
                    else:rows+=run_case(source,data,evaluator,seed,beta,mode,noise)
        for beta in BETAS:
            path=OUT/f'seed{seed}_exact_beta{beta:g}_noise-1_raw.json'
            if path.exists():rows+=json.loads(path.read_text())['rows']
            else:rows+=run_case(source,data,evaluator,seed,beta,'exact',raw=True)
        save(out,dict(complete=True,seed=seed,rows=rows,preparation=prep,reference_preparation=[r0,r1],baseline=baseline))
        log(stage='seed',seed=seed,status='complete',rows=len(rows))
    log(stage='experiment',status='complete')


if __name__=='__main__':main()
