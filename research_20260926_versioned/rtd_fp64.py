"""Corrected, matched communication arithmetic. Preserve all initial RTD records."""
from benchmark import *


def targets(qs,seed,noise,kind,reps=1,indices=None):
    arr=[qs[i] if indices is None else qs[i][indices] for i in range(1,10)]
    n=len(arr[0]);cost=Ledger();cost.dl(n*32+32);cost.v['forward_samples']+=9*n
    q=helmert(dtype=torch.float64)
    shared=torch.randn((n,10),device=DEV,dtype=torch.float64,generator=rng(seed*3000+max(noise,0)))
    if kind=='digital8':
        v=torch.stack([(a*255).round()/255 for a in arr]).mean(0)
        v=v/v.sum(1,keepdim=True)
        cost.v['digital_ul_bits']+=9*n*10*8;cost.v['pilot_real']+=72;cost.v['metadata_bits']+=288
        return v.float(),cost,{}
    if kind=='V1':payload=[a@q for a in arr];nz=shared@q
    elif kind=='centered':payload=[a-.1 for a in arr];nz=shared
    else:payload=arr;nz=shared
    received,power=transmit(payload,[1/9]*9,list(range(1,10)),seed*3000+max(noise,0),reps,
                            snr=None if noise==-1 else 20,ledger=cost,noise=nz)
    if kind=='V1':received=.1+received@q.T
    elif kind=='centered':received=received+.1
    cost.v['metadata_bits']+=9*32
    return simplex(received).float(),cost,power


def main():
    for seed in SEEDS:
        path=OUT/f'RTDfp64_seed{seed}.json'
        if path.exists() and json.loads(path.read_text()).get('complete'):continue
        begin=time.perf_counter();data=cm.load_data(seed)
        initial=json.loads((OUT/f'RTD_seed{seed}.json').read_text());assert initial['complete']
        qs=[]
        for i in range(10):
            model=cm.load_model(CP/f'teacher_seed{seed}_client{i}.pt')
            q=cm.probabilities(model,data['x'][data['public']],2.).double()
            qs.append(q/q.sum(1,keepdim=True))
        allq=torch.stack(qs).mean(0).float();qr=torch.stack(qs[1:]).mean(0).float()
        source,ssec=distill(data,seed,allq);ref,rsec=distill(data,seed,qr)
        cm.save_model(CP/f'rtd_fp64_reference_seed{seed}.pt',ref)
        _,sp=cm.evaluate(source,ref,data);rows=[]
        def record(name,version,m,cost,noise=None,extra=None):
            met,_=cm.evaluate(m,ref,data,sp)
            rows.append(dict(method=name,version=version,noise=noise,metrics=met,cost=cost,extra=extra))
        record('No-op','baseline',source,Ledger().report())
        rc=Ledger();rc.packet(20000,9,1);rc.dl(2000*32+32+63562*32);rc.v['student_samples']+=38400
        record('Target-free KD','reference',ref,rc.report(),extra={'student_seconds':rsec})
        for kind in ['Original','V1']:
            target,cost,power=targets(qs,seed,-1,kind)
            delta=float((target-qr).abs().max());different=int((target!=qr).sum())
            assert delta<1e-7
            m,sec=distill(data,seed,target)
            cost.dl(63562*32);cost.v['student_samples']+=38400
            record('RTD-Air-noiseless',kind,m,cost.report(),-1,
                dict(target_max_error=delta,target_different_values=different,
                     parameter_max_error=float((cm.flat(m)-cm.flat(ref)).abs().max()),student_seconds=sec))
        for noise in range(4):
            cases=[('RTD-Air','Original',1,None),('RTD-Air','Original',4,None),('RTD-Air','V1',1,None),('RTD-Air','V1',4,None),
                   ('Centered-full10','centered',1,None)]
            subset=torch.randperm(2000,device=DEV,generator=rng(19523))[:1000]
            cases.append(('RTD-Air-query1000','V1',1,subset))
            if noise==0:cases.append(('Digital8','digital8',1,None))
            for name,kind,reps,indices in cases:
                target,cost,power=targets(qs,seed,noise,kind,reps,indices)
                model,sec=distill(data,seed,target,indices)
                cost.dl(63562*32);cost.v['student_samples']+=38400
                true=qr if indices is None else qr[indices]
                record(name,kind if kind in ['Original','V1'] else 'baseline',model,cost.report(),noise,
                    dict(repeats=reps,student_seconds=sec,target_mse=float((target-true).square().mean()),power=power))
            save(path,dict(seed=seed,rows=rows,complete=False))
            log(stage='3-fp64',seed=seed,noise=noise,status='comparison_batch_complete')
        plainref=cm.load_model(CP/f'seed{seed}_plain_retain.pt');cross,_=cm.evaluate(ref,plainref,data)
        out={k:v for k,v in initial.items() if k not in ['rows','seconds','cross_reference_metrics']}
        out.update(rows=rows,complete=True,seconds=time.perf_counter()-begin,cross_reference_metrics=cross,
            source_student_seconds=ssec,numerical_addendum_sha256=hashlib.sha256((ROOT/'RTD_NUMERICAL_ADDENDUM_KO.md').read_bytes()).hexdigest(),
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),arithmetic='FP64 normalized prediction/OTA; FP32 student')
        save(path,out);log(stage='3-fp64',seed=seed,status='complete',seconds=out['seconds'])


if __name__=='__main__':main()
