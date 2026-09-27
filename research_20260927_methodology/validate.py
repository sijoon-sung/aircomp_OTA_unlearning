from pathlib import Path
import sys,json,math,hashlib,statistics as st,time
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path.insert(0,str(BASE/'research_20260926_versioned'))
import benchmark as b
import rtd_fp64 as kd
torch,cm,rel=b.torch,b.cm,b.rel
OLD=BASE/'research_20260926_versioned';ORTH=BASE/'research_20260927_orthogonality'
OUT=ROOT/'validation';OUT.mkdir(exist_ok=True)


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,list):
        for v in x:finite(v)


def audit_records():
    manifest={};counts={};datasets={}
    for directory in [OLD,ORTH]:
        e=read(directory/'results/environment.json')
        name='benchmark.py' if directory==OLD else 'run_experiment.py'
        assert e['protocol_sha256']==sha(directory/'PROTOCOL_KO.md')
        assert e['code_sha256']==sha(directory/name)
        manifest[str(directory/name)]=sha(directory/name)
    for family in ['DS','OG','RTDfp64']:
        recs=[read(p) for p in sorted((OLD/'results').glob(f'{family}_seed*.json'))]
        assert len(recs)==3 and all(r['complete'] for r in recs)
        finite(recs);datasets[family]=recs;counts[family]=sum(len(r['rows']) for r in recs)
    recs=[read(ORTH/f'results/seed{s}_complete.json') for s in [381,382,383]]
    assert all(r['complete'] for r in recs);finite(recs);datasets['partial']=recs;counts['partial']=sum(len(r['rows']) for r in recs)
    assert counts=={'DS':441,'OG':54,'RTDfp64':87,'partial':198}
    equalities=[]
    for r in datasets['RTDfp64']:
        assert r['code_sha256']==sha(OLD/'rtd_fp64.py')
        assert r['numerical_addendum_sha256']==sha(OLD/'RTD_NUMERICAL_ADDENDUM_KO.md')
        for row in r['rows']:
            if row['method']=='RTD-Air-noiseless':
                assert row['extra']['target_max_error']==0 and row['extra']['parameter_max_error']==0
                equalities.append((r['seed'],row['version']))
    assert len(equalities)==6
    stats={}
    def agg(label,family,predicate,metric):
        seedvals=[];cost=[];sizes=[]
        for rec in datasets[family]:
            rows=[r for r in rec['rows'] if predicate(r)]
            assert rows,label
            seedvals.append(st.mean(r['metrics'][metric] for r in rows));sizes.append(len(rows))
            cost.append(st.mean(r['cost']['total_real_uses'] for r in rows))
        stats[label]=dict(mean=st.mean(seedvals),sd=st.stdev(seedvals),seed_values=seedvals,
                          samples_per_seed=sizes,cost=st.mean(cost))
    for v in ['Original','V1']:
        agg('DS_'+v,'DS',lambda r:r['method']=='DS-Air' and r['version']==v and r['peak'] is True,'parameter_error_ratio')
    for name in ['Original-cost-matched','DS-Air-subspace48']:
        agg('DS_'+name,'DS',lambda r:r['method']==name,'parameter_error_ratio')
    for step in [1,10]:
        for mode in ['sketch','ota20','ota40']:
            for beta in [.5,1.]:
                agg(f'partial_{mode}_{step}_{beta}','partial',lambda r:r['mode']==mode and r['beta']==beta and
                    r['step']==step and r['normalization']=='target_norm','forget_normalized_js_ref0')
    for v in ['Original','V1']:
        for repeats in [1,4]:
            agg(f'RTD_{v}_{repeats}','RTDfp64',lambda r:r['method']=='RTD-Air' and r['version']==v and
                r['extra']['repeats']==repeats,'forget_normalized_js')
    return dict(passed=True,raw_rows=counts,code_manifest=manifest,existing_noiseless_KD_equalities=equalities,statistics=stats)


def ds_math():
    d=12;k=10;c=3;dtype=torch.float64;device='cuda'
    g=torch.Generator(device=device).manual_seed(927101)
    a=torch.randn(k,d,d,device=device,dtype=dtype,generator=g)
    hs=a.transpose(1,2)@a+torch.eye(d,device=device,dtype=dtype)
    cs=torch.randn(k,d,c,device=device,dtype=dtype,generator=g)
    w0=torch.linalg.solve(hs.mean(0),cs.mean(0));hr=hs[1:].mean(0);cr=cs[1:].mean(0)
    wr=torch.linalg.solve(hr,cr);bs=hs[1:]@w0-cs[1:];br=bs.mean(0);p=torch.linalg.inv(hr)
    v=p@br;pre=(p@bs).mean(0)
    ga=hs[0]@w0-cs[0];recovered=-9*hr@v
    local_inv=torch.linalg.solve(hs[1:],bs).mean(0)
    u=torch.linalg.qr(torch.randn(d,7,device=device,dtype=dtype,generator=g),mode='complete').Q
    kept=u[:,:7];null=u[:,7:8]@torch.ones(1,c,device=device,dtype=dtype)
    cp=cs.clone();cp[0]+=null;cp[1]-=null
    wp=torch.linalg.solve(hs.mean(0),cp.mean(0));bp=hs[1:]@wp-cp[1:]
    small=lambda bb: kept@torch.linalg.solve(kept.T@hr@kept,kept.T@bb.mean(0))
    out=dict(exact_delete_error=float((w0-v-wr).abs().max()),common_inverse_sum_error=float((pre-v).abs().max()),
        target_gradient_reconstruction_error=float((ga-recovered).abs().max()),
        local_inverse_averaging_difference=float((local_inv-v).norm()),
        subspace_same_source_error=float((wp-w0).abs().max()),
        subspace_same_projected_payload_error=float((kept.T@bp-kept.T@bs).abs().max()),
        subspace_same_correction_error=float((small(bp)-small(bs)).abs().max()),
        subspace_changed_target_gradient_norm=float(((hs[0]@wp-cp[0])-ga).norm()),
        full_dimension=d,subspace_dimension=7)
    for key,value in out.items():
        if key.endswith('_error'):assert value<1e-10,(key,value)
    assert out['local_inverse_averaging_difference']>.01 and out['subspace_changed_target_gradient_norm']>1
    return out


def projection_math():
    g=torch.Generator(device='cuda').manual_seed(927102)
    q=torch.linalg.qr(torch.randn(17,4,device='cuda',dtype=torch.float64,generator=g)).Q
    p=q@q.T;eye=torch.eye(17,device='cuda',dtype=torch.float64)
    u=torch.randn(17,device='cuda',dtype=torch.float64,generator=g)
    v=(eye-.5*p)@u;recovered=v+p@v
    out=dict(partial_operator_rank=int(torch.linalg.matrix_rank(eye-.5*p)),
             full_projection_rank=int(torch.linalg.matrix_rank(eye-p)),
             known_span_partial_inverse_error=float((recovered-u).abs().max()),
             dimension=17,retained_span_dimension=4)
    assert out['partial_operator_rank']==17 and out['full_projection_rank']==13
    assert out['known_span_partial_inverse_error']<1e-10
    return out


def replay_projection():
    rows=[]
    for seed in [381,382,383]:
        data=cm.load_data(seed);m=cm.load_model(ORTH/f'checkpoints/seed{seed}_source.pt')
        source=cm.clone_model(m)
        vs=[b.model_grad(m,*cm.batch(data,i,cm.rng(seed*8100+i),256),target=(i==0)) for i in list(range(1,10))+[0]]
        unit,norms=rel.normalized_rows(torch.stack(vs))
        sk=rel.quantize_rows(rel.srht(unit,2048,seed*100),16)
        weights=rel.relation_coefficients(sk);weights[:-1]*=.5
        got,power=b.transmit(list(unit),weights.tolist(),list(range(1,10))+[0],0,snr=None)
        cm.assign(m,cm.flat(m)-.01*rel.rescale_fedosd(got,norms[-1]))
        xf=data['x'][data['forget']]
        ref=cm.load_model(ORTH/f'checkpoints/seed{seed}_reference0.pt')
        qp=cm.probabilities(ref,xf);sp=cm.probabilities(source,xf);mp=cm.probabilities(m,xf)
        ratio=float(cm.js(mp,qp).mean()/cm.js(sp,qp).mean())
        old=read(ORTH/f'results/seed{seed}_sketch_beta0.5_noise-1_targetnorm.json')['rows'][0]
        err=abs(ratio-old['metrics']['forget_normalized_js_ref0'])
        norm_error=abs(float((weights@unit).norm())-old['first_geometry']['ideal_residual_norm'])
        assert err<1e-5 and norm_error<1e-7
        rows.append(dict(seed=seed,replayed_js_ratio=ratio,recorded_js_ratio=old['metrics']['forget_normalized_js_ref0'],
                         js_difference=err,residual_norm_difference=norm_error,power=power))
    return rows


def replay_kd():
    rows=[]
    for seed in [371,372,373]:
        data=cm.load_data(seed);qs=[]
        for i in range(10):
            model=cm.load_model(OLD/f'checkpoints/teacher_seed{seed}_client{i}.pt')
            q=cm.probabilities(model,data['x'][data['public']],2.).double();qs.append(q/q.sum(1,keepdim=True))
        qr=torch.stack(qs[1:]).mean(0);qa=torch.stack(qs).mean(0)
        leakage=float(((qa-.9*qr)/.1-qs[0]).abs().max());assert leakage<1e-10
        for kind in ['Original','V1']:
            received,cost,power=kd.targets(qs,seed,-1,kind)
            err=float((received-qr.float()).abs().max());assert err==0
            rows.append(dict(seed=seed,version=kind,noiseless_target_max_error=err,
                             teacher_prediction_reconstruction_max_error=leakage,power=power))
    return rows


if __name__=='__main__':
    start=time.perf_counter();assert torch.cuda.is_available()
    report=dict(record_audit=audit_records(),DS_math=ds_math(),projection_math=projection_math())
    report['projection_checkpoint_replay']=replay_projection()
    report['KD_teacher_checkpoint_replay']=replay_kd()
    report.update(passed=True,device=torch.cuda.get_device_name(),new_training_runs=0,
                  seconds=time.perf_counter()-start,code_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'VALIDATION_PLAN_KO.md'))
    cm.write(OUT/'verification.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
