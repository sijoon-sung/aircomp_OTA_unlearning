"""Privacy/communication sweep with integer frames and a target-only baseline."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import time
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
DEP=ROOT.parent/'research_20260928_cr_realdata/run_experiment.py'
spec=importlib.util.spec_from_file_location('frozen_cr_helpers',DEP)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
OUT=ROOT/'results'
D,C,Q=base.D,base.C,base.Q
DEV,DT=base.DEV,base.DT
CH,MC=16,8
RHOS=[.5,1.,2.,4.,8.]
SEEDS=[202609293,202609294,202609295]
METHODS=['Independent','CR-Fixed','CR-Power','CR-Uses','A-Only']
CONDITIONS=['unit','rayleigh','csi5']
KS=[10,100,1000]
HI=math.nextafter(math.sqrt(.5),0.)
RATE=base.RATE
CP=math.ceil(Q/8)
sha,write,normal=base.sha,base.write,base.normal


def eps(rho):return rho+2*math.sqrt(rho*math.log(1e5))


def design(method,eh,var):
    if method=='A-Only':
        beta=torch.ones(CH,device=DEV,dtype=DT)
        coef=torch.zeros_like(eh);coef[:,0]=-1
        return beta,coef,torch.full((CH,),var,device=DEV,dtype=DT),2*var
    if method=='CR-Uses':
        aa=1/eh[:,0];ar=(1/eh[:,1:]).max(-1).values
        beta=(ar/(aa+ar)).clamp(0,HI)
    else:beta=base.beta_for(method,eh)
    coef=(1-beta[:,None])*torch.ones_like(eh);coef[:,0]=-beta
    return beta,coef,(1-beta.square())*var,var


def frame_ledger(rad):
    n=rad['active'];r=rad['repeats']
    payload=Q*r;payload_total=(Q+CP)*r
    pilot_ul=9*n;pilot_dl=torch.full_like(n,9)
    bits=480+64*n+D*C*32
    digit=torch.ceil(bits/RATE);digit_prefix=torch.ceil(digit/8)
    digit_total=digit+digit_prefix
    total=payload_total+pilot_ul+pilot_dl+digit_total
    # Nominal payload CP energy is proportional to length; radio simulation
    # also audits actual cyclic-prefix energy on explicit samples separately.
    ep=rad['payload_energy']*(Q+CP)/Q
    ep_bound=rad['payload_energy_bound']*(Q+CP)/Q
    energy=ep+pilot_ul+pilot_dl+digit_total
    exposure=n*payload_total
    result=dict(repeats=r,active=n,payload_data_real=payload,payload_real_with_prefix=payload_total,
                ul_pilot_real=pilot_ul,dl_pilot_real=pilot_dl,control_and_head_bits=bits,
                digital_data_real=digit,digital_prefix_real=digit_prefix,total_real=total,
                deletion_ms_at_1M=total/1000,deletion_ms_at_100k=total/100,
                deletion_ms_at_10M=total/10000,payload_energy=ep,payload_energy_bound=ep_bound,
                accounted_tx_energy=energy,payload_client_active_real=exposure,
                max_block_energy=rad['max_block_energy'])
    for kappa in [0.,1e-6,1e-4,.01]:
        result['tx_plus_payload_circuit_'+str(kappa)]=energy+kappa*exposure
    return result


@torch.no_grad()
def verify_waveforms():
    assert torch.cuda.is_available()
    # This explicit-symbol check uses synthetic unit-norm statistics, not images.
    s=normal((1000,Q),51000);s=s/s.norm(dim=-1,keepdim=True)
    checks=[];generator=base.gen(51001)
    for kind in ['unit','rayleigh']:
        _,h,_,eh=base.channels(kind,1000,51002)
        for rho in [.5,2.,8.]:
            var=4/rho
            for method in METHODS[:4]:
                b,coef,vnew,vfinal=design(method,eh,var)
                mean,rad=base.radio(s,coef,h,eh,vnew)
                idx=0;r=int(rad['repeats'][idx]);scale=float(rad['scale'][idx])
                tx=coef[idx,:,None]*s/(scale*eh[idx,:,None])
                clean=(h[idx,:,None]*tx).sum(0)
                clean_frame=torch.cat((clean[-CP:],clean))
                total=torch.zeros((64,Q),device=DEV,dtype=DT);count=0
                for offset in range(0,r,256):
                    n=min(256,r-offset)
                    rx=clean_frame+math.sqrt(base.NU2)*torch.randn((64,n,Q+CP),device=DEV,dtype=DT,generator=generator)
                    count+=rx.numel()//64
                    total+=rx[:,:,CP:].sum(1)
                average=scale*total/r
                empirical=float((average-mean[idx]).square().mean())
                expected=float(vnew[idx]);relative=abs(empirical/expected-1)
                assert count==r*(Q+CP) and relative<.06
                # Energy carried by the actual copied prefix, unlike the length proxy.
                prefix_energy=float((tx[:,-CP:].square().sum()+tx.square().sum())*r)
                rho0=2/var
                ra=rho0/(1-float(b[idx])**2);rr=2*rho0/(1+float(b[idx]))
                assert max(ra,rr)<=rho+1e-9
                checks.append(dict(channel=kind,rho_cap=rho,method=method,repeats=r,
                                   counted_payload_real=count,expected_payload_real=r*(Q+CP),
                                   empirical_noise_variance=empirical,expected_noise_variance=expected,
                                   relative_variance_error=relative,actual_cyclic_prefix_payload_energy=prefix_energy))
    # The target-only baseline's exact joint covariance uses final variance 2v.
    for rho in RHOS:
        var=4/rho
        cov=torch.tensor([[var,var],[var,2*var]],device=DEV,dtype=DT)
        inv=torch.linalg.inv(cov)
        da=torch.tensor([2.,0.],device=DEV,dtype=DT);dr=torch.tensor([2.,2.],device=DEV,dtype=DT)
        assert abs(float(.5*da@inv@da)-rho)<1e-10
        assert abs(float(.5*dr@inv@dr)-rho/2)<1e-10
    out=dict(passed=True,checks=checks,source_sha256=sha(__file__),dependency_sha256=sha(DEP),
             protocol_sha256=sha(ROOT/'PROTOCOL_KO.md'))
    write(OUT/'waveform_checks.json',out)
    print('Waveform/covariance checks passed:',len(checks),flush=True)


@torch.no_grad()
def run(paths):
    audit=json.loads((OUT/'waveform_checks.json').read_text())
    assert audit['passed'] and audit['source_sha256']==sha(__file__)
    assert audit['dependency_sha256']==sha(DEP) and audit['protocol_sha256']==sha(ROOT/'PROTOCOL_KO.md')
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();totalrows=0;dh={}
    for di,(name,path) in enumerate(paths.items()):
        x,y,xt,yt=base.dataset(path)
        dh[name]={p.name:sha(p) for p in sorted(path.glob('*ubyte'))}
        proj=normal((784,D-1),2026092890).float()/math.sqrt(784)
        z=torch.cat((F.relu((x-.5)@proj),torch.ones(len(x),1,device=DEV)),1).to(DT)
        zt=torch.cat((F.relu((xt-.5)@proj),torch.ones(len(xt),1,device=DEV)),1).to(DT)
        del x,xt
        for seed in SEEDS:
            for k in KS:
                t0=time.perf_counter();rows=[]
                ids=base.partition(y,seed,k);s,raw,clip=base.stats(z,y,ids)
                sr=s[1:].sum(0);wt=base.decode(sr,k)
                ref=dict(clean_accuracy=float(base.accuracy(wt,zt,yt)),public_accuracy=float((yt==0).double().mean()),
                         client_samples=ids.shape[1],deleted_fraction=1/k,**clip)
                seed0=seed*10000+di*1000+k
                n0=normal((CH,MC,Q),seed0+601);n1=normal((CH,MC,Q),seed0+602)
                for ci,condition in enumerate(CONDITIONS):
                    h0,h1,e0,e1=base.channels(condition,k,seed0+ci*11)
                    g0,g1=h0/e0,h1/e1
                    radio_ref=g1[:,1:]@s[1:]
                    for rho in RHOS:
                        var=4/rho;rho0=2/var
                        m0,r0=base.radio(s,torch.ones_like(h0),h0,e0,torch.full((CH,),var,device=DEV,dtype=DT))
                        prep=frame_ledger(r0)
                        src_prefix=r0['repeats']*(s[None,:,-CP:]/(r0['scale'][:,None,None]*e0[:,:,None])).square().sum((1,2))
                        prep_energy=prep['accounted_tx_energy']+r0['payload_energy']+src_prefix-prep['payload_energy']
                        z0=math.sqrt(var)*n0
                        source=m0[:,None,:]+z0;w0=base.decode(source,k)
                        noop_acc=base.accuracy(w0[:,:2],zt,yt).reshape(CH,2).mean(-1)
                        w0cf=base.decode(source-g0[:,0,None,None]*s[0][None,None,:],k)
                        src_impact=(w0-w0cf).square().mean((-1,-2))
                        for method in METHODS:
                            beta,coef,vnew,vfinal=design(method,e1,var)
                            mu,rad=base.radio(s,coef,h1,e1,vnew)
                            mean=beta[:,None]*m0+mu
                            noise=beta[:,None,None]*z0+vnew.sqrt()[:,None,None]*n1
                            w=base.decode(mean[:,None,:]+noise,k)
                            same_noise_ref=base.decode(sr[None,None,:]+noise,k)
                            cf_coef=beta*g0[:,0]+coef[:,0]*g1[:,0]
                            wc=base.decode(mean[:,None,:]+noise-cf_coef[:,None,None]*s[0][None,None,:],k)
                            cf_mse=(w-wc).square().mean((-1,-2))
                            rel=((w-same_noise_ref).square().sum((-1,-2))/same_noise_ref.square().sum((-1,-2)).clamp_min(1e-30)).sqrt().max(-1).values
                            acc=base.accuracy(w[:,:2],zt,yt).reshape(CH,2)
                            # Observe source and raw update: noises are independent there.
                            ri=2*g0.square()/var+2*coef.square()*g1.square()/vnew[:,None]
                            d2=ri[:,0]*float(s[0].square().sum())/2
                            kl=(mean-sr).square().sum(-1)/(2*vfinal)
                            # Compare A-Only to CR's original (smaller-variance) reference too.
                            kl_original=.5*((mean-sr).square().sum(-1)/var+Q*(vfinal/var-1-math.log(vfinal/var)))
                            cost=frame_ledger(rad)
                            # Replace prefix-length energy proxy with actual transmitted prefix energy.
                            extra=rad['repeats']*(coef[:,:,None]*s[None,:,-CP:]/(rad['scale'][:,None,None]*e1[:,:,None])).square().sum((1,2))
                            actual_payload=rad['payload_energy']+extra
                            energy_correction=actual_payload-cost['payload_energy']
                            cost['payload_energy_length_proxy']=cost['payload_energy'];cost['payload_energy']=actual_payload
                            cost['accounted_tx_energy']+=energy_correction
                            for kapp in [0.,1e-6,1e-4,.01]:cost['tx_plus_payload_circuit_'+str(kapp)]+=energy_correction
                            # Public bound for data+prefix: copied prefix norm cannot exceed full norm.
                            cost['payload_energy_bound_conservative']=2*rad['payload_energy_bound']
                            assert bool((actual_payload<=cost['payload_energy_bound_conservative']*(1+1e-10)+1e-12).all())
                            values=dict(beta=beta,accuracy=acc.mean(-1),accuracy_noise_sd=acc.std(-1),source_accuracy=noop_acc,
                                rho_A=ri[:,0],rho_all=ri.max(-1).values,oracle_AUC=base.phi(torch.sqrt(d2/2)),
                                statistic_KL_matched_variance=kl,statistic_KL_original_reference=kl_original,
                                statistic_KL_radio_reference=(mean-radio_ref).square().sum(-1)/(2*vfinal),
                                coupled_head_relative_error_max=rel,counterfactual_A_impact_mse=cf_mse.mean(-1),
                                source_A_impact_mse=src_impact.mean(-1),counterfactual_A_ratio=(cf_mse/src_impact.clamp_min(1e-30)).mean(-1),
                                A_residual_coefficient=cf_coef,**{'cost_'+a:b for a,b in cost.items()},
                                prep_real=prep['total_real'],prep_accounted_tx_energy=prep_energy,
                                lifecycle_real=prep['total_real']+cost['total_real'],
                                lifecycle_accounted_tx_energy=prep_energy+cost['accounted_tx_energy'])
                            if condition!='csi5':
                                assert float(kl.max())<1e-18 and float(rel.max())<1e-9
                                assert float(ri.max())<=rho+1e-9
                                assert float(cf_mse.max())<1e-18
                            col={a:b.cpu().tolist() for a,b in values.items()}
                            for j in range(CH):
                                row=dict(dataset=name,seed=seed,K=k,condition=condition,rho_cap=rho,epsilon_cap=eps(rho),
                                         source_variance=var,final_variance=vfinal,method=method,channel_draw=j)
                                row.update({a:b[j] for a,b in col.items()});rows.append(row)
                obj=dict(references=ref,rows=rows,elapsed_seconds=time.perf_counter()-t0,
                         source_sha256=sha(__file__),dependency_sha256=sha(DEP),protocol_sha256=sha(ROOT/'PROTOCOL_KO.md'))
                write(OUT/f'{name}_seed{seed}_K{k}.json',obj);totalrows+=len(rows)
                print(f'{name} seed={seed} K={k}: {len(rows)} rows, {obj["elapsed_seconds"]:.2f}s',flush=True)
        del z,zt,y,yt
    torch.cuda.synchronize()
    write(OUT/'completion.json',dict(completed=True,rows=totalrows,expected_rows=2*3*3*3*5*5*CH,
          elapsed_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
          gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
          source_sha256=sha(__file__),dependency_sha256=sha(DEP),protocol_sha256=sha(ROOT/'PROTOCOL_KO.md'),
          data_hashes=dh,seeds=SEEDS,channel_draws=CH,noise_draws=MC,utility_noise_draws=2))
    print('COMPLETE',totalrows,round(time.perf_counter()-start,2),'s',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--math',action='store_true')
    ap.add_argument('--fashion',type=Path);ap.add_argument('--mnist',type=Path);args=ap.parse_args()
    if args.math:verify_waveforms()
    else:
        assert args.fashion and args.mnist
        run({'FashionMNIST':args.fashion,'MNIST':args.mnist})
