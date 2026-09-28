"""CR-Air bounded real-data evaluation. One CUDA process; no private artifacts saved."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
DEV,DT='cuda',torch.float64
SEEDS=[202609290,202609291,202609292]
KS=[10,100,1000]
CONDITIONS=['unit','rayleigh','weak_A','weak_R','csi5']
METHODS=['Independent','CR-Fixed','CR-Power','No-op']
D,C,B,VAR,NU2,LAM=16,10,1.,8.,.01,.01
CH,MC=16,8
QH=D*(D+1)//2
Q=QH+D*C
RHO0=.25
HI=math.nextafter(math.sqrt(.5),0.)
CP=1.125
RATE=.5*math.log2(101)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def gen(seed):
    return torch.Generator(device=DEV).manual_seed(int(seed))


def normal(shape,seed):
    return torch.randn(shape,device=DEV,dtype=DT,generator=gen(seed))


def dataset(path):
    def load(prefix):
        x=np.fromfile(path/(prefix+'-images-idx3-ubyte'),dtype=np.uint8,offset=16).copy().reshape(-1,784)
        y=np.fromfile(path/(prefix+'-labels-idx1-ubyte'),dtype=np.uint8,offset=8).copy()
        return torch.tensor(x,device=DEV,dtype=torch.float32)/255.,torch.tensor(y,device=DEV,dtype=torch.long)
    return (*load('train'),*load('t10k'))


def partition(y,seed,k):
    g=gen(seed);pools=[]
    for label in range(10):
        ix=torch.where(y==label)[0]
        pools.append(ix[torch.randperm(len(ix),device=DEV,generator=g)[:1000]])
    groups=k//10;n=500//groups
    clients=[]
    for label in range(10):
        for j in range(groups):
            clients.append(torch.cat((pools[label][j*n:(j+1)*n],pools[(label+1)%10][500+j*n:500+(j+1)*n])))
    ids=torch.stack(clients)
    assert ids.numel()==10000 and len(torch.unique(ids))==10000
    return ids


def stats(z,y,ids):
    zz=z[ids];yy=F.one_hot(y[ids],C).to(DT)
    h=zz.transpose(-1,-2)@zz/ids.shape[1]
    cc=zz.transpose(-1,-2)@yy/ids.shape[1]
    ix=torch.triu_indices(D,D,device=DEV)
    raw=torch.cat((h[:,ix[0],ix[1]],cc.flatten(1)),dim=1)
    norms=raw.norm(dim=1)
    clipped=raw/(norms/B).clamp_min(1)[:,None]
    assert float(clipped.norm(dim=1).max())<=B+1e-12
    return clipped,raw,dict(clipped_fraction=float((norms>B).double().mean()),
                            mean_raw_norm=float(norms.mean()),max_clipped_norm=float(clipped.norm(dim=1).max()))


def decode(v,k):
    ix=torch.triu_indices(D,D,device=DEV)
    h=torch.zeros((*v.shape[:-1],D,D),device=DEV,dtype=DT)
    h[...,ix[0],ix[1]]=v[...,:QH]/k
    h[...,ix[1],ix[0]]=v[...,:QH]/k
    ev,vec=torch.linalg.eigh(h)
    hh=(vec*ev.clamp_min(0).unsqueeze(-2))@vec.transpose(-1,-2)
    w=torch.linalg.solve(hh+LAM*torch.eye(D,device=DEV,dtype=DT),v[...,QH:].reshape(*v.shape[:-1],D,C)/k)
    return w.float().to(DT)


def accuracy(w,z,y):
    logits=torch.einsum('nd,mdc->mnc',z.float(),w.reshape(-1,D,C).float())
    return (logits.argmax(-1)==y[None]).double().mean(-1)


def channels(kind,k,seed):
    if kind in ['rayleigh','csi5']:
        h0=(normal((CH,k,2),seed).square().sum(-1)/2).sqrt()
        h1=(normal((CH,k,2),seed+1).square().sum(-1)/2).sqrt()
    else:
        h0=torch.ones((CH,k),device=DEV,dtype=DT);h1=h0.clone()
        if kind=='weak_A':h1[:,0]=.01
        if kind=='weak_R':h1[:,1]=.01
    e0=h0*torch.exp(.05*normal((CH,k),seed+2)) if kind=='csi5' else h0.clone()
    e1=h1*torch.exp(.05*normal((CH,k),seed+3)) if kind=='csi5' else h1.clone()
    return h0,h1,e0,e1


def beta_for(method,estimated):
    if method=='Independent':return torch.zeros(CH,device=DEV,dtype=DT)
    if method=='CR-Fixed':return torch.full((CH,),.5,device=DEV,dtype=DT)
    aa=B*B/estimated[:,0].square();rr=(B*B/estimated[:,1:].square()).sum(-1)
    return (2*rr/(2*rr+aa+torch.sqrt(aa*aa+4*aa*rr))).clamp(0,HI)


def radio(s,coef,h,estimated,variance):
    required=(coef.abs()*B/estimated).amax(1)
    repeats=torch.ceil(NU2*required.square()/variance).clamp_min(1)
    scale=torch.sqrt(repeats*variance/NU2)
    tx=coef[:,:,None]*s[None]/(scale[:,None,None]*estimated[:,:,None])
    mean=(h[:,:,None]*tx).sum(1)*scale[:,None]
    block=tx.square().sum(-1)
    actual_energy=block.sum(-1)*repeats
    upper=NU2/variance*((coef*B/estimated).square().sum(-1))
    assert float(block.max())<=1+1e-10
    assert bool((actual_energy<=upper*(1+1e-10)+1e-12).all())
    active=(coef!=0).sum(-1).to(DT)
    return mean,dict(repeats=repeats,scale=scale,payload_energy=actual_energy,
                     payload_energy_bound=upper,max_block_energy=block.max(-1).values,active=active)


def ledger(rad):
    active=rad['active'];rep=rad['repeats']
    ul_pilot=8*active
    feedback=64*active
    dl_real=8+(480+feedback+D*C*32)/RATE
    shared_payload=Q*rep
    total=CP*(shared_payload+ul_pilot+dl_real)
    ep=rad['payload_energy']
    obj=dict(repeats=rep,active_clients=active,payload_real=shared_payload,ul_pilot_real=ul_pilot,
             dl_real=dl_real,total_real=total,payload_energy=CP*ep,
             payload_energy_bound=CP*rad['payload_energy_bound'],max_block_energy=rad['max_block_energy'])
    for pilot in [0.,.001,.01,1.]:
        obj['accounted_tx_energy_pilot_'+str(pilot)]=CP*(ep+pilot*(ul_pilot+8)+(dl_real-8))
    return obj


def empty_ledger():
    zz=torch.zeros(CH,device=DEV,dtype=DT)
    rad={name:zz.clone() for name in ['active','repeats','payload_energy','payload_energy_bound','max_block_energy']}
    return {name:zz.clone() for name in ledger(rad)}


def phi(v):
    return .5*(1+torch.erf(v/math.sqrt(2)))


def evaluate(mean,noise,retained,radio_retained,source,source_w,cf_coef,source_cf_mse,beta,
             g0,g1,s,wt,zt,yt,k,method):
    val=mean[:,None,:]+noise
    w=decode(val,k)
    reference=retained[None,None,:]+noise
    wr=decode(reference,k)
    cf=decode(val-cf_coef[:,None,None]*s[0][None,None,:],k)
    nn=(w-wr).square().sum((-1,-2))
    coupled_rel=torch.sqrt(nn/(wr.square().sum((-1,-2)).clamp_min(1e-30)))
    clean_mse=(w-wt).square().mean((-1,-2))
    denom=(source_w-wt).square().mean((-1,-2)).clamp_min(1e-30)
    cf_mse=(w-cf).square().mean((-1,-2))
    acc=accuracy(w[:,:2],zt,yt).reshape(CH,2)
    if method=='No-op':
        rho_clients=RHO0*g0.square()
    else:
        c=torch.ones_like(g1)*(1-beta[:,None]);c[:,0]=-beta
        rho_clients=RHO0*(g0.square()+c.square()*g1.square()/(1-beta.square())[:,None])
    rr=dict(accuracy=acc.mean(-1),accuracy_noise_sd=acc.std(-1),
            clean_parameter_mse=clean_mse.mean(-1),normalized_clean_mse=(clean_mse/denom).mean(-1),
            coupled_model_relative_error_max=coupled_rel.max(-1).values,
            statistic_KL_ideal=(mean-retained).square().sum(-1)/(2*VAR),
            statistic_KL_radio_reference=(mean-radio_retained).square().sum(-1)/(2*VAR),
            counterfactual_A_model_mse=cf_mse.mean(-1),source_A_model_mse=source_cf_mse.mean(-1),
            counterfactual_A_influence_ratio=(cf_mse/source_cf_mse.clamp_min(1e-30)).mean(-1),
            A_residual_coefficient=cf_coef,A_source_coefficient=g0[:,0],
            rho_A=rho_clients[:,0],rho_retained_max=rho_clients[:,1:].amax(-1),
            rho_all=rho_clients.amax(-1),beta=beta)
    # Oracle attack: known actual s_A versus empty statistics; NOT a membership-inference experiment.
    d2=rho_clients[:,0]*float(s[0].square().sum())/(2*B*B)
    rr['oracle_binary_AUC']=phi(torch.sqrt(d2/2))
    return rr


@torch.no_grad()
def math_checks():
    assert torch.cuda.is_available()
    s=normal((10,Q),910)
    s=s/s.norm(dim=1,keepdim=True)
    h=torch.exp(normal((CH,10),911))
    beta=beta_for('CR-Power',h)
    coef=(1-beta[:,None])*torch.ones_like(h);coef[:,0]=-beta
    mean,rad=radio(s,coef,h,h,(1-beta.square())*VAR)
    sr=s[1:].sum(0);all_s=s.sum(0)
    err=float((beta[:,None]*all_s+mean-sr).abs().max())
    assert err<1e-12
    rho_a=RHO0/(1-beta.square());rho_r=2*RHO0/(1+beta)
    assert float(torch.maximum(rho_a,rho_r).max())<=.5+1e-12
    precision=torch.linalg.inv(VAR*torch.stack((torch.stack((torch.ones_like(beta),beta),-1),
                                              torch.stack((beta,torch.ones_like(beta)),-1)),-2))
    da=torch.tensor([2.,0.],device=DEV,dtype=DT);dr=torch.tensor([2.,2.],device=DEV,dtype=DT)
    assert torch.allclose(.5*torch.einsum('i,kij,j->k',da,precision,da),rho_a)
    assert torch.allclose(.5*torch.einsum('i,kij,j->k',dr,precision,dr),rho_r)
    # Independent raw-repetition noise verification at a fixed block.
    reps=7;desired_var=6.;scale=math.sqrt(reps*desired_var/NU2)
    raw=math.sqrt(NU2)*normal((20000,reps,4),912)
    empirical=float((scale*raw.mean(1)).square().mean())
    assert abs(empirical/desired_var-1)<.02
    # One client is removed rather than made statistically irrelevant in source.
    source_w=decode(all_s+normal((Q,),913),10)
    changed_w=decode(sr+normal((Q,),913),10)
    assert float((source_w-changed_w).square().sum())>0
    obj=dict(passed=True,mean_error=err,raw_repetition_noise_variance=empirical,
             code_sha256=sha(__file__),protocol_sha256=sha(ROOT/'PROTOCOL_KO.md'))
    write(OUT/'math_checks.json',obj);print(json.dumps(obj),flush=True)


@torch.no_grad()
def run(paths):
    chk=json.loads((OUT/'math_checks.json').read_text())
    assert chk['passed'] and chk['code_sha256']==sha(__file__) and chk['protocol_sha256']==sha(ROOT/'PROTOCOL_KO.md')
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();count=0;data_hash={}
    for ds_index,(name,path) in enumerate(paths.items()):
        x,y,xt,yt=dataset(path)
        data_hash[name]={p.name:sha(p) for p in sorted(path.glob('*ubyte'))}
        proj=normal((784,D-1),2026092890).float()/math.sqrt(784)
        z=torch.cat((F.relu((x-.5)@proj),torch.ones(len(x),1,device=DEV)),1).to(DT)
        zt=torch.cat((F.relu((xt-.5)@proj),torch.ones(len(xt),1,device=DEV)),1).to(DT)
        del x,xt
        for seed in SEEDS:
            for k in KS:
                begin=time.perf_counter();rows=[]
                ids=partition(y,seed,k);s,raw,clipinfo=stats(z,y,ids)
                sr=s[1:].sum(0);sc=s.sum(0)
                wt=decode(sr,k);wc=decode(sc,k);wu=decode(raw[1:].sum(0),k)
                refs=dict(clean_clipped_retained_accuracy=float(accuracy(wt,zt,yt)),
                          clean_clipped_source_accuracy=float(accuracy(wc,zt,yt)),
                          clean_unclipped_retained_accuracy=float(accuracy(wu,zt,yt)),
                          public_only_accuracy=float((yt==0).double().mean()),
                          clean_A_model_mse=float((wc-wt).square().mean()),
                          client_samples=int(ids.shape[1]),deleted_samples=int(ids.shape[1]),
                          deleted_fraction=1/k,statistics_dim=Q,**clipinfo)
                base=seed*10000+ds_index*1000+k
                z0=math.sqrt(VAR)*normal((CH,MC,Q),base+501)
                n1=normal((CH,MC,Q),base+502)
                for cond_index,condition in enumerate(CONDITIONS):
                    h0,h1,e0,e1=channels(condition,k,base+cond_index*7)
                    g0,g1=h0/e0,h1/e1
                    m0,r0=radio(s,torch.ones_like(h0),h0,e0,torch.full((CH,),VAR,device=DEV,dtype=DT))
                    source=m0[:,None,:]+z0;w0=decode(source,k)
                    sourcecf=decode(source-g0[:,0,None,None]*s[0][None,None,:],k)
                    src_cf_mse=(w0-sourcecf).square().mean((-1,-2))
                    prep=ledger(r0)
                    radio_ret=(g1[:,1:]@s[1:])
                    for method in METHODS:
                        if method=='No-op':
                            beta=torch.zeros(CH,device=DEV,dtype=DT)
                            mean=m0;noise=z0;cf_coef=g0[:,0]
                            cost=empty_ledger()
                        else:
                            beta=beta_for(method,e1)
                            coef=(1-beta[:,None])*torch.ones_like(h1);coef[:,0]=-beta
                            mu,r1=radio(s,coef,h1,e1,(1-beta.square())*VAR)
                            mean=beta[:,None]*m0+mu
                            noise=beta[:,None,None]*z0+torch.sqrt((1-beta.square())*VAR)[:,None,None]*n1
                            cf_coef=beta*(g0[:,0]-g1[:,0])
                            cost=ledger(r1)
                        metrics=evaluate(mean,noise,sr,radio_ret,source,w0,cf_coef,src_cf_mse,beta,
                                         g0,g1,s,wt,zt,yt,k,method)
                        if condition!='csi5' and method!='No-op':
                            assert float(metrics['statistic_KL_ideal'].max())<=1e-18
                            assert float(metrics['coupled_model_relative_error_max'].max())<=1e-9
                            assert float(metrics['rho_all'].max())<=.5+1e-10
                        col={**metrics,**{'cost_'+a:b for a,b in cost.items()},
                             **{'prep_'+a:b for a,b in prep.items()}}
                        col['lifecycle_real']=cost['total_real']+prep['total_real']
                        col['lifecycle_accounted_tx_energy']=cost['accounted_tx_energy_pilot_1.0']+prep['accounted_tx_energy_pilot_1.0']
                        converted={a:b.detach().cpu().tolist() for a,b in col.items()}
                        for j in range(CH):
                            row=dict(dataset=name,seed=seed,K=k,condition=condition,method=method,channel_draw=j,
                                     noise_draws=MC,utility_noise_draws=2)
                            row.update({a:b[j] for a,b in converted.items()});rows.append(row)
                    print(f'{name} seed={seed} K={k} {condition} complete',flush=True)
                compute=dict(local_private_samples=10000,local_statistics_recompute_at_deletion_samples=0,
                             encoder_MAC_per_sample=784*(D-1),dense_gram_moment_MAC_per_sample=D*D+D*C,
                             client_statistics_cache_bytes_fp64=Q*8,server_statistics_state_bytes_fp64=Q*8,
                             model_bytes_fp32=D*C*4,head_solves_per_live_phase=1,
                             dense_solve_order='O(D^3 + D^2*C)',
                             test_metrics_are_evaluator_only=True)
                obj=dict(references=refs,compute_ledger=compute,rows=rows,
                         code_sha256=sha(__file__),protocol_sha256=sha(ROOT/'PROTOCOL_KO.md'),
                         elapsed_seconds=time.perf_counter()-begin)
                write(OUT/f'{name}_seed{seed}_K{k}.json',obj);count+=len(rows)
                print(f'SAVED {name} seed={seed} K={k}, {len(rows)} rows, {obj["elapsed_seconds"]:.2f}s',flush=True)
        del z,zt,y,yt
    torch.cuda.synchronize()
    write(OUT/'completion.json',dict(completed=True,rows=count,expected_rows=2*3*3*5*4*CH,
          elapsed_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
          gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
          seeds=SEEDS,channels=CH,noise_draws=MC,utility_noise_draws=2,
          code_sha256=sha(__file__),protocol_sha256=sha(ROOT/'PROTOCOL_KO.md'),data_hashes=data_hash))
    print(f'COMPLETE {count} rows, {time.perf_counter()-start:.2f}s',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--math',action='store_true')
    ap.add_argument('--fashion',type=Path);ap.add_argument('--mnist',type=Path);args=ap.parse_args()
    if args.math:math_checks()
    else:
        assert args.fashion and args.mnist
        run({'FashionMNIST':args.fashion,'MNIST':args.mnist})
