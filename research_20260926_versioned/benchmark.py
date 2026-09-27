"""Versioned, frozen, sequential GPU pilot. Prior result directories are read-only."""
from pathlib import Path
import sys, json, math, time, hashlib, argparse
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'research_20260926_methods123'))
sys.path.insert(0,str(BASE/'research_20260926'))
sys.path.insert(0,'C:/Users/DISLAB/Desktop/split_learning/ota_ful')
import torch
import torch.nn.functional as F
import numpy as np
import common as cm
import run_pilots as pilot
import relation_protocol as rel
from vendor_adapters import kd_loss, fedquit_teacher

OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
CP=ROOT/'checkpoints';CP.mkdir(exist_ok=True)
DEV='cuda'; SEEDS=[371,372,373]


def save(path,obj):cm.write(path,obj)
def log(**kw):print(json.dumps(kw,ensure_ascii=False),flush=True)
def rng(seed):return cm.rng(seed)
def fp32(x):return x.float().to(x.dtype)


class Ledger:
    def __init__(self):
        self.v={k:0. for k in ['ul_real','dl_bits','digital_ul_bits','metadata_bits','pilot_real',
            'gradient_samples','forward_samples','student_samples','local_matrix_elements']}
    def packet(self,d,k,r=1):
        self.v['ul_real']+=d*r;self.v['metadata_bits']+=32*k;self.v['pilot_real']+=8*k
    def dl(self,n):self.v['dl_bits']+=n
    def report(self):
        rate=.5*math.log2(101)
        total=1.125*(self.v['ul_real']+self.v['pilot_real']+(self.v['dl_bits']+self.v['digital_ul_bits']+self.v['metadata_bits'])/rate)
        return dict(self.v,total_real_uses=total,rate_bits_per_real_use=rate,cp_factor=1.125)


def transmit(vectors, weights, clients, seed, repeats=1, snr=20, ledger=None, peak=True, unit=False, noise=None):
    """Client-private input simulator; receiver returns aggregate, plus allowed power metadata."""
    shape=vectors[0].shape;v=torch.stack([a.reshape(-1)*w for a,w in zip(vectors,weights)])
    h=torch.ones(len(v),device=DEV,dtype=v.dtype) if unit else torch.tensor([cm.CHANNELS[i] for i in clients],device=DEV,dtype=v.dtype)
    bounds=v.square().mean(1).sqrt()/h
    if peak:bounds=torch.maximum(bounds,v.abs().amax(1)/(2*h))
    a=bounds.max().clamp_min(1e-15)
    tx=v/(a*h[:,None]);rx=(h[:,None]*tx).sum(0)*a
    avg=float(tx.square().mean(1).max());pk=float(tx.square().max())
    assert avg<=1+2e-6
    if peak:assert pk<=4+2e-6
    if snr is not None:
        z=torch.randn(v.shape[1],device=DEV,dtype=v.dtype,generator=rng(seed)) if noise is None else noise.reshape(-1).to(v.dtype)
        rx=rx+z*a/math.sqrt(10**(snr/10)*repeats)
    if ledger:ledger.packet(v.shape[1],len(v),repeats)
    return rx.reshape(shape),dict(scale=float(a),max_average_power=avg,max_peak_power=pk,
                                  variance=0. if snr is None else float(a*a)/(10**(snr/10)*repeats))


def helmert(c=10,dtype=torch.float32):
    q=torch.zeros(c,c-1,device=DEV,dtype=dtype)
    for j in range(c-1):
        q[:j+1,j]=1/math.sqrt((j+1)*(j+2));q[j+1,j]=-(j+1)/math.sqrt((j+1)*(j+2))
    return q


def gated_forward(model,x,g):
    a=F.relu(model.conv1(x))*g[:8][None,:,None,None]
    a=model.pool(a)
    a=F.relu(model.conv2(a))*g[8:][None,:,None,None]
    return model.fc2(F.relu(model.fc1(a.flatten(1))))


def apply_gates(model,g):
    out=cm.clone_model(model)
    with torch.no_grad():
        for layer,gg in [(out.conv1,g[:8]),(out.conv2,g[8:])]:
            layer.weight*=gg[:,None,None,None];layer.bias*=gg
    return out


def math_checks():
    q=helmert(dtype=torch.float64)
    pr=torch.softmax(torch.randn(11,10,device=DEV,dtype=torch.float64,generator=rng(20)),1)
    err=float((pr-(.1+(pr@q)@q.T)).abs().max())
    assert err<1e-14
    model=cm.new_model(1);x=torch.randn(8,1,28,28,device=DEV,generator=rng(2));y=torch.arange(8,device=DEV)%10
    gate=torch.ones(24,device=DEV,requires_grad=True)
    forward_err=float((model(x)-gated_forward(model,x,gate)).abs().max());assert forward_err<1e-6
    gg=torch.linspace(.9,1.1,24,device=DEV)
    apply_err=float((apply_gates(model,gg)(x)-gated_forward(model,x,gg)).abs().max());assert apply_err<2e-6
    a=torch.autograd.grad(F.cross_entropy(gated_forward(model,x[:2],gate),y[:2]),gate)[0]
    r=torch.autograd.grad(F.cross_entropy(gated_forward(model,x[2:],gate),y[2:]),gate)[0]
    allg=torch.autograd.grad(F.cross_entropy(gated_forward(model,x,gate),y),gate)[0]
    contrast_err=float((allg-r-.25*(a-r)).abs().max());assert contrast_err<1e-6
    vv=[torch.randn(13,device=DEV,dtype=torch.float64,generator=rng(i)) for i in [1,2,3]]
    rx,st=transmit(vv,[.2,.3,.5],[0,1,2],0,snr=None)
    mac_err=float((rx-sum(w*v for w,v in zip([.2,.3,.5],vv))).abs().max());assert mac_err<1e-14
    h=torch.diag(torch.tensor([1.,2.,3.],device=DEV,dtype=torch.float64));b=torch.tensor([1.,2.,4.],device=DEV,dtype=torch.float64)
    p=torch.linalg.inv(h);u=torch.eye(3,device=DEV,dtype=torch.float64)[:,:2]
    ps=u@torch.linalg.solve(u.T@h@u,u.T)
    assert torch.linalg.matrix_rank(ps)==2 and (ps@torch.tensor([0.,0.,1.],device=DEV,dtype=h.dtype)).norm()==0
    inv_bad=abs((1/1+1/4)/2-1/2.5);assert inv_bad>.2
    out=dict(passed=True,probability_reconstruction_max_error=err,gate_forward_max_error=forward_err,
        gate_application_max_error=apply_err,gate_gradient_contrast_max_error=contrast_err,mac_max_error=mac_err,
        partial_operator_rank=2,partial_operator_dimension=3,full_operator_stationarity_disclosure='not resolved by preconditioning')
    save(OUT/'math_checks.json',out);log(stage='math',**out)


def training(data,seed,ortho=False,exclude=False):
    tag=f'seed{seed}_'+('ortho' if ortho else 'plain')+('_retain' if exclude else '_all')
    cp=CP/(tag+'.pt');mp=OUT/(tag+'_training.json')
    if cp.exists() and mp.exists():return cm.load_model(cp),json.loads(mp.read_text())
    m,cost,sec=cm.train_fl(data,seed,150,ortho,exclude)
    cm.save_model(cp,m)
    meta=dict(cost=cost,seconds=sec,validation_accuracy=cm.accuracy(m,data['x'][data['val']],data['y'][data['val']]),
        parameter_sha256=cm.vec_hash(cm.flat(m)),source_training='150 rounds; original frozen settings')
    save(mp,meta);return m,meta


def head_metrics(w,w0,wr,ztest,ytest,zforget):
    gap=float((w0-wr).square().sum())
    p=F.softmax(zforget@w,1);p0=F.softmax(zforget@w0,1);pr=F.softmax(zforget@wr,1)
    js=float(cm.js(p,pr).mean());js0=float(cm.js(p0,pr).mean())
    return dict(parameter_error_ratio=float((w-wr).square().sum())/gap,
        test_accuracy=float(((ztest@w).argmax(1)==ytest).double().mean()),
        forget_js=js,forget_normalized_js=js/max(js0,1e-20))


def head_received_h(hs,seed,ledger,peak=True):
    d=hs.shape[1];ix=torch.triu_indices(d,d,device=DEV)
    vec,_=transmit([h[ix[0],ix[1]] for h in hs],[1/9]*9,list(range(1,10)),seed,32,ledger=ledger,peak=peak,unit=True)
    hr=torch.zeros_like(hs[0]);hr[ix[0],ix[1]]=vec;hr[ix[1],ix[0]]=vec
    val,q=torch.linalg.eigh(hr);val=val.clamp_min(.01)
    return (q*val)@q.T


def head_correct(hhat,b,seed,method,ledger,peak=True,blocks=5,reps_budget=40):
    eig,q=torch.linalg.eigh(hhat);eig=fp32(eig);q=fp32(q)
    d=b.shape[1];per=d//blocks
    ledger.dl(d*d*32);ledger.dl(blocks*32)
    pre=method=='V1'
    if pre:ledger.dl(d*32)
    payload=torch.einsum('ij,kjl->kil',q.T,b)
    if pre:payload=payload/eig[None,:,None]
    bb=payload.reshape(9,blocks,per,10)
    scales=bb.square().mean((2,3)).sqrt()
    if peak:scales=torch.maximum(scales,bb.abs().amax((2,3))/2)
    variance=(scales.max(0).values/9).square()/100
    raw=per*10*variance
    weighted=raw if pre else 10*variance*eig.reshape(blocks,per).pow(-2).sum(1)
    alloc_coeff=raw if method=='gradient_MSE' else weighted
    reps=pilot.greedy(alloc_coeff,reps_budget).int()
    got=[];power=[]
    for j in range(blocks):
        vals,st=transmit(list(bb[:,j]),[1/9]*9,list(range(1,10)),seed+j,int(reps[j]),ledger=ledger,peak=peak,unit=True)
        got.append(vals);power.append(st)
    rec=torch.cat(got)
    correction=q@(rec if pre else rec/eig[:,None])
    return correction,dict(repetitions=reps.tolist(),power=power)


@torch.no_grad()
def stage1():
    x,y,xt,yt=pilot.load_data()
    proj=torch.randn((784,64),device=DEV,generator=rng(20260922))/math.sqrt(784)
    z=torch.cat([F.relu((x-.5)@proj),torch.ones(len(x),1,device=DEV)],1).double()
    zt=torch.cat([F.relu((xt-.5)@proj),torch.ones(len(xt),1,device=DEV)],1).double()
    yy=F.one_hot(y,10).double();eye=torch.eye(65,device=DEV,dtype=z.dtype)
    for seed in [202609263,202609264,202609265]:
        path=OUT/f'DS_seed{seed}.json'
        if path.exists() and json.loads(path.read_text()).get('complete'):continue
        t0=time.perf_counter();pools=pilot.partition(y,seed)
        hs=torch.stack([z[p].T@z[p]/len(p)+.01*eye for p in pools]);cs=torch.stack([z[p].T@yy[p]/len(p) for p in pools])
        h,c=hs[1:].mean(0),cs[1:].mean(0);w0=fp32(torch.linalg.solve(hs.mean(0),cs.mean(0)))
        wr=torch.linalg.solve(h,c);b=hs[1:]@w0-cs[1:];zf=z[pools[0]]
        assert float((w0-torch.linalg.solve(h,b.mean(0))-wr).norm())<1e-10
        rows=[dict(method='No-op',version='baseline',noise=None,metrics=head_metrics(w0,w0,wr,zt,yt,zf),cost=Ledger().report()),
              dict(method='Exact retrain',version='reference',noise=None,metrics=head_metrics(wr,w0,wr,zt,yt,zf),cost=None)]
        u=torch.linalg.qr(torch.randn(65,48,device=DEV,dtype=z.dtype,generator=rng(90317))).Q
        u=fp32(u)
        subhs=u.T@hs[1:]@u;subb=u.T@b
        # A distinct protocol transcript; never combine its observations with full-H protocol.
        for noise in range(16):
            ns=seed*100+noise*20
            for peak in [True,False]:
                hcost=Ledger();hhat=head_received_h(hs[1:],ns,hcost,peak)
                for method,budget in [('Original',40),('V1',40),('Original-cost-matched',44),('gradient_MSE',40)]:
                    if not peak and method not in ['Original','V1']:continue
                    ledger=Ledger();ledger.v=hcost.v.copy();ledger.dl(2*650*32)
                    delta,diag=head_correct(hhat,b,ns+1,'Original' if method=='Original-cost-matched' else method,
                                            ledger,peak,reps_budget=budget)
                    w=fp32(w0-delta)
                    rows.append(dict(method='DS-Air' if method in ['Original','V1'] else method,
                        version=method if method in ['Original','V1'] else 'baseline',noise=noise,peak=peak,
                        metrics=head_metrics(w,w0,wr,zt,yt,zf),cost=ledger.report(),diagnostic=diag,
                        disclosure='full-rank stationarity reconstruction possible; performance comparator'))
                if peak:
                    ledger=Ledger();ledger.v=hcost.v.copy();ledger.dl(650*32)
                    cc,_=transmit(list(cs[1:]),[1/9]*9,list(range(1,10)),ns+7,8,ledger=ledger,unit=True)
                    w=fp32(torch.linalg.solve(hhat,cc))
                    rows.append(dict(method='Noisy stats retrain',version='baseline',noise=noise,peak=True,
                        metrics=head_metrics(w,w0,wr,zt,yt,zf),cost=ledger.report()))
                    ledger=Ledger();ledger.dl(2*650*32)
                    dh,_=transmit([hh.diag() for hh in hs[1:]],[1/9]*9,list(range(1,10)),ns+8,32,ledger=ledger,unit=True)
                    bg,_=transmit(list(b),[1/9]*9,list(range(1,10)),ns+9,8,ledger=ledger,unit=True)
                    w=fp32(w0-bg/dh.clamp_min(.01)[:,None])
                    rows.append(dict(method='Diagonal Newton',version='baseline',noise=noise,peak=True,
                        metrics=head_metrics(w,w0,wr,zt,yt,zf),cost=ledger.report()))
                    ledger=Ledger();ledger.dl(2*650*32+32)
                    hsmall=head_received_h(subhs,ns+10,ledger)
                    dd,diag=head_correct(hsmall,subb,ns+11,'V1',ledger,blocks=4)
                    w=fp32(w0-u@dd)
                    rows.append(dict(method='DS-Air-subspace48',version='V1',noise=noise,peak=True,
                        metrics=head_metrics(w,w0,wr,zt,yt,zf),cost=ledger.report(),diagnostic=diag,
                        disclosure='rank-limited separate transcript, not DP; subspace bias evaluated against full retrain'))
            save(path,dict(seed=seed,rows=rows,complete=False))
        ledger=Ledger();w=w0.clone()
        for step in range(40):
            ledger.dl(650*32)
            gs=hs[1:]@w-cs[1:]
            agg,_=transmit(list(gs),[1/9]*9,list(range(1,10)),seed+step,1,snr=None,ledger=ledger,unit=True)
            w=fp32(w-.1*agg)
        ledger.dl(650*32)
        rows.append(dict(method='Retained GD40 noiseless',version='baseline',noise=None,
            metrics=head_metrics(w,w0,wr,zt,yt,zf),cost=ledger.report()))
        prep=Ledger();prep.packet(2145,10,1);prep.packet(650,10,1);prep.dl(650*32+32)
        save(path,dict(seed=seed,rows=rows,complete=True,seconds=time.perf_counter()-t0,
            preparation_cost=prep.report(),source_model='public random encoder, exact ridge head',reference='full retained ridge optimum'))
        log(stage=1,seed=seed,status='complete',seconds=time.perf_counter()-t0)


@torch.no_grad()
def activation_vectors(model,data,matched=False):
    sums=[];counts=[]
    for ix in data['pools']:
        aa=[]
        for xx in data['x'][ix].split(128):
            a=F.relu(model.conv1(xx));b=F.relu(model.conv2(model.pool(a)))
            aa.append(torch.cat([a.amax((2,3)),b.amax((2,3))],1))
        aa=torch.cat(aa).double();yy=data['y'][ix]
        sums.append(torch.stack([aa[yy==c].sum(0) for c in range(10)]))
        counts.append(torch.stack([(yy==c).sum() for c in range(10)]).double())
    sums=torch.stack(sums);counts=torch.stack(counts)
    if not matched:
        return [sums[i].sum(0)/counts[i].sum()*(1 if i==0 else -1/9) for i in range(10)]
    pf=counts[0]/counts[0].sum();nr=counts[1:].sum(0)
    return [(pf[:,None]*sums[0]/counts[0][:,None]).sum(0)]+[-(pf[:,None]*sums[i]/nr[:,None]).sum(0) for i in range(1,10)]


def gate_gradients(model,data):
    gs=[]
    for pool in data['pools']:
        result=torch.zeros(24,device=DEV)
        for ix in pool.split(128):
            gate=torch.ones(24,device=DEV,requires_grad=True)
            loss=F.cross_entropy(gated_forward(model,data['x'][ix],gate),data['y'][ix])
            result+=torch.autograd.grad(loss,gate)[0].detach()*len(ix)/len(pool)
        gs.append(result)
    return gs


def prune_gate(score,mild=False):
    gate=torch.ones(24,device=DEV)
    for start,end,k in [(0,8,2),(8,24,4)]:
        idx=score[start:end].topk(k).indices+start
        strength=torch.full((k,),.1,device=DEV) if mild else torch.maximum(torch.full((k,),.5,device=DEV),1-torch.arange(1,k+1,device=DEV)/k)
        gate[idx]=1-strength
    return gate


def model_grad(model,bx,by,target=False):
    loss=rel.uce(model(bx),by) if target else F.cross_entropy(model(bx),by)
    return cm.flatten(torch.autograd.grad(loss,tuple(model.parameters()))).detach()


def cnn_baseline(source,data,seed,mode):
    m=cm.clone_model(source);ledger=Ledger();d=cm.flat(m).numel();streams={i:rng(seed*8100+i) for i in range(10)}
    for step in range(10):
        ledger.dl(d*32)
        if mode=='retained_sgd':
            vs=[model_grad(m,*cm.batch(data,i,streams[i],64)) for i in range(1,10)]
            grad,_=transmit(vs,[1/9]*9,list(range(1,10)),seed+step,snr=None,ledger=ledger)
            ledger.v['gradient_samples']+=9*64
        else:
            vs=[model_grad(m,*cm.batch(data,i,streams[i],64)) for i in range(1,10)]
            vs.append(model_grad(m,*cm.batch(data,0,streams[0],64),target=True))
            rows=torch.stack(vs);unit,norms=rel.normalized_rows(rows)
            if mode=='fedosd':
                weights=rel.relation_coefficients(unit)
                residual=weights@unit
                ledger.v['digital_ul_bits']+=10*d*32;ledger.v['pilot_real']+=80
            else:
                sketches=rel.quantize_rows(rel.srht(unit,2048,seed*100+step),16)
                weights=rel.relation_coefficients(sketches)
                residual,_=transmit(list(unit),weights.tolist(),list(range(1,10))+[0],seed*200+step,ledger=ledger)
                ledger.v['digital_ul_bits']+=10*(2048*16+32)
                ledger.v['metadata_bits']+=10*32;ledger.dl(10*32+32)
            grad=rel.rescale_fedosd(residual,norms[-1]);ledger.v['gradient_samples']+=640
        cm.assign(m,cm.flat(m)-.01*grad)
    ledger.dl(d*32)
    return m,ledger.report()


def stage2():
    for seed in SEEDS:
        path=OUT/f'OG_seed{seed}.json'
        if path.exists() and json.loads(path.read_text()).get('complete'):continue
        begin=time.perf_counter();data=cm.load_data(seed)
        source,prep=training(data,seed,ortho=True);ref,refprep=training(data,seed,ortho=True,exclude=True)
        _,sp=cm.evaluate(source,ref,data)
        rows=[]
        def record(name,version,model,cost=None,noise=None,diagnostic=None):
            met,_=cm.evaluate(model,ref,data,sp)
            rows.append(dict(method=name,version=version,noise=noise,metrics=met,cost=cost,diagnostic=diagnostic))
        record('No-op','baseline',source,Ledger().report());record('Target-free ortho retrain','reference',ref,refprep['cost'])
        av=activation_vectors(source,data);mv=activation_vectors(source,data,True);gg=gate_gradients(source,data)
        v1=[.1*gg[0]]+[-.1/9*g for g in gg[1:]]
        for noise in [-1,0,1,2,3]:
            snr=None if noise==-1 else 20
            for version,vectors in [('Original',av),('V1',v1)]:
                ledger=Ledger();ledger.dl(2*63562*32)
                score,power=transmit(vectors,[1]*10,list(range(10)),seed*2000+max(noise,0),8,snr=snr,ledger=ledger)
                gate=prune_gate(score) if version=='Original' else 1+(.1*score).clamp(-.05,.05)
                if version=='Original':ledger.v['forward_samples']+=12000
                else:ledger.v['gradient_samples']+=12000
                record('OG-Air'+('-noiseless' if noise==-1 else ''),version,apply_gates(source,gate),ledger.report(),noise,
                       dict(gate_change_norm=float((gate-1).norm()),changed_channels=int((gate!=1).sum()),power=power))
        for name,score in [('Original-mild',sum(av)),('Class-matched-mild',sum(mv))]:
            ledger=Ledger();ledger.dl(2*63562*32);ledger.packet(24,10,1);ledger.v['forward_samples']+=12000
            if name.startswith('Class'):ledger.v['metadata_bits']+=10*10*32;ledger.dl(20*32)
            record(name,'ablation',apply_gates(source,prune_gate(score,mild=True)),ledger.report())
        ledger=Ledger();ledger.dl(2*63562*32);ledger.packet(24,9,1);ledger.v['gradient_samples']+=10800
        gr=sum(gg[1:])/9
        record('Retained gate step','baseline',apply_gates(source,1+(-.1*gr).clamp(-.05,.05)),ledger.report())
        for name,mode in [('Retained SGD10','retained_sgd'),('FedOSD-core10','fedosd'),('Legacy-Sketch-V1','sketch')]:
            model,cost=cnn_baseline(source,data,seed,mode)
            record(name,'baseline',model,cost)
        save(path,dict(seed=seed,rows=rows,complete=True,seconds=time.perf_counter()-begin,
            preparation=prep,reference_preparation=refprep,reference='paired target-free orthogonal FedAvg',
            original_gate_gradient_norms=[float(g.norm()) for g in gg]))
        log(stage=2,seed=seed,status='complete',seconds=time.perf_counter()-begin)


def simplex(v):
    u=v.sort(dim=1,descending=True).values;cssv=u.cumsum(1)-1
    j=torch.arange(1,v.shape[1]+1,device=DEV,dtype=v.dtype)
    rho=(u-cssv/j>0).sum(1)-1;theta=cssv.gather(1,rho[:,None])/(rho[:,None]+1)
    return (v-theta).clamp_min(0)


def teacher(data,seed,i):
    cp=CP/f'teacher_seed{seed}_client{i}.pt';mp=OUT/f'teacher_seed{seed}_client{i}.json'
    if cp.exists() and mp.exists():return cm.load_model(cp),json.loads(mp.read_text())
    m=cm.new_model(seed);gen=rng(seed*9000+i);begin=time.perf_counter()
    for _ in range(500):
        bx,by=cm.batch(data,i,gen,64);g=model_grad(m,bx,by)
        cm.assign(m,cm.flat(m)-.03*g)
    cm.save_model(cp,m);meta=dict(samples=32000,seconds=time.perf_counter()-begin,sha256=cm.vec_hash(cm.flat(m)))
    save(mp,meta);return m,meta


def distill(data,seed,target,indices=None):
    model=cm.new_model(seed);gen=rng(seed+444)
    x=data['x'][data['public'] if indices is None else data['public'][indices]]
    begin=time.perf_counter()
    for _ in range(600):
        ix=torch.randint(len(x),(64,),device=DEV,generator=gen)
        loss=kd_loss(model(x[ix]),target[ix],2.)
        grads=torch.autograd.grad(loss,tuple(model.parameters()))
        with torch.no_grad():
            for p,g in zip(model.parameters(),grads):p.add_(g,alpha=-.05)
    cm.sync();return model,time.perf_counter()-begin


def teacher_targets(qs,seed,noise,kind,repeats=1,indices=None):
    active=list(range(1,10));matrix=[qs[i] if indices is None else qs[i][indices] for i in active]
    n=matrix[0].shape[0];ledger=Ledger();ledger.dl(n*32+32);ledger.v['forward_samples']+=9*n
    q=helmert();snr=None if noise==-1 else 20
    shared=torch.randn((n,10),device=DEV,generator=rng(seed*3000+max(noise,0)))
    if kind=='digital8':
        qq=torch.stack([(v*255).round()/255 for v in matrix]).mean(0)
        qq=qq/qq.sum(1,keepdim=True).clamp_min(1e-10)
        ledger.v['digital_ul_bits']+=9*n*10*8;ledger.v['pilot_real']+=9*8;ledger.v['metadata_bits']+=9*32
        return qq,ledger,{}
    if kind=='V1':vs=[v@q for v in matrix];nz=shared@q
    elif kind=='centered':vs=[v-.1 for v in matrix];nz=shared
    else:vs=matrix;nz=shared
    agg,power=transmit(vs,[1/9]*9,active,seed*3000+max(noise,0),repeats,snr=snr,ledger=ledger,noise=nz)
    if kind=='V1':agg=.1+agg@q.T
    elif kind=='centered':agg=agg+.1
    ledger.v['metadata_bits']+=9*32
    return simplex(agg),ledger,power


def fedquit_adapt(source,data,seed):
    m=cm.clone_model(source);tea=cm.clone_model(source);streams={i:rng(seed*8000+i) for i in range(10)};cost=Ledger()
    for step in range(100):
        cost.dl(63562*32);vs=[]
        for i in range(10):
            bx,by=cm.batch(data,i,streams[i],32)
            if i==0:
                with torch.no_grad():target=fedquit_teacher(tea(bx),by)
                loss=kd_loss(m(bx),target)
            else:loss=F.cross_entropy(m(bx),by)
            vs.append(cm.flatten(torch.autograd.grad(loss,tuple(m.parameters()))).detach())
        grad,_=transmit(vs,[.5]+[.5/9]*9,list(range(10)),seed+step,snr=None,ledger=cost)
        cm.assign(m,cm.flat(m)-.01*grad);cost.v['gradient_samples']+=320
    cost.dl(63562*32);return m,cost.report()


def stage3():
    for seed in SEEDS:
        path=OUT/f'RTD_seed{seed}.json'
        if path.exists() and json.loads(path.read_text()).get('complete'):continue
        begin=time.perf_counter();data=cm.load_data(seed);qs=[];tmeta=[]
        for i in range(10):
            t,meta=teacher(data,seed,i);tmeta.append(meta)
            qs.append(cm.probabilities(t,data['x'][data['public']],2.))
        source,ssec=distill(data,seed,torch.stack(qs).mean(0))
        qr=torch.stack(qs[1:]).mean(0);ref,rsec=distill(data,seed,qr)
        cm.save_model(CP/f'rtd_reference_seed{seed}.pt',ref)
        _,sp=cm.evaluate(source,ref,data);rows=[]
        def record(name,version,m,cost=None,noise=None,extra=None):
            met,_=cm.evaluate(m,ref,data,sp)
            rows.append(dict(method=name,version=version,noise=noise,metrics=met,cost=cost,extra=extra))
        record('No-op','baseline',source,Ledger().report())
        rc=Ledger();rc.packet(20000,9,1);rc.dl(2000*32+32+63562*32);rc.v['student_samples']+=38400
        record('Target-free KD','reference',ref,rc.report(),extra={'student_seconds':rsec})
        # One actual noiseless re-encoding run; same full target-free objective.
        target,cost,power=teacher_targets(qs,seed,-1,'V1');m,sec=distill(data,seed,target)
        cost.dl(63562*32);cost.v['student_samples']+=38400
        record('RTD-Air-noiseless','V1',m,cost.report(),-1,dict(target_max_error=float((target-qr).abs().max()),student_seconds=sec))
        for noise in range(4):
            cases=[('RTD-Air','Original',1,None),('RTD-Air','Original',4,None),('RTD-Air','V1',1,None),('RTD-Air','V1',4,None),
                   ('Centered-full10','centered',1,None)]
            subset=torch.randperm(2000,device=DEV,generator=rng(19523))[:1000]
            cases.append(('RTD-Air-query1000','V1',1,subset))
            if noise==0:cases.append(('Digital8','digital8',1,None))
            for name,kind,reps,indices in cases:
                target,cost,power=teacher_targets(qs,seed,noise,kind,reps,indices)
                m,sec=distill(data,seed,target,indices);cost.dl(63562*32);cost.v['student_samples']+=38400
                true=qr if indices is None else qr[indices]
                record(name,kind if kind in ['Original','V1'] else 'baseline',m,cost.report(),noise,
                       dict(repeats=reps,student_seconds=sec,target_mse=float((target-true).square().mean()),power=power))
            save(path,dict(seed=seed,rows=rows,complete=False))
            log(stage=3,seed=seed,noise=noise,status='comparison_batch_complete')
        # External methods have a distinct FedAvg source/reference; do not rank with KD family.
        plain,pp=training(data,seed);plainref,rp=training(data,seed,exclude=True)
        _,psp=cm.evaluate(plain,plainref,data);external=[]
        for name,m,cost in [('No-op',plain,Ledger().report()),('Target-free FedAvg',plainref,rp['cost'])]:
            met,_=cm.evaluate(m,plainref,data,psp);external.append(dict(method=name,metrics=met,cost=cost))
        m,cost=fedquit_adapt(plain,data,seed);met,_=cm.evaluate(m,plainref,data,psp)
        external.append(dict(method='FedQUIT-logit-min + CE adaptation',metrics=met,cost=cost))
        oq=cm.probabilities(plain,data['x'][data['public']],2.);m,sec=distill(data,seed,oq)
        met,_=cm.evaluate(m,plainref,data,psp)
        oldcost=Ledger();oldcost.dl(63562*32);oldcost.v['student_samples']+=38400
        external.append(dict(method='Old-global-teacher KD',metrics=met,cost=oldcost.report(),student_seconds=sec))
        cross,_=cm.evaluate(ref,plainref,data)
        prep=Ledger();prep.packet(20000,10,1);prep.dl(2000*32+32+63562*32);prep.v['student_samples']+=38400
        save(path,dict(seed=seed,rows=rows,complete=True,external_fedavg=external,plain_preparation=pp,
            teachers=tmeta,preparation=prep.report(),public_data_bytes_uncached=2000*784,
            source_student_seconds=ssec,reference='target-free independent-teacher KD at 2000 queries',
            cross_reference_metrics=cross,seconds=time.perf_counter()-begin))
        log(stage=3,seed=seed,status='complete',seconds=time.perf_counter()-begin)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['math','1','2','3','all'],default='all');args=p.parse_args()
    if args.stage in ['math','all']:math_checks()
    assert json.loads((OUT/'math_checks.json').read_text())['passed']
    save(OUT/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL_KO.md').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seeds=SEEDS,one_worker=True))
    for k,fn in [('1',stage1),('2',stage2),('3',stage3)]:
        if args.stage in [k,'all']:fn()
    log(status='requested_stages_complete')


if __name__=='__main__':main()
