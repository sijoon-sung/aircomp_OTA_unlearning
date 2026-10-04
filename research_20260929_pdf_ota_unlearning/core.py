"""PDF section 6.2, explicit client/aggregate boundary; no past client updates."""
from pathlib import Path
import copy, gzip, hashlib, json, math, os, subprocess, threading, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
DEVICE='cuda'
torch.set_num_threads(2)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True)

def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def log(**kw):print(json.dumps(kw,ensure_ascii=False),flush=True)
def gen(seed):return torch.Generator(device=DEVICE).manual_seed(int(seed))
def flat(m):return torch.cat([p.detach().flatten() for p in m.parameters()])
def assign(m,v):
    off=0
    with torch.no_grad():
        for p in m.parameters():p.copy_(v[off:off+p.numel()].view_as(p));off+=p.numel()
    return m
def clone(m):return copy.deepcopy(m)
def normalize(v):return v/v.norm(dim=-1,keepdim=True).clamp_min(1e-20)
def cosine(a,b):return float((normalize(a)*normalize(b)).sum())
def save_model(path,m):
    if Path(path).exists():raise FileExistsError(path)
    torch.save({k:v.cpu() for k,v in m.state_dict().items()},path)

class DeterministicPool(nn.Module):
    """Same 4x4 adaptive average as PyTorch, implemented by fixed linear maps."""
    def __init__(self):
        super().__init__()
        for n in [7,8]:
            a=torch.zeros(4,n)
            for i in range(4):
                lo=math.floor(i*n/4);hi=math.ceil((i+1)*n/4);a[i,lo:hi]=1/(hi-lo)
            self.register_buffer('pool'+str(n),a,persistent=False)
    def forward(self,x):
        a=getattr(self,'pool'+str(x.shape[-1]))
        return torch.matmul(torch.matmul(a,x),a.T)

class Net(nn.Module):
    def __init__(self,channels=1):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(channels,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),DeterministicPool(),
            nn.Flatten(),nn.Linear(32*4*4,64),nn.ReLU(),nn.Linear(64,10))
    def forward(self,x):return self.net(x)
def model(seed,channels=1):
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed);return Net(channels).to(DEVICE)

def idx(dataset,kind):
    root=BASE/'data_cache'/dataset/'raw'
    name={'x':'train-images-idx3-ubyte','y':'train-labels-idx1-ubyte','xt':'t10k-images-idx3-ubyte','yt':'t10k-labels-idx1-ubyte'}[kind]
    p=root/name;b=p.read_bytes() if p.exists() else gzip.open(str(p)+'.gz','rb').read()
    if kind in ('x','xt'):return np.frombuffer(b,dtype=np.uint8,offset=16).copy().reshape(-1,1,28,28)
    return np.frombuffer(b,dtype=np.uint8,offset=8).copy().astype(np.int64)

def load_data(dataset,alpha,seed,backdoor=False):
    if dataset=='CIFAR10':
        z=np.load(BASE/'data_cache/CIFAR10/data.npz',allow_pickle=False)
        x,y,xt,yt=[z[k] for k in ['x','y','xt','yt']]
    else:x,y,xt,yt=[idx(dataset,k) for k in ['x','y','xt','yt']]
    rng=np.random.default_rng(seed)
    tr=[];dev=[]
    for c in range(10):
        ids=rng.permutation(np.flatnonzero(y==c));tr.extend(ids[:1200]);dev.extend(ids[1200:1400])
    tr=np.array(tr);dev=np.array(dev)
    for attempt in range(10000):
        pools=[[] for _ in range(20)]
        for c in range(10):
            ids=rng.permutation(tr[y[tr]==c]);f=rng.dirichlet(np.full(20,alpha))
            for i,part in enumerate(np.split(ids,(np.cumsum(f)[:-1]*len(ids)).astype(int))):pools[i].extend(part)
        if min(map(len,pools))>=60:break
    else:raise RuntimeError('Partition infeasible')
    pools=[np.array(p,dtype=np.int64) for p in pools]
    assert len(np.unique(np.concatenate(pools)))==12000
    assert not np.intersect1d(tr,dev).size
    # Materialize only this experiment's subset, preserving original indices on disk.
    original=np.concatenate([*pools,dev]);xx=x[original].copy();yy=y[original].copy()
    offsets=np.cumsum([0]+[len(p) for p in pools]);localp=[np.arange(offsets[i],offsets[i+1]) for i in range(20)]
    devlocal=np.arange(12000,14000)
    clean_y=yy.copy();poisoned=np.array([],dtype=int)
    if backdoor:
        poisoned=rng.permutation(localp[0])[:int(.8*len(localp[0]))]
        xx[poisoned,:,-4:-1,-4:-1]=255;yy[poisoned]=0
    def gx(a):return torch.tensor(a,device=DEVICE,dtype=torch.float32).div_(255).sub_(.5).div_(.5)
    xg=gx(xx);yg=torch.tensor(yy,device=DEVICE,dtype=torch.long)
    tx=gx(xt);ty=torch.tensor(yt,device=DEVICE,dtype=torch.long)
    # Class-matched nonmember audit. In backdoor case poison status is also matched.
    mids=[];nids=[]
    for c in range(10):
        mm=localp[0][clean_y[localp[0]]==c];nn=devlocal[clean_y[devlocal]==c]
        size=min(len(mm),len(nn));mids.extend(rng.permutation(mm)[:size]);nids.extend(rng.permutation(nn)[:size])
    mids=np.array(mids,dtype=int);nids=np.array(nids,dtype=int)
    nx=xg[nids].clone();ny=yg[nids].clone()
    if backdoor:
        mask=np.isin(mids,poisoned);nx[mask,:,-4:-1,-4:-1]=1;ny[mask]=0
    reta=rng.permutation(np.concatenate(localp[1:]))[:2000]
    return dict(x=xg,y=yg,tx=tx,ty=ty,pools=[torch.tensor(p,device=DEVICE) for p in localp],
        counts=[len(p) for p in localp],dev=torch.tensor(devlocal,device=DEVICE),
        retain=torch.tensor(reta,device=DEVICE),mx=xg[mids],my=yg[mids],nx=nx,ny=ny,
        original_indices=original,original_pools=pools,backdoor=backdoor,seed=seed,dataset=dataset,
        alpha=alpha,poisoned=poisoned,channels=x.shape[1])

def batch(data,i,seed,n):
    p=data['pools'][i];r=torch.randint(len(p),(n,),generator=gen(seed),device=DEVICE)
    return data['x'][p[r]],data['y'][p[r]]

def gradient(m,x,y,uce=False):
    m.train();out=m(x)
    loss=-torch.log1p(-.5*out.softmax(1).gather(1,y[:,None])).mean() if uce else F.cross_entropy(out,y)
    return torch.cat([g.detach().flatten() for g in torch.autograd.grad(loss,tuple(m.parameters()))])

@torch.no_grad()
def predict(m,x):
    m.eval();return torch.cat([m(b).softmax(1) for b in x.split(256)])

def fwht(v):
    n=v.shape[-1];h=1;x=v
    while h<n:
        z=x.reshape(*x.shape[:-1],-1,2,h);a=z[...,0,:];b=z[...,1,:]
        x=torch.stack((a+b,a-b),dim=-2).reshape_as(x);h*=2
    return x/math.sqrt(n)

class Sketch:
    def __init__(self,d,s,seed,dtype=torch.float32):
        self.d=d;self.s=s;self.n=1<<(d-1).bit_length();self.identity=s==d
        if not self.identity:
            g=gen(seed);self.sign=(2*torch.randint(2,(self.n,),device=DEVICE,generator=g)-1).to(dtype)
            self.ids=torch.randperm(self.n,device=DEVICE,generator=g)[:s];self.scale=math.sqrt(self.n/s)
    def forward(self,v):
        if self.identity:return v.clone()
        return fwht(F.pad(v,(0,self.n-self.d))*self.sign)[...,self.ids]*self.scale
    def adjoint(self,z):
        if self.identity:return z.clone()
        x=torch.zeros((*z.shape[:-1],self.n),device=z.device,dtype=z.dtype)
        x[...,self.ids]=z
        return (fwht(x)*self.sign)[...,:self.d]*self.scale

def exact_null(rows,v):
    # Float64 Gram eigensolve, rank-thresholded; only called by explicit oracle/audit.
    a=rows.double();b=v.double();gram=a@a.T
    lam,u=torch.linalg.eigh(gram);mask=lam>max(float(lam.max())*1e-10,1e-14)
    coeff=u[:,mask]@((u[:,mask].T@(a@b))/lam[mask])
    return (b-a.T@coeff).to(v.dtype)

class Cost:
    def __init__(self):
        self.v={k:0 for k in ['ul_analog_reals','ul_digital_bits','dl_bits','pilot_reals','ul_slots','dl_slots','gradient_samples','client_steps']}
    def dl(self,d):self.v['dl_bits']+=32*d;self.v['dl_slots']+=1
    def digital(self,d):self.v['ul_digital_bits']+=32*d;self.v['ul_slots']+=1
    def ul(self,d,n):self.v['ul_analog_reals']+=d;self.v['pilot_reals']+=8*n;self.v['ul_slots']+=1
    def work(self,n,steps=1):self.v['gradient_samples']+=n;self.v['client_steps']+=steps
    def report(self):
        r=dict(self.v);rate=.5*math.log2(101)
        r['equivalent_real_uses_at_20db']=r['ul_analog_reals']+r['pilot_reals']+(r['ul_digital_bits']+r['dl_bits'])/rate
        return r

class AirComp:
    """Channel simulator sees client waveforms. Server receives only sum, variance, counts."""
    def __init__(self,snr,seed,cost,threshold=.1):
        self.snr=snr;self.seed=seed;self.cost=cost;self.threshold=threshold;self.call=0
    def transmit(self,vectors,bound,weights=None):
        d=vectors.shape[1];n=len(vectors)
        if weights is None:weights=torch.ones(n,device=DEVICE,dtype=vectors.dtype)
        b=torch.as_tensor(bound,device=DEVICE,dtype=vectors.dtype)
        norms=vectors.norm(dim=1,keepdim=True)
        clipped=vectors*torch.minimum(torch.ones_like(norms),b/norms.clamp_min(1e-20))
        weighted=clipped*weights[:,None]
        self.cost.ul(d,n)
        if self.snr is None:return weighted.sum(0),0.,n
        rng=gen(self.seed+104729*self.call);self.call+=1
        power=-torch.log(torch.rand(n,device=DEVICE,generator=rng).clamp_min(1e-12))
        active=power>=self.threshold
        beta=float(b)*float(weights.max())/math.sqrt(d*self.threshold)
        sigma=beta/10**(self.snr/20)
        # Equivalent to sum sqrt(power_i)*[weighted_i/(sqrt(power_i)*beta)] + noise.
        out=weighted[active].sum(0)+sigma*torch.randn(d,device=DEVICE,dtype=vectors.dtype,generator=rng)
        return out,sigma,int(active.sum())

class ProjectionClient:
    def __init__(self,z):self.z=normalize(z)
    def message(self,d,clip):return (self.z@d).clamp(-clip,clip)*self.z

def iterative_projection(gu,retained,sketch,K,channel,clip=.1,stopping=True):
    # retained vectors are used here only to construct isolated simulated clients.
    clients=[ProjectionClient(sketch.forward(g)) for g in retained]
    zu=sketch.forward(gu);d=-zu
    channel.cost.digital(sketch.s+1);channel.cost.dl(sketch.s+1)
    omega=1/len(clients);trace=[]
    sigma=0. if channel.snr is None else clip/math.sqrt(sketch.s*channel.threshold)/10**(channel.snr/20)
    floor=omega**2*sketch.s*sigma**2*K
    if stopping and float(gu.square().sum())<=floor:
        return torch.zeros_like(gu),dict(inner=[],floor_stop=True,floor=floor,K_actual=0)
    for k in range(K):
        payload=torch.stack([client.message(d,clip) for client in clients])
        total,sigma,active=channel.transmit(payload,clip)
        d=d-omega*total;channel.cost.dl(sketch.s)
        trace.append(dict(k=k+1,received_norm=float(total.norm()),sigma=sigma,active=active))
        if stopping and sigma>0 and float(total.square().sum())<1.2*sketch.s*sigma**2:break
    lifted=sketch.adjoint(d)
    return normalize(lifted)*gu.norm(),dict(inner=trace,floor_stop=False,floor=floor,K_actual=len(trace))

def train(data,seed,exclude=False,offset=0,rounds=80,start=None,clients=None,stage=0,cost=None):
    m=model(seed,data['channels']) if start is None else clone(start)
    local=clone(m);ids=list(range(1 if exclude else 0,20)) if clients is None else list(clients)
    co=Cost() if cost is None else cost;ch=AirComp(20,seed*11000+offset+stage*500000,co)
    begin=time.time()
    if not ids:return m,co.report(),0.
    weights=torch.tensor([data['counts'][i] for i in ids],device=DEVICE,dtype=torch.float32);weights/=weights.sum()
    for r in range(rounds):
        anchor=flat(m);co.dl(len(anchor));updates=[]
        for i in ids:
            assign(local,anchor)
            for step in range(2):
                x,y=batch(data,i,seed*100000+offset+stage*10000000+r*1000+i*10+step,64)
                assign(local,flat(local)-.05*gradient(local,x,y));co.work(64)
            updates.append(flat(local)-anchor)
        avg,_,_=ch.transmit(torch.stack(updates),1.,weights)
        assign(m,anchor+avg)
        if rounds>=40 and (r+1)%20==0:
            p=predict(m,data['x'][data['dev']]);acc=float((p.argmax(1)==data['y'][data['dev']]).float().mean())
            log(event='training',dataset=data['dataset'],alpha=data['alpha'],seed=seed,backdoor=data['backdoor'],exclude=exclude,offset=offset,round=r+1,dev_acc=acc)
    co.dl(len(flat(m)));torch.cuda.synchronize()
    return m,co.report(),time.time()-begin

def recovery(m,data,seed,direction,rounds,cost,snr=20):
    ids=list(range(1,20));weights=torch.tensor(data['counts'][1:],device=DEVICE,dtype=torch.float32);weights/=weights.sum()
    ch=AirComp(snr,seed+70000000,cost);u=normalize(direction) if direction is not None else None
    for r in range(rounds):
        cost.dl(len(flat(m)));vectors=[]
        for i in ids:
            x,y=batch(data,i,seed*100000+60000000+r*1000+i,128);g=gradient(m,x,y)
            if u is not None:g=g-(g@u)*u
            vectors.append(g);cost.work(128)
        avg,_,_=ch.transmit(torch.stack(vectors),1.,weights)
        assign(m,flat(m)-.03*avg)
    return m

METHODS=[
    dict(name='digital_svd',kind='svd',snr=None,ratio=1.),
    dict(name='full_iter_ideal_K4',kind='iter',snr=None,ratio=1.),
    dict(name='full_iter_20_K4',kind='iter',snr=20,ratio=1.),
    dict(name='sumorth_20',kind='sum',snr=20,ratio=1.),
    dict(name='ga_20',kind='ga',snr=20,ratio=1.),
    dict(name='sketch_exact_10',kind='sketch_svd',snr=None,ratio=.1),
    dict(name='sketch_ideal_10_K4',kind='iter',snr=None,ratio=.1),
    dict(name='sketch_ideal_100_K4',kind='iter',snr=None,ratio=.01),
    dict(name='proposed_20_10_K4',kind='iter',snr=20,ratio=.1),
    dict(name='proposed_20_100_K4',kind='iter',snr=20,ratio=.01),
    dict(name='proposed_0_10_K4',kind='iter',snr=0,ratio=.1),
]

def unlearn(source,data,seed,method):
    m=clone(source);cost=Cost();D=len(flat(m));cost.dl(D)
    sk=Sketch(D,max(1,int(D*method['ratio'])),seed+800000)
    ch=AirComp(method['snr'],seed+500000,cost);traces=[];last=torch.zeros(D,device=DEVICE)
    begin=time.time()
    for t in range(20):
        x,y=batch(data,0,seed*100000+40000000+t*1000,128)
        gu=gradient(m,x,y,uce=method['kind']!='ga');cost.work(128)
        retained=[]
        if method['kind']!='ga':
            for i in range(1,20):
                x,y=batch(data,i,seed*100000+40000000+t*1000+i,128)
                retained.append(gradient(m,x,y));cost.work(128)
        info={}
        if method['kind']=='iter':d,info=iterative_projection(gu,retained,sk,4,ch)
        elif method['kind']=='svd':
            d=normalize(exact_null(torch.stack(retained),-gu))*gu.norm()
            cost.digital(20*D)
        elif method['kind']=='sketch_svd':
            z=sk.forward(torch.stack(retained));v=exact_null(z,-sk.forward(gu))
            d=normalize(sk.adjoint(v))*gu.norm();cost.digital(20*sk.s+1)
        elif method['kind']=='sum':
            total,_,_=ch.transmit(normalize(torch.stack(retained)),1.)
            q=normalize(total);d=normalize(-gu+(gu@q)*q)*gu.norm();cost.digital(D+1)
        elif method['kind']=='ga':
            d,_,_=ch.transmit(gu[None],1.);cost.digital(1)
        else:raise ValueError(method)
        if not torch.isfinite(d).all():raise ArithmeticError('Nonfinite direction')
        if retained:
            oracle=exact_null(torch.stack(retained),-gu)
            conflict=float((normalize(torch.stack(retained))@normalize(d)).square().mean().sqrt())
            info.update(cosine_to_full_svd=cosine(d,oracle),retained_cosine_rms=conflict)
        info.update(round=t+1,gu_norm=float(gu.norm()),direction_norm=float(d.norm()))
        traces.append(info)
        if info.get('floor_stop'):break
        last=d;assign(m,flat(m)+.2*d)
        # All clients can reconstruct the full update from the final sketch and norm;
        # full-vector methods require a full update broadcast.
        if method['kind']!='iter':cost.dl(D)
    before_recovery=clone(m);before_cost=cost.report()
    m=recovery(m,data,seed,last,20,cost,snr=20)
    cost.dl(D);torch.cuda.synchronize()
    return m,cost.report(),time.time()-begin,traces,before_recovery,before_cost

def auc(pos,neg):
    diff=pos[:,None]-neg[None,:]
    return float(((diff>0).float()+.5*(diff==0).float()).mean())

@torch.no_grad()
def evaluation_cache(m,data):
    sets={'forget':data['x'][data['pools'][0]],'retain':data['x'][data['retain']],
          'test':data['tx'],'member':data['mx'],'nonmember':data['nx']}
    if data['backdoor']:
        tx=data['tx'][data['ty']!=0].clone();tx[:,:,-4:-1,-4:-1]=1;sets['trigger']=tx
    return {k:predict(m,v) for k,v in sets.items()}

def metrics(cache,ref,data,vector,refvector):
    r={}
    for k,y in [('forget',data['y'][data['pools'][0]]),('retain',data['y'][data['retain']]),('test',data['ty'])]:
        p=cache[k].clamp_min(1e-12);q=ref[k].clamp_min(1e-12);mid=(p+q)/2
        r[k+'_acc']=float((p.argmax(1)==y).float().mean())
        r[k+'_js']=float((.5*(p*(p.log()-mid.log())+q*(q.log()-mid.log()))).sum(1).mean())
        r[k+'_kl']=float((q*(q.log()-p.log())).sum(1).mean())
        r[k+'_disagreement']=float((p.argmax(1)!=q.argmax(1)).float().mean())
    member=cache['member'].gather(1,data['my'][:,None]).clamp_min(1e-12).log().flatten()
    nonmember=cache['nonmember'].gather(1,data['ny'][:,None]).clamp_min(1e-12).log().flatten()
    r['loss_mia_auc']=auc(member,nonmember);r['mia_count']=len(member)
    r['parameter_relative_l2']=float((vector-refvector).norm()/refvector.norm())
    if data['backdoor']:r['asr']=float((cache['trigger'].argmax(1)==0).float().mean())
    return r

class Power:
    def __init__(self):self.samples=[];self.stop=threading.Event()
    def sample(self):
        while not self.stop.is_set():
            try:
                v=subprocess.check_output(['nvidia-smi','--query-gpu=power.draw,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,creationflags=0x08000000).strip().split(',')
                self.samples.append([time.time()]+list(map(float,v)))
            except Exception:pass
            self.stop.wait(5)
    def __enter__(self):self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()
    def report(self):
        wh=sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(self.samples,self.samples[1:]))/3600
        return dict(gpu_board_Wh=wh,samples=self.samples,sampling_seconds=5,external_payment=0,
                    scope='whole GPU board; not whole host, electricity tariff unavailable')
