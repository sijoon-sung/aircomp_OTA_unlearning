from pathlib import Path
import json, math, time, hashlib, os
import numpy as np
import torch
import torch.nn.functional as F
from vendor_adapters import ConvNet2, orth_dist

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'; RESULTS.mkdir(exist_ok=True)
DATA=Path('C:/Users/DISLAB/Desktop/split_learning/ota_ful/data/FashionMNIST/raw')
DEVICE='cuda'
CHANNELS=[.7,1.,.8,1.2,.9,1.1,.75,1.05,.85,1.15]
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

def write(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def log(**obj): print(json.dumps(obj,ensure_ascii=False),flush=True)
def rng(seed):return torch.Generator(device=DEVICE).manual_seed(int(seed))
def sync():torch.cuda.synchronize()
def flat(model):return torch.cat([p.detach().reshape(-1) for p in model.parameters()])
def flatten(ts):return torch.cat([t.reshape(-1) for t in ts])
def assign(model,vector):
    offset=0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(vector[offset:offset+p.numel()].view_as(p));offset+=p.numel()
    assert offset==len(vector)
    return model
def new_model(seed):
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        m=ConvNet2(28,10,in_channels=1,hidden=32,n_kernels=8).to(DEVICE)
    return m
def clone_model(model):return assign(new_model(0),flat(model))
def vec_hash(v):return hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()

def load_data(seed):
    def read(prefix):
        x=np.fromfile(DATA/(prefix+'-images-idx3-ubyte'),dtype=np.uint8,offset=16).copy().reshape(-1,1,28,28)
        y=np.fromfile(DATA/(prefix+'-labels-idx1-ubyte'),dtype=np.uint8,offset=8).copy()
        return torch.tensor(x,device=DEVICE,dtype=torch.float32)/255.,torch.tensor(y,device=DEVICE,dtype=torch.long)
    x,y=read('train');tx,ty=read('t10k')
    g=rng(seed+123000);pools=[[] for _ in range(10)];public=[];val=[]
    for c in range(10):
        ids=torch.where(y==c)[0];ids=ids[torch.randperm(len(ids),generator=g,device=DEVICE)]
        for i in range(10):pools[i].append(ids[i*60:(i+1)*60])
        pools[c].append(ids[600:900]);pools[(c-1)%10].append(ids[900:1200])
        public.append(ids[1200:1400]);val.append(ids[1400:1600])
    pools=[torch.cat(p) for p in pools];public=torch.cat(public);val=torch.cat(val)
    allids=torch.cat(pools+[public,val]);assert allids.unique().numel()==16000
    private=torch.cat(pools);retained=torch.cat(pools[1:]);forget=pools[0]
    nonmember=[]
    for c in range(10):nonmember.append(torch.where(ty==c)[0][:int((y[forget]==c).sum())])
    return dict(x=x,y=y,tx=tx,ty=ty,pools=pools,public=public,val=val,
                private=private,retained=retained,forget=forget,nonmember=torch.cat(nonmember))

def batch(data,client,generator,n=64):
    pool=data['pools'][client]
    ix=pool[torch.randint(len(pool),(n,),device=DEVICE,generator=generator)]
    return data['x'][ix],data['y'][ix]

class Cost:
    def __init__(self,snr=20):
        self.snr=snr;self.c={k:0. for k in ['ul_payload_real','dl_bits','metadata_bits','pilot_real',
            'client_gradient_samples','client_hvp_samples','teacher_forward_samples','student_train_samples']}
        self.messages=0
    @property
    def rate(self):return .5*math.log2(1+10**(self.snr/10))
    def digital(self,bits):self.c['dl_bits']+=bits
    def packet(self,dim,nclients,repeats=1):
        self.c['ul_payload_real']+=float(torch.as_tensor(repeats).sum()) if torch.as_tensor(repeats).ndim else dim*float(repeats)
        self.c['metadata_bits']+=32*nclients
        self.c['pilot_real']+=8*nclients;self.messages+=1
    def total(self):return 1.125*(self.c['ul_payload_real']+self.c['pilot_real']+(self.c['dl_bits']+self.c['metadata_bits'])/self.rate)
    def report(self):return dict(self.c,messages=self.messages,total_real_uses=self.total(),snr_db=self.snr,
                                cp_factor=1.125,scope='equalized real MAC; modeled communication, measured simulator time separately')

def ota(vectors,weights,clients,snr,generator,repeats=1,cost=None):
    """Receiver returns only the aggregate. Norms allowed, individual vectors simulator-private."""
    d=vectors[0].numel()
    weighted=[v.reshape(-1)*w for v,w in zip(vectors,weights)]
    # Explicit channel inversion: x_i=weighted_i/(h_i*scale), y=scale*(sum h_i*x_i+z).
    h=torch.tensor([CHANNELS[i] for i in clients],device=DEVICE,dtype=weighted[0].dtype)
    scale=torch.stack([v.square().mean().sqrt()/h[i] for i,v in enumerate(weighted)]).max().clamp_min(1e-12)
    received=torch.zeros_like(weighted[0])
    for i,v in enumerate(weighted):received+=h[i]*(v/(h[i]*scale))
    result=scale*received
    if snr is not None:
        r=torch.as_tensor(repeats,device=DEVICE,dtype=result.dtype)
        result+=scale/((10**(snr/10)*r).sqrt())*torch.randn(d,device=DEVICE,dtype=result.dtype,generator=generator)
    if cost is not None:cost.packet(d,len(clients),repeats)
    return result

def ortho_loss(model):return orth_dist(model.conv1.weight).square()+orth_dist(model.conv2.weight).square()

@torch.no_grad()
def probabilities(model,x,temperature=1.):
    model.eval()
    return torch.cat([F.softmax(model(b)/temperature,dim=1) for b in x.split(256)])

def accuracy(model,x,y):return float((probabilities(model,x).argmax(1)==y).float().mean())

def js(p,q):
    p=p.clamp_min(1e-12);q=q.clamp_min(1e-12);m=(p+q)/2
    return .5*((p*(p.log()-m.log())).sum(1)+(q*(q.log()-m.log())).sum(1))

def auc_score(pos,neg):
    # Exact pairwise AUC with tied values; 1200 x 1200 only.
    v=pos[:,None]-neg[None,:]
    return float(((v>0).float()+.5*(v==0).float()).mean())

@torch.no_grad()
def evaluate(model,reference,data,source_probs=None):
    sets={'forget':(data['x'][data['forget']],data['y'][data['forget']]),
          'retain':(data['x'][data['retained'][::4]],data['y'][data['retained'][::4]]),
          'test':(data['tx'],data['ty'])}
    out={};cache={}
    for name,(x,y) in sets.items():
        p=probabilities(model,x);q=probabilities(reference,x);cache[name]=p
        out[name+'_accuracy']=float((p.argmax(1)==y).float().mean())
        out[name+'_reference_accuracy']=float((q.argmax(1)==y).float().mean())
        out[name+'_js']=float(js(p,q).mean())
        if source_probs is not None:
            denom=float(js(source_probs[name],q).mean())
            out[name+'_noop_js']=denom
            out[name+'_normalized_js']=out[name+'_js']/max(denom,1e-12)
    p=cache['forget'];yf=sets['forget'][1]
    nm=data['nonmember'];q=probabilities(model,data['tx'][nm]);yn=data['ty'][nm]
    lp=p[torch.arange(len(p),device=DEVICE),yf].clamp_min(1e-12).log()
    ln=q[torch.arange(len(q),device=DEVICE),yn].clamp_min(1e-12).log()
    out['loss_mia_auc']=auc_score(lp,ln)
    return out,cache

def train_fl(data,seed,rounds,ortho=False,exclude=False,sampling_offset=0,start=None,start_round=0):
    model=new_model(seed) if start is None else start
    local=new_model(seed);active=list(range(1 if exclude else 0,10));cost=Cost()
    generators={i:rng(seed*1000+i+sampling_offset) for i in active}
    # Resume calibration replays minibatch draws to preserve its exact schedule.
    for i,g in generators.items():
        for _ in range(start_round*4):torch.randint(len(data['pools'][i]),(64,),device=DEVICE,generator=g)
    begin=time.perf_counter();model.train();dim=flat(model).numel()
    for r in range(start_round,rounds):
        for phase in range(2 if ortho else 1):
            anchor=flat(model);cost.digital(dim*32);updates=[]
            for i in active:
                assign(local,anchor);local.train()
                for _ in range(2 if ortho else 4):
                    bx,by=batch(data,i,generators[i]);loss=F.cross_entropy(local(bx),by)
                    if ortho:
                        loss=loss+.001*ortho_loss(local)
                        if phase==1:loss=loss+.0001*sum((p-a).square().sum() for p,a in zip(local.parameters(),split_params(local,anchor)))
                    grads=torch.autograd.grad(loss,tuple(local.parameters()))
                    with torch.no_grad():
                        for p,g in zip(local.parameters(),grads):p.add_(g,alpha=-.03)
                    cost.c['client_gradient_samples']+=64
                updates.append(flat(local)-anchor)
            update=ota(updates,[1/len(active)]*len(active),active,None,None,cost=cost)
            assign(model,anchor+update)
        if (r+1)%25==0:
            va=accuracy(model,data['x'][data['val']],data['y'][data['val']])
            log(stage='training',seed=seed,ortho=ortho,exclude=exclude,round=r+1,val_accuracy=va)
    sync();elapsed=time.perf_counter()-begin
    return model,cost.report(),elapsed

def split_params(model,v):
    arr=[];o=0
    for p in model.parameters():arr.append(v[o:o+p.numel()].view_as(p));o+=p.numel()
    return tuple(arr)

def save_model(path,model):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)

def load_model(path):
    m=new_model(0);m.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True));return m
