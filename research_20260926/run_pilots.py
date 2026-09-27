"""Bounded CUDA pilots. See PROTOCOL.md; original experiments are read-only."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
DATA = Path(r'C:\Users\DISLAB\Desktop\split_learning\ota_ful\data\FashionMNIST\raw')
SEEDS = (202609260, 202609261, 202609262)
DEV = 'cuda'
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def log(**kw):
    print(json.dumps(kw), flush=True)


def gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def load_data():
    def read(prefix):
        x = np.fromfile(DATA / (prefix + '-images-idx3-ubyte'), dtype=np.uint8, offset=16).copy().reshape(-1, 784)
        y = np.fromfile(DATA / (prefix + '-labels-idx1-ubyte'), dtype=np.uint8, offset=8).copy()
        return torch.tensor(x, device=DEV, dtype=torch.float32) / 255., torch.tensor(y, device=DEV, dtype=torch.long)
    return (*read('train'), *read('t10k'))


def partition(y, seed):
    # Each class has two owners. No reused sample within or between clients.
    gg = gen(seed)
    pools = [[] for _ in range(10)]
    for label in range(10):
        ids = torch.where(y == label)[0]
        ids = ids[torch.randperm(len(ids), device=DEV, generator=gg)[:1000]]
        pools[label].append(ids[:500])
        pools[(label - 1) % 10].append(ids[500:])
    pools = [torch.cat(x) for x in pools]
    assert len(torch.unique(torch.cat(pools))) == 10000
    return pools


def greedy(coeff, budget):
    # Exact integer optimum for equal-sized blocks, sum a_j/r_j, r_j >= 1.
    vals = coeff.detach().cpu().double().tolist()
    r = [1] * len(vals)
    for _ in range(budget - len(vals)):
        j = max(range(len(vals)), key=lambda k: vals[k] / (r[k] * (r[k] + 1)))
        r[j] += 1
    return torch.tensor(r, device=DEV, dtype=coeff.dtype)


def rate(snr):
    # Explicit abstract half-Shannon goodput. No coded modem claim.
    return .5 * math.log2(1 + 10 ** (snr / 10))


def a_uses(real_values):
    return math.ceil(real_values / 2) * 1.125


def math_checks():
    dtype = torch.float64
    hb = torch.tensor([[2., 1.], [1., 2.]], device=DEV, dtype=dtype)
    da = torch.tensor([1., 0.], device=DEV, dtype=dtype)
    eta = .1
    after_b = da - eta * (hb @ da)
    subtract = after_b - da
    expected = -eta * hb @ da
    g_b_at_start = torch.tensor([0., 1.], device=DEV, dtype=dtype)
    orthogonal_dot = float(g_b_at_start @ da)
    # Two retained gradients with the same aggregate and different spans.
    g1 = torch.tensor([[1., 0.], [0., 1.], [0., 0.]], device=DEV, dtype=dtype)
    g2 = torch.tensor([[1., 0.], [1., 0.], [1., -1.]], device=DEV, dtype=dtype)
    u = torch.tensor([0., 0., 1.], device=DEV, dtype=dtype)
    p1 = u - g1 @ torch.linalg.pinv(g1) @ u
    p2 = u - g2 @ torch.linalg.pinv(g2) @ u
    # Horizontal mixed activations represent a different loss even in scalar LS.
    z = torch.tensor([1., -1.], device=DEV, dtype=dtype)
    labels = torch.tensor([1., -1.], device=DEV, dtype=dtype)
    w = torch.tensor(0., device=DEV, dtype=dtype)
    original_grad = ((w * z - labels) * z).mean()
    mixed_grad = (w * z.mean() - labels.mean()) * z.mean()
    assert torch.allclose(subtract, expected)
    assert torch.allclose(g1.sum(1), g2.sum(1))
    assert torch.linalg.norm(p1 - p2) > .9
    # HVP: add locally computed H_i v, without transmitting local Hessians.
    hs = torch.stack([hb, hb + torch.eye(2, device=DEV, dtype=dtype)])
    hvp_err = float(torch.linalg.norm((hs @ da).mean(0) - hs.mean(0) @ da))
    write('math_checks.json', dict(historical_subtraction_residual=subtract.tolist(),
          residual_norm=float(subtract.norm()), first_order_orthogonal_dot=orthogonal_dot,
          same_aggregate_projection_difference=float((p1-p2).norm()),
          horizontal_individual_loss_gradient=float(original_grad),
          horizontal_mixed_loss_gradient=float(mixed_grad), hvp_linearity_error=hvp_err))


def curvature(x, y, xt, yt):
    # Encoder is independent of all client data and never trained.
    gg = gen(20260922)
    proj = torch.randn((784, 64), device=DEV, generator=gg) / math.sqrt(784)
    z = torch.cat([F.relu((x-.5) @ proj), torch.ones((len(x),1), device=DEV)], 1).double()
    zt = torch.cat([F.relu((xt-.5) @ proj), torch.ones((len(xt),1), device=DEV)], 1).double()
    yy = F.one_hot(y, 10).double()
    eye = torch.eye(65, device=DEV, dtype=torch.float64)
    ii = torch.triu_indices(65, 65, device=DEV)
    rows = []
    for seed in SEEDS:
        pools = partition(y, seed)
        hs = torch.stack([z[p].T @ z[p] / len(p) + .01*eye for p in pools])
        cs = torch.stack([z[p].T @ yy[p] / len(p) for p in pools])
        h, c = hs[1:].mean(0), cs[1:].mean(0)
        w0 = torch.linalg.solve(hs.mean(0), cs.mean(0))
        wr = torch.linalg.solve(h, c)
        local_b = hs[1:] @ w0 - cs[1:]
        b = local_b.mean(0)
        exact = w0 - torch.linalg.solve(h, b)
        assert (exact-wr).norm() < 1e-10
        initgap = float((w0-wr).square().sum())
        src_acc = float(((zt @ w0).argmax(1) == yt).double().mean())
        ref_acc = float(((zt @ wr).argmax(1) == yt).double().mean())
        scale_h = hs[1:,ii[0],ii[1]].square().mean(1).sqrt().max()
        scale_c = cs[1:].square().mean((1,2)).sqrt().max()
        for snr in (10,20,30):
            n0 = 10**(-snr/10)
            for noise in range(16):
                rng = gen(seed + snr*100 + noise)
                nh = torch.randn(ii.shape[1], device=DEV, dtype=h.dtype, generator=rng)*scale_h*math.sqrt(n0/32)/9
                eh = torch.zeros_like(h)
                eh[ii[0],ii[1]] = nh
                eh[ii[1],ii[0]] = nh
                hraw = h+eh
                lam, q = torch.linalg.eigh(hraw)
                lam = lam.clamp_min(.01)
                hhat = (q*lam) @ q.T
                common_noise = torch.randn((65,10), device=DEV, dtype=h.dtype, generator=rng)
                for scope, hh in [('exact_H_diagnostic', h), ('received_H', hhat)]:
                    eig, basis = torch.linalg.eigh(hh)
                    bj = torch.einsum('ij,kjl->kil', basis.T, local_b)
                    # Metadata is per-client block RMS, not a disclosed full gradient.
                    scale = bj.reshape(9,5,13,10).square().mean((2,3)).sqrt().max(0).values
                    variance = scale.square()*n0/81
                    rawcoeff = 130*variance
                    weighted = 10*variance*eig.reshape(5,13).pow(-2).sum(1)
                    reps = dict(uniform=torch.full((5,),8.,device=DEV,dtype=h.dtype),
                                gradient_mse=greedy(rawcoeff,40),
                                deletion_mse=greedy(weighted,40))
                    for method, rr in reps.items():
                        nn = common_noise * (variance/rr).sqrt().repeat_interleave(13)[:,None]
                        ww = w0 - basis @ ((basis.T @ b + nn)/eig[:,None])
                        diff = ww-wr
                        grad_ul = a_uses(130)*float(rr.sum())
                        h_ul = a_uses(2145)*32 if scope == 'received_H' else 0
                        basis_dl = 65*65*32/rate(snr)
                        model_dl = 2*650*32/rate(snr)  # source and delivered result
                        meta = (9*6*32)/rate(snr)
                        pilots = 9*8*6
                        rows.append(dict(seed=seed,snr=snr,noise=noise,scope=scope,method=method,
                            error_ratio=float(diff.square().sum())/initgap,
                            objective_gap=float((diff*(h@diff)).sum()/2),
                            accuracy=float(((zt@ww).argmax(1)==yt).double().mean()),
                            source_accuracy=src_acc,retrain_accuracy=ref_acc,
                            init_squared_gap=initgap,repeats=rr.int().tolist(),
                            expected_parameter_noise=float((weighted/rr).sum()),
                            expected_raw_noise=float((rawcoeff/rr).sum()),
                            hessian_ul=h_ul,gradient_ul=grad_ul,basis_dl=basis_dl,
                            model_dl=model_dl,metadata=meta,pilots=pilots,
                            total_uses=h_ul+grad_ul+basis_dl+model_dl+meta+pilots))
                nc = common_noise*scale_c*math.sqrt(n0/8)/9
                ww = torch.linalg.solve(hhat, c+nc)
                diff = ww-wr
                total = a_uses(2145)*32 + a_uses(650)*8 + 650*32/rate(snr) + 9*2*32/rate(snr) + 9*8*2
                rows.append(dict(seed=seed,snr=snr,noise=noise,scope='received_H',method='stats_retrain',
                    error_ratio=float(diff.square().sum())/initgap, objective_gap=float((diff*(h@diff)).sum()/2),
                    accuracy=float(((zt@ww).argmax(1)==yt).double().mean()),source_accuracy=src_acc,
                    retrain_accuracy=ref_acc,total_uses=total))
            log(stage='curvature', seed=seed, snr=snr)
        write('curvature_rows.json', rows)


WIDTHS = [16]*5+[32]*3+[64]*2


def initial(seed):
    gg = gen(seed)
    return [torch.randn((64,784),device=DEV,generator=gg)*math.sqrt(2/784), torch.zeros(64,device=DEV),
            torch.randn((256,64),device=DEV,generator=gg)*math.sqrt(2/64),torch.zeros(256,device=DEV),
            torch.randn((10,256),device=DEV,generator=gg)*math.sqrt(2/256),torch.zeros(10,device=DEV)]


def predict(state,x,width):
    w,b,v,c,u,d = state
    return F.linear(F.relu(F.linear(F.relu(F.linear(x,w[:width],b[:width])),v[:,:width],c)),u,d)


def local_split(state,x,y,ids,width):
    s = [state[0][:width].clone(),state[1][:width].clone(),state[2][:,:width].clone(),
         state[3].clone(),state[4].clone(),state[5].clone()]
    s = [p.detach().requires_grad_(True) for p in s]
    for batch in ids:
        w,b,v,c,u,d = s
        front = F.relu(F.linear(x[batch],w,b))
        transmitted = front.detach().half().float().requires_grad_(True)
        pred = F.linear(F.relu(F.linear(transmitted,v,c)),u,d)
        loss = F.cross_entropy(pred,y[batch])
        server_g = torch.autograd.grad(loss,[transmitted,v,c,u,d])
        returned = server_g[0].half().float()
        front_g = torch.autograd.grad(front,[w,b],grad_outputs=returned)
        with torch.no_grad():
            for p,g in zip(s, list(front_g)+list(server_g[1:])):
                p.sub_(.03*g)
    return [p.detach() for p in s]


def sfl(x,y,xt,yt,split):
    rows=[]
    for seed in SEEDS:
        pools=partition(y,seed)
        if split=='iid':
            ids=torch.cat(pools)
            ids=ids[torch.randperm(len(ids),device=DEV,generator=gen(seed+811))]
            pools=list(ids.reshape(10,1000).unbind(0))
        schedules=[]
        for rnd in range(80):
            schedules.append([p[torch.randint(len(p),(2,32),device=DEV,generator=gen(seed+10000*rnd+i))] for i,p in enumerate(pools)])
        configs=[('exact',20)]+[(m,s) for s in (10,20) for m in ('digital8','uniform','coverage')]
        for method,snr in configs:
            state=initial(seed)
            counters=dict(activation_ul=0.,cut_gradient_dl=0.,front_ul=0.,front_dl=0.,labels=0.,metadata=0.,pilots=0.)
            allmse=[]; tailmse=[]; repeats=[]
            for rnd in range(80):
                # FP32 server master; actual FP16 front-model broadcast and decoding.
                broadcast=[state[0].half().float(),state[1].half().float()]+state[2:]
                locals_=[local_split(broadcast,x,y,schedules[rnd][i],WIDTHS[i]) for i in range(10)]
                front=torch.stack([torch.cat([state[0],state[1][:,None]],1) for _ in range(10)])
                # Only active coordinates are populated with local deltas.
                front.zero_()
                for i,ws in enumerate(WIDTHS):
                    front[i,:ws,:784]=locals_[i][0]-broadcast[0][:ws]
                    front[i,:ws,784]=locals_[i][1]-broadcast[1][:ws]
                active=torch.tensor([10.,5.,2.,2.],device=DEV)
                blocks=front.reshape(10,4,16*785)
                scales=blocks.square().mean(2).sqrt().max(0).values.clamp_min(1e-12)
                ideal=blocks.sum(0)/active[:,None]
                rr=torch.full((4,),4.,device=DEV)
                if method=='coverage':
                    rr=greedy(scales.square()/active.square(),16)
                received=ideal.clone()
                if method in ('uniform','coverage'):
                    nn=torch.randn(ideal.shape,device=DEV,generator=gen(seed+100000*rnd+snr))
                    received+=nn*(scales/active/math.sqrt(10**(snr/10))/rr.sqrt())[:,None]
                    counters['front_ul']+=a_uses(16*785)*float(rr.sum())
                    counters['metadata']+=10*4*32/rate(snr)
                    counters['pilots']+=10*4*8
                elif method=='digital8':
                    quant=torch.zeros_like(blocks)
                    for i,ws in enumerate(WIDTHS):
                        for j in range(ws//16):
                            scale=blocks[i,j].abs().max().clamp_min(1e-12)/127
                            quant[i,j]=(blocks[i,j]/scale).round().clamp(-127,127)*scale
                    received=quant.sum(0)/active[:,None]
                    counters['front_ul']+=sum(WIDTHS)*785*8/rate(snr)
                    counters['metadata']+=sum(w//16 for w in WIDTHS)*32/rate(snr)
                    counters['pilots']+=10*8
                allmse.append(float((received-ideal).square().mean()))
                tailmse.append(float((received[2:]-ideal[2:]).square().mean()))
                repeats.append(rr.int().tolist())
                update=received.reshape(64,785)
                new=[state[0]+update[:,:784],state[1]+update[:,784],state[2].clone()]
                for j in range(64):
                    eligible=[i for i,w in enumerate(WIDTHS) if j<w]
                    new[2][:,j]=torch.stack([locals_[i][2][:,j] for i in eligible]).mean(0)
                new += [torch.stack([ls[j] for ls in locals_]).mean(0) for j in (3,4,5)]
                state=[t.detach() for t in new]
                assert all(torch.isfinite(t).all() for t in state)
                counters['activation_ul']+=2*32*sum(WIDTHS)*16/rate(snr)
                counters['cut_gradient_dl']+=2*32*sum(WIDTHS)*16/rate(snr)
                counters['front_dl']+=64*785*16/rate(snr)
                counters['labels']+=2*32*10*4/rate(snr)
                if rnd in (19,39,59,79):
                    log(stage='sfl',seed=seed,method=method,snr=snr,round=rnd+1)
            with torch.no_grad():
                accs={str(w):float((predict(state,xt,w).argmax(1)==yt).float().mean()) for w in (16,32,64)}
            # Count initial front broadcast, already same dimension as per-round DL.
            counters['front_dl']+=64*785*16/rate(snr)
            client_macs=80*2*32*sum(2*784*w for w in WIDTHS)
            full_same_width=80*2*32*sum(2*784*w+3*w*256+3*256*10 for w in WIDTHS)
            full_width64=80*2*32*10*(2*784*64+3*64*256+3*256*10)
            row=dict(seed=seed,method=method,snr=snr,split=split,accuracy=accs,mean_mse=float(np.mean(allmse)),
                tail_mse=float(np.mean(tailmse)),mean_repeats=np.mean(repeats,axis=0).tolist(),
                ledger=counters,total_uses=sum(counters.values()),client_macs=client_macs,
                full_same_width_macs=full_same_width,full_width64_macs=full_width64,
                state_sha=hashlib.sha256(b''.join(t.cpu().numpy().tobytes() for t in state)).hexdigest())
            rows.append(row);write('sfl_'+split+'_rows.json',rows)
            log(stage='sfl_done',seed=seed,method=method,snr=snr,acc=accs,total=row['total_uses'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['math','curvature','sfl','all'],default='all')
    parser.add_argument('--split',choices=['label2','iid'],default='label2')
    args=parser.parse_args()
    assert torch.cuda.is_available()
    t=time.perf_counter()
    tag=args.phase+'_'+args.split if args.phase=='sfl' else args.phase
    write('manifest_'+tag+'.json',dict(seeds=SEEDS,device=torch.cuda.get_device_name(0),torch=torch.__version__,
        script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha=hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest(),phase=args.phase,split=args.split))
    math_checks()
    if args.phase!='math':
        x,y,xt,yt=load_data()
        if args.phase in ('curvature','all'):curvature(x,y,xt,yt)
        if args.phase in ('sfl','all'):sfl(x,y,xt,yt,args.split)
    write('complete_'+tag+'.json',dict(seconds=time.perf_counter()-t,phase=args.phase,split=args.split))
