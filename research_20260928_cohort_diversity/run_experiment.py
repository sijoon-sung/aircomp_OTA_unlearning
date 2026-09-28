"""Cohort-preserving frequency selection / causal waiting for DS-Air.

Reuses the frozen ridge/PHY implementation, not its stored measurements.
All budgets include acquisition overhead. No retained client is discarded.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import time
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'
BASEFILE = ROOT.parent/'research_20260928_ds_revision/run_experiment.py'
spec = importlib.util.spec_from_file_location('frozen_ds', BASEFILE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
EXPECTED_BASE_HASH = 'aae15d2d281a20f6e02bf4d2c18b194874727c610808404630bbf1520c3528d6'
assert base.sha(BASEFILE) == EXPECTED_BASE_HASH
base.MC = 16
_old_utility = base.utility
base.utility = lambda ws, *args: _old_utility(ws[:1], *args)
SEEDS = [202609284, 202609285, 202609286]
D, RANK, DRAWS, K, TAU, MAX_TRIES = 64, 48, 32, 9, .4, 8
CONDITIONS = [
    dict(name='iid20', snr=20, corr=0., csi=0., weak=1.),
    dict(name='iid30', snr=30, corr=0., csi=0., weak=1.),
    dict(name='correlated20', snr=20, corr=.9, csi=0., weak=1.),
    dict(name='static20', snr=20, corr=1., csi=0., weak=1.),
    dict(name='csi05_20', snr=20, corr=0., csi=.05, weak=1.),
    dict(name='weak_client20', snr=20, corr=0., csi=0., weak=.2),
]
METHODS = ['Original', 'V1', 'SubspaceV1', 'FD2-V1', 'FD8-Original',
           'FD8-V1', 'FD8-SubspaceV1', 'CG-V1', 'CG-SubspaceV1']


def hashes():
    return dict(code_sha256=base.sha(__file__), protocol_sha256=base.sha(ROOT/'PROTOCOL_KO.md'),
                reused_code_sha256=base.sha(BASEFILE))


def bank(seed, corr=0., csi=0., weak=1.):
    common = base.normal((1, K, 2), seed)
    independent = base.normal((8, K, 2), seed+1)
    g = math.sqrt(corr)*common+math.sqrt(1-corr)*independent
    h = g.square().sum(-1).div(2).sqrt()
    h[:, 0] *= weak
    he = h*(csi*base.normal((8, K), seed+2)).exp()
    return h, he


def policy(method, he):
    if method.startswith('FD'):
        n = 2 if method.startswith('FD2') else 8
        return int(he[:n].amin(1).argmax()), n, True
    if method.startswith('CG'):
        eligible = torch.where(he.amin(1) >= TAU)[0]
        return (int(eligible[0]), int(eligible[0])+1, True) if len(eligible) else (-1, 8, False)
    return 0, 1, True


def probe_cost(snr):
    return base.CP*(K*8+K*32/base.rate(snr))


def extra_cost(method, n, snr):
    search = method.startswith(('FD', 'CG'))
    extra_pilots = (n-1)*K*8
    bits = n*K*32+(67 if search else 0)
    return dict(total=base.CP*(extra_pilots+bits/base.rate(snr)),
                extra_pilot_real=extra_pilots, extra_digital_bits=bits,
                acquisition_total=n*probe_cost(snr)+(base.CP*67/base.rate(snr) if search else 0))


def finish_cost(method, n, snr, active, budget, completed):
    # Previous failed probes have already occurred in earlier coherence windows.
    previous_probes = (n-1)*probe_cost(snr) if method.startswith('CG') else 0.
    last_active = active-previous_probes
    latency = {}
    for multiplier in [1, 4, 16]:
        window = multiplier*budget
        if method.startswith('CG'):
            value = (n-1)*window+last_active if completed else n*window
        else:
            value = active
        latency[str(multiplier)] = value
    return dict(active_total=active, latency=latency, previous_probe_airtime=previous_probes,
                coherence_window_base=budget, complete=completed)


def pending(method, n, snr, budget, source_accuracy, rank):
    bits = 256+67+n*K*32
    active = base.CP*(n*K*8+bits/base.rate(snr))
    met = dict(MSE_ratio=1., theoretical_MSE_ratio=1., bias_ratio=1., variance_ratio=0.,
               MonteCarlo_SE_ratio=0., deployed_MSE_ratio=1., test_accuracy=source_accuracy,
               forget_JS_ratio=1., CSI_mean_shift_norm=0.)
    return dict(method=method, status='deletion_pending', rank=rank, metrics=met, power=dict(mean=0., peak=0.),
                attack=None, participating_clients=0, retained_target_clients=K, budget=budget,
                selected_band=-1, attempts=n, cost=finish_cost(method,n,snr,active,budget,False),
                timeout_cost=dict(pilot_real=n*K*8, digital_bits=bits), correction_repeats=0,
                unlearning_success=False, note='No model release/update; deletion remains pending.')


@torch.no_grad()
def math_checks():
    samples = 100000
    h = base.normal((samples,8,K,2), 88001).square().sum(-1).div(2).sqrt()
    minimum = h.square().amin(2)
    statistics = {}
    for count in [2,4,8]:
        best = minimum[:,:count].amax(1)
        expected = -K*count*sum((-1)**j*math.comb(count-1,j)*math.log(j+1) for j in range(count))
        empirical = float((1/best).mean())
        xs = torch.tensor([.02,.05,.1,.2,.5], device=base.DEV, dtype=base.DT)
        observed = (best[:,None] <= xs[None]).double().mean(0)
        cdf = (1-(-K*xs).exp()).pow(count)
        error = float((observed-cdf).abs().max())
        assert error < .008
        if count >= 4:
            assert abs(empirical/expected-1) < .04
        statistics[str(count)] = dict(expected_inverse_minimum=expected, MC_inverse_minimum=empirical,
                                     maximum_CDF_error=error)
    eligible = minimum >= TAU**2
    complete = eligible.any(1)
    p = math.exp(-K*TAU**2)
    completion = 1-(1-p)**8
    observed = float(complete.double().mean())
    assert abs(completion-observed) < .008
    attempts = torch.where(complete,eligible.int().argmax(1)+1,8)
    expected_attempts = completion/p
    assert abs(float(attempts.double().mean())-expected_attempts) < .03
    # No future inspection in CG: first admissible channel, not best of bank.
    dummy = torch.tensor([[.1]*K,[.5]*K,[2.]*K]+[[.1]*K]*5,device=base.DEV)
    assert policy('CG-V1',dummy)==(1,2,True) and policy('FD8-V1',dummy)==(2,8,True)
    assert policy('CG-V1',dummy*.1)==(-1,8,False)
    hs, he = bank(88101)
    chosen, _, _ = policy('FD8-V1',he)
    z = base.normal((K,3,80),88102)
    mean, scale, _, power, tx = base.prepare(z,hs[chosen],he[chosen])
    mac_error = float((mean-z.sum(0)).abs().max())
    assert mac_error < 1e-12 and power['mean'] <= 1+1e-10 and power['peak'] <= 4+1e-10
    eye = torch.eye(12,device=base.DEV,dtype=base.DT)
    a = base.normal((10,12,12),88103)
    hh = a@a.transpose(-1,-2)+eye
    cc = base.normal((10,12,3),88104)
    w0 = torch.linalg.solve(hh.mean(0),cc.mean(0))
    wr = torch.linalg.solve(hh[1:].mean(0),cc[1:].mean(0))
    b = (hh[1:]@w0-cc[1:]).mean(0)
    exact = w0-torch.linalg.solve(hh[1:].mean(0),b)
    identity = float((exact-wr).abs().max())
    assert identity < 1e-12
    same1 = bank(88201,csi=0.)[0]
    same2 = bank(88201,csi=.05)[0]
    assert torch.equal(same1,same2)
    static,_ = bank(88202,corr=1.)
    assert torch.equal(static,static[:1].expand_as(static))
    budget=base.ledger(D,D,20,32,'Original',[8]*8)['total']+extra_cost('Original',1,20)['total']
    ledger_checks={}
    for method in METHODS:
        kind=method.split('-')[-1]
        r=RANK if kind=='SubspaceV1' else D
        ex=extra_cost(method,8 if method.startswith(('FD8','CG')) else 2 if method.startswith('FD2') else 1,20)
        reps=base.repeat_budget(D,r,20,32,kind,budget-ex['total'])
        assert reps>=r//8
        ledger_checks[method]=reps
    check=dict(passed=True,selection_statistics=statistics,gate_single_try_probability=p,
               gate_completion_exact=completion,gate_completion_MC=observed,
               gate_attempts_exact=expected_attempts,gate_attempts_MC=float(attempts.double().mean()),
               all_client_MAC_error=mac_error,power=power,quadratic_deletion_identity=identity,
               CSI_conditions_paired=True,static_bank_identical=True,available_repeat_budgets=ledger_checks,
               **hashes())
    base.write(OUT/'math_checks.json',check)
    print(json.dumps(check),flush=True)


@torch.no_grad()
def run(paths):
    check=json.loads((OUT/'math_checks.json').read_text())
    assert check['passed'] and all(check[k]==v for k,v in hashes().items())
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    total=0
    manifests={}
    for ds_index,(ds,path) in enumerate(paths.items()):
        x,y,xt,yt=base.dataset(path)
        manifests[ds]={p.name:base.sha(p) for p in sorted(path.glob('*ubyte'))}
        proj=torch.randn((784,D-1),generator=base.rng(202609280+D),device=base.DEV)/math.sqrt(784)
        z=torch.cat([F.relu((x-.5)@proj),torch.ones(len(x),1,device=base.DEV)],1).double()
        zt=torch.cat([F.relu((xt-.5)@proj),torch.ones(len(xt),1,device=base.DEV)],1).double()
        yy=F.one_hot(y,10).double()
        eye=torch.eye(D,device=base.DEV,dtype=base.DT)
        u=torch.linalg.qr(base.normal((D,RANK),2026092800+D)).Q
        for seed in SEEDS:
            pools=base.partition(y,seed)
            hs=torch.stack([z[p].T@z[p]/len(p)+base.LAM*eye for p in pools])
            cs=torch.stack([z[p].T@yy[p]/len(p) for p in pools])
            w0=base.fp32(torch.linalg.solve(hs.mean(0),cs.mean(0)))
            wr=torch.linalg.solve(hs[1:].mean(0),cs[1:].mean(0))
            bs=hs[1:]@w0-cs[1:]
            ga=hs[0]@w0-cs[0]
            subh,subb=u.T@hs[1:]@u,u.T@bs
            zf=z[pools[0]]
            refp=torch.softmax(zf@wr,-1)
            js0=float(base.js(torch.softmax(zf@w0,-1),refp).mean())
            eval_data=(zt,yt,zf,refp,js0)
            refs=dict(source_accuracy=float(((zt@w0).argmax(-1)==yt).double().mean()),
                      retained_accuracy=float(((zt@wr).argmax(-1)==yt).double().mean()),
                      initial_parameter_gap=float((w0-wr).square().sum()),initial_forget_JS=js0)
            rows=[]
            for config in CONDITIONS:
                snr=config['snr']
                ceiling=base.ledger(D,D,snr,32,'Original',[8]*8)['total']+extra_cost('Original',1,snr)['total']
                for draw in range(DRAWS):
                    # Same actual channels/AWGN across scenario changes; source-specific draw bank.
                    ns=seed*100000+ds_index*10000+draw*100
                    h,he=bank(ns,config['corr'],config['csi'],config['weak'])
                    cache={}
                    for method in METHODS:
                        kind=method.split('-')[-1]
                        small=kind=='SubspaceV1'
                        r=RANK if small else D
                        selected,n,complete=policy(method,he)
                        if not complete:
                            row=pending(method,n,snr,ceiling,refs['source_accuracy'],r)
                        else:
                            key=(selected,small)
                            if key not in cache:
                                cache[key]=base.basis(subh if small else hs[1:],h[selected],he[selected],snr,32,
                                                     ns+21 if small else ns+20)
                            eig,q,hd=cache[key]
                            ex=extra_cost(method,n,snr)
                            row=base.evaluate_method(kind,D,r,eig,q,u if small else eye,
                                      subh if small else hs[1:],subb if small else bs,cs[1:],w0,wr,ga,
                                      h[selected],he[selected],snr,32,ns+40,ceiling-ex['total'],eval_data)
                            assert row['status']=='ok'
                            ledger=row.pop('cost')
                            active=ledger['total']+ex['total']
                            row.update(method=method,participating_clients=K,retained_target_clients=K,
                                       budget=ceiling,base_ledger=ledger,search_cost=ex,
                                       cost=finish_cost(method,n,snr,active,ceiling,True),
                                       correction_repeats=sum(ledger['repeats']),selected_band=selected,attempts=n,
                                       curvature=hd,unlearning_success=row['metrics']['MSE_ratio']<1,
                                       selected_actual_min_gain=float(h[selected].min()),
                                       selected_estimated_min_gain=float(he[selected].min()))
                        row.update(dataset=ds,seed=seed,condition=config['name'],SNR=snr,draw=draw,
                                   band_correlation=config['corr'],CSI_error=config['csi'],weak_gain=config['weak'])
                        assert row['cost']['active_total']<=ceiling+1e-7
                        rows.append(row)
                print(json.dumps(dict(dataset=ds,seed=seed,condition=config['name'],rows=len(rows))),flush=True)
            total+=len(rows)
            base.write(OUT/f'{ds}_seed{seed}.json',dict(complete=True,references=refs,rows=rows,**hashes()))
        del x,y,xt,yt,z,zt,yy
    torch.cuda.synchronize()
    completion=dict(complete=True,rows=total,elapsed_seconds=time.perf_counter()-start,
                    peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
                    torch_version=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(),
                    seeds=SEEDS,channel_draws=DRAWS,correction_MC=base.MC,utility_MC=1,data=manifests,**hashes())
    base.write(OUT/'completion.json',completion)
    print(json.dumps(completion),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['math','run'],required=True)
    ap.add_argument('--fashion',type=Path)
    ap.add_argument('--mnist',type=Path)
    args=ap.parse_args()
    assert torch.cuda.is_available()
    if args.stage=='math':
        math_checks()
    else:
        assert args.fashion and args.mnist
        run({'FashionMNIST':args.fashion,'MNIST':args.mnist})
