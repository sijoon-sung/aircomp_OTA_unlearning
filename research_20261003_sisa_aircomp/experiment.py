"""Finite preregistered SISA/AirComp experiment. One GPU worker, immutable outputs."""
from pathlib import Path
import argparse, hashlib, json, math, os, subprocess, sys, time, traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / 'research_20260929_pdf_ota_unlearning'))
from core import model, flat, assign, gradient, load_data, predict, gen, Power

EPS = 1e-4
SIGMA2 = .01

def key(*args):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, args)).encode()).digest()[:8], 'little') % (2**62)

def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def save(path, value):
    assert not Path(path).exists(), path
    torch.save(value, path)

def log(**value):
    print(json.dumps(value, ensure_ascii=False), flush=True)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def radio_audit(dest):
    rng = np.random.default_rng(10600)
    d, m = 64, 5
    delta = rng.normal(size=(m, d))
    delta *= rng.uniform(.01, .2, m)[:, None] / np.linalg.norm(delta, axis=1)[:, None]
    h = rng.uniform(.1, 1, m)
    findings = []
    for mode in ['fixed', 'norm']:
        bounds = np.ones(m) if mode == 'fixed' else np.linalg.norm(delta, axis=1)
        beta = float(np.max(bounds / (m*h*np.sqrt(d))))
        waves = delta / (m*h[:, None]*beta)
        received = beta*np.sum(h[:, None]*waves, axis=0)
        error = float(np.max(np.abs(received-delta.mean(axis=0))))
        r = max(1, math.ceil(d*beta**2*SIGMA2/EPS))
        variance = beta**2*SIGMA2/r
        noise = rng.normal(size=(25000, d))*math.sqrt(variance)
        measured = float(np.mean(np.sum(noise**2, axis=1)))
        expected = d*variance
        bins = np.zeros((3, 3*d))
        for j in range(3):
            bins[j, j*d:(j+1)*d] = received*(j+1)
        rx = bins.sum(axis=0)
        leakage = max(float(np.max(np.abs(rx[j*d:(j+1)*d]-received*(j+1)))) for j in range(3))
        row = dict(mode=mode, noiseless_maxabs=error, max_power=float(np.max(np.mean(waves**2, axis=1))),
                   repeats=r, mse_expected=expected, mse_measured=measured,
                   mse_relative_error=abs(measured/expected-1), disjoint_bin_leakage=leakage)
        assert error < 1e-10 and row['max_power'] <= 1+1e-12
        assert row['mse_relative_error'] < .05 and expected <= EPS and leakage == 0
        findings.append(row)
    # Work-conserving orthogonal FDMA never creates a spectral resource discount.
    loads = np.array([1., 2., 7.])
    dump(dest/'radio_audit.json', dict(rows=findings, tdma=float(loads.sum()),
         elastic_fdma=float(loads.sum()), fixed_fdma=float(3*loads.max()), passed=True))

def prob_js(a, b):
    a, b = np.maximum(a.astype(np.float64), 1e-30), np.maximum(b.astype(np.float64), 1e-30)
    a /= a.sum(axis=1, keepdims=True)
    b /= b.sum(axis=1, keepdims=True)
    mid = (a+b)/2
    return max(0., float(np.mean(np.sum(.5*a*np.log(a/mid)+.5*b*np.log(b/mid), axis=1))))

class Engine:
    def __init__(self, data, seed, K, method, rounds, stress=False):
        self.data, self.seed, self.K, self.method, self.rounds = data, seed, K, method, rounds
        self.local = model(seed, data['channels'])
        self.D = flat(self.local).numel()
        self.initial = [flat(model(key(seed,c,'init')%2**31, data['channels'])).clone() for c in range(K)]
        rng = np.random.default_rng(key(seed, 'slow_channel'))
        base = 10**(rng.uniform(-20, 0, 20)/20)
        if stress:
            labels = [int(torch.bincount(data['y'][p], minlength=10).argmax()) for p in data['pools']]
            order = np.lexsort((np.arange(20), np.array(labels)))
            base[order] = np.sort(base)
        self.base_h = base
        if method.startswith('channel'):
            order = np.argsort(base, kind='stable')
        else:
            order = np.random.default_rng(key(seed, 'routing')).permutation(20)
        self.groups = [list(map(int, g)) for g in np.array_split(order, K)]
        self.affected = next(c for c,g in enumerate(self.groups) if 0 in g)
        self.h = np.stack([base*np.random.default_rng(key(seed,t,'fading')).uniform(.85,1.15,20)
                           for t in range(rounds)])
        self.checkpoints = {}

    def physical_receive(self, w, ids, c, t, noise_branch):
        # Client updates remain inside this simulated medium; server gets one aggregate.
        clipped, norms = [], []
        for i in ids:
            assign(self.local, w)
            pool = self.data['pools'][i]
            for step in range(2):
                indices = pool[torch.randint(len(pool), (64,), device='cuda', generator=gen(key(self.seed,i,t,step,'batch')))]
                grad = gradient(self.local, self.data['x'][indices], self.data['y'][indices])
                assign(self.local, flat(self.local)-.05*grad)
            delta = flat(self.local)-w
            norm = float(delta.norm())
            delta = delta*min(1., 1/max(norm, 1e-30))
            norms.append(min(norm, 1.))
            clipped.append(delta)
        n = len(ids)
        h = self.h[t, ids]
        norms = np.array(norms)
        adaptive = self.method.endswith('norm')
        bounds = norms if adaptive else np.ones(n)
        beta = max(1e-20, float(np.max(bounds/(n*h*np.sqrt(self.D)))))
        repeats = max(1, math.ceil(self.D*beta**2*SIGMA2/EPS))
        var = beta**2*SIGMA2/repeats
        power = norms**2/(n*n*h*h*beta*beta*self.D)
        assert float(power.max()) <= 1.00001
        aggregate = torch.stack(clipped).mean(0)
        noise = torch.randn(self.D, device='cuda', generator=gen(key(self.seed,c,t,noise_branch,'noise')))*math.sqrt(var)
        received = aggregate+noise
        control_bits = 64*n+64+(32*n if adaptive else 0)
        pilots = 8*n
        ul = repeats*self.D
        dl_bits = 32*self.D
        # Normalized symbol energy, not RF Joules or physical wall time.
        energy = repeats*float(power.sum())*self.D
        event = dict(t=t, c=c, ids=ids, active=n, repeats=repeats, beta=beta,
                     expected_mse=self.D*var, actual_noise_norm2=float(noise.square().sum()),
                     max_client_symbol_power=float(power.max()), min_h=float(h.min()),
                     norm_scalar_count=n if adaptive else 0, ul_reals=ul, pilot_reals=pilots,
                     control_bits=control_bits, dl_bits=dl_bits,
                     total_re=ul+pilots+control_bits/2+dl_bits/2,
                     total_re_dl6=ul+pilots+control_bits/2+dl_bits/6,
                     ul_signal_energy=energy, ul_control_pilot_energy=control_bits/2+pilots,
                     local_calls=n, local_steps=2*n, gradient_samples=128*n)
        return received, event

    def fit(self, vectors, deleted=False, only=None, capture=False, label=''):
        vectors = [v.clone() for v in vectors]
        events = []
        torch.cuda.synchronize()
        began = time.perf_counter()
        for t in range(self.rounds):
            for c in (range(self.K) if only is None else [only]):
                if capture and t in (0, self.rounds//2):
                    self.checkpoints[f'{c}_{t}'] = vectors[c].cpu().clone()
                ids = [i for i in self.groups[c] if not(deleted and i == 0)]
                assert len(ids) >= 3
                branch = 1 if deleted and c == self.affected else 0
                update, event = self.physical_receive(vectors[c], ids, c, t, branch)
                vectors[c] = vectors[c]+update
                events.append(event)
            if (t+1)%40 == 0:
                log(event='fit_progress', phase=label, seed=self.seed, K=self.K, method=self.method, round=t+1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-began
        columns = ['ul_reals','pilot_reals','control_bits','dl_bits','total_re','total_re_dl6',
                   'ul_signal_energy','ul_control_pilot_energy','local_calls','local_steps',
                   'gradient_samples','norm_scalar_count']
        cost = {s: sum(e[s] for e in events) for s in columns}
        cost['ul_blocks'] = sum(e['repeats'] for e in events)
        cost['logical_aggregates'] = len(events)
        cost['tdma_delay_units'] = cost['total_re']
        cost['elastic_fdma_delay_units'] = cost['total_re']
        cost['fixed_fdma_delay_units'] = sum(self.K*max(e['total_re'] for e in events if e['t']==t)
                                               for t in range(self.rounds))
        cost['max_client_symbol_power'] = max(e['max_client_symbol_power'] for e in events)
        cost['max_expected_mse'] = max(e['expected_mse'] for e in events)
        cost['min_active_clients'] = min(e['active'] for e in events)
        cost['seconds'] = elapsed
        return vectors, dict(cost=cost, events=events)

    @torch.no_grad()
    def predictions(self, vectors):
        buckets = dict(dev=self.data['x'][self.data['dev']], test=self.data['tx'],
                       forgotten=self.data['x'][self.data['pools'][0]], retain=self.data['x'][self.data['retain']])
        out = {s:[] for s in buckets}
        for w in vectors:
            assign(self.local, w)
            for s,x in buckets.items():
                out[s].append(predict(self.local,x).cpu().numpy())
        return {s:np.mean(np.stack(v),axis=0) for s,v in out.items()}

def one_case(out, phase, seed, K, method, rounds, stress=False):
    name = f'{phase}_s{seed}_K{K}_{method}'
    folder = out/name
    folder.mkdir(exist_ok=False)
    log(event='case_start', case=name)
    began = time.perf_counter()
    data = load_data('FashionMNIST', .5, seed)
    engine = Engine(data,seed,K,method,rounds,stress)
    np.savez_compressed(folder/'partition.npz', original=data['original_indices'],
                        **{f'client_{i}':p for i,p in enumerate(data['original_pools'])})
    label_map = dict(dev=data['y'][data['dev']].cpu().numpy(), test=data['ty'].cpu().numpy(),
                     forgotten=data['y'][data['pools'][0]].cpu().numpy(), retain=data['y'][data['retain']].cpu().numpy())
    np.savez_compressed(folder/'labels.npz', **label_map)
    dump(folder/'config.json', dict(seed=seed,phase=phase,K=K,method=method,rounds=rounds,D=engine.D,
         stress=stress,groups=engine.groups,affected=engine.affected,deleted=0,base_h=engine.base_h.tolist(),
         counts=data['counts'],epsilon=EPS,sigma2=SIGMA2))
    source, src = engine.fit(engine.initial, capture=True, label='source')
    sp = engine.predictions(source)
    save(folder/'source_models.pt',[v.cpu() for v in source])
    save(folder/'checkpoints.pt',engine.checkpoints)
    np.savez_compressed(folder/'source_probabilities.npz', **sp)
    dump(folder/'source_ledger.json', src)
    ref, rf = engine.fit(engine.initial, deleted=True, label='full_reference')
    rp = engine.predictions(ref)
    save(folder/'reference_models.pt',[v.cpu() for v in ref])
    np.savez_compressed(folder/'reference_probabilities.npz', **rp)
    dump(folder/'reference_ledger.json',rf)
    start = [v.clone() for v in source]
    start[engine.affected] = engine.checkpoints[f'{engine.affected}_0'].to('cuda').clone()
    replay, ur = engine.fit(start, deleted=True, only=engine.affected, label='shard_replay')
    up = engine.predictions(replay)
    save(folder/'replay_models.pt',[v.cpu() for v in replay])
    np.savez_compressed(folder/'replay_probabilities.npz', **up)
    dump(folder/'replay_ledger.json',ur)
    equal = all(torch.equal(a,b) for a,b in zip(ref,replay))
    unaffected = all(torch.equal(source[c],ref[c]) and torch.equal(source[c],replay[c])
                     for c in range(K) if c != engine.affected)
    authorized = all(e['c']==engine.affected and 0 not in e['ids'] for e in ur['events'])
    pdiff = max(float((a-b).abs().max()) for a,b in zip(ref,replay))
    prediction_diff = max(float(np.max(np.abs(up[s]-rp[s]))) for s in up)
    assert equal and unaffected and authorized and prediction_diff <= 1e-6
    acc = lambda p,s:float(np.mean(p[s].argmax(1)==label_map[s]))
    row = dict(case=name, phase=phase,seed=seed,K=K,method=method,stress=stress,rounds=rounds,
               dev_accuracy=acc(up,'dev'),test_accuracy=acc(up,'test'),
               source_dev_accuracy=acc(sp,'dev'),source_test_accuracy=acc(sp,'test'),
               forgotten_accuracy=acc(up,'forgotten'),retain_accuracy=acc(up,'retain'),
               noop_forget_js=prob_js(sp['forgotten'],rp['forgotten']),
               replay_forget_js=prob_js(up['forgotten'],rp['forgotten']),
               replay_exact=equal,unaffected_equal=unaffected,authorized_participation=authorized,
               max_parameter_difference=pdiff,max_prediction_difference=prediction_diff,
               source_cost=src['cost'],reference_cost=rf['cost'],delete_cost=ur['cost'],
               checkpoint_bytes=2*K*engine.D*4,final_model_bytes=K*engine.D*4,
               case_seconds=time.perf_counter()-began)
    dump(folder/'result.json',row)
    dump(folder/'complete.json',dict(exact=equal,case=name))
    log(event='case_complete',case=name,accuracy=row['test_accuracy'],dev=row['dev_accuracy'],
        exact=equal,UL=ur['cost']['ul_reals'],total_re=ur['cost']['total_re'],seconds=row['case_seconds'])
    del engine,data,source,ref,replay
    torch.cuda.empty_cache()
    return row

def select_k(rows):
    globalrow = next(r for r in rows if r['K']==1)
    candidates=[]
    for k in [2,4,5]:
        cand=next(r for r in rows if r['K']==k and r['method']=='channel_norm')
        base=next(r for r in rows if r['K']==k and r['method']=='random_norm')
        eligible=(cand['dev_accuracy']>=base['dev_accuracy']-.02 and
                  cand['dev_accuracy']>=globalrow['dev_accuracy']-.05 and cand['replay_exact'])
        score=.5*cand['delete_cost']['local_calls']/globalrow['delete_cost']['local_calls']+.5*cand['delete_cost']['total_re']/globalrow['delete_cost']['total_re']
        candidates.append(dict(K=k,eligible=eligible,score=score,candidate_dev=cand['dev_accuracy'],
                               random_dev=base['dev_accuracy'],global_dev=globalrow['dev_accuracy']))
    ok=[r for r in candidates if r['eligible']]
    choice=min(ok,key=lambda r:(r['score'],r['K']))['K'] if ok else 2
    return dict(chosen_K=choice,eligible_selection=bool(ok),candidates=candidates,
                rule='Preregistered dev-only selection; test never used; K2 exploratory fallback if none eligible')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['preflight','main'],required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=ROOT/args.out
    out.mkdir(exist_ok=False)
    inventory=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],text=True)
    foreign=[s for s in inventory.splitlines()[1:] if s.strip() and 'ChatGPT.exe' not in s and not s.strip().startswith(str(os.getpid())+',')]
    if foreign:
        dump(out/'blocked.json',dict(foreign=foreign))
        raise RuntimeError('Existing GPU compute process; experiment not started')
    dump(out/'environment.json',dict(pid=os.getpid(),gpu_inventory=inventory,torch=torch.__version__,numpy=np.__version__,
         gpu=torch.cuda.get_device_name(0),code_sha256=sha(__file__),prereg_sha256=sha(ROOT/'PREREGISTRATION.md'),
         core_sha256=sha(BASE/'research_20260929_pdf_ota_unlearning/core.py')))
    began=time.perf_counter()
    rows=[]
    error=None
    with Power() as meter:
        try:
            radio_audit(out)
            if args.mode=='preflight':
                rows.append(one_case(out,'preflight',10600,2,'channel_norm',4))
            else:
                rows.append(one_case(out,'screen',10601,1,'random_norm',120))
                for k in [2,4,5]:
                    for method in ['random_fixed','channel_fixed','random_norm','channel_norm']:
                        rows.append(one_case(out,'screen',10601,k,method,120))
                selection=select_k(rows)
                dump(out/'selection.json',selection)
                log(event='selection',**selection)
                k=selection['chosen_K']
                for seed in [10611,10612,10613]:
                    for kk,method in [(1,'random_norm'),(k,'random_norm'),(k,'channel_norm')]:
                        rows.append(one_case(out,'confirm',seed,kk,method,120))
                for kk,method in [(1,'random_norm'),(k,'random_norm'),(k,'channel_norm')]:
                    rows.append(one_case(out,'stress',10621,kk,method,120,stress=True))
            dump(out/'results.json',rows)
        except BaseException:
            error=traceback.format_exc()
            dump(out/'failure.json',dict(error=error,completed_cases=len(rows)))
            dump(out/'cost_interrupted.json',dict(seconds=time.perf_counter()-began,**meter.report(),
                 error=error,completed_cases=len(rows),scope_note='Includes partial failed run wall time and available board samples'))
            raise
        finally:
            torch.cuda.synchronize()
    # Meter has stopped before energy integration, so the last sample is stable.
    dump(out/'cost_total.json',dict(seconds=time.perf_counter()-began,**meter.report(),error=error,
         completed_cases=len(rows),local_calls=sum(r[p]['local_calls'] for r in rows for p in ['source_cost','reference_cost','delete_cost'])))
    dump(out/'complete.json',dict(cases=len(rows),all_replay_exact=all(r['replay_exact'] for r in rows)))
    log(event='worker_complete',out=str(out),seconds=time.perf_counter()-began,cases=len(rows))

if __name__=='__main__':
    main()
