"""DS-Air bounded revision; NumPy/PyTorch only, one CUDA process.

No individual data, features or model checkpoints are saved. Exact references
are evaluator-only; scheduling sees only curvature and declared power metadata.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
DEV, DT = 'cuda', torch.float64
SEEDS = [202609281, 202609282, 202609283]
DIMS, SNRS = [32, 64, 128], [10, 20, 30]
CHANNELS = ['unit', 'rayleigh', 'rayleigh_csi5']
K, C, WIDTH, MC, H_DRAWS = 9, 10, 8, 32, 4
LAM, CP = .01, 1.125
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def rng(seed):
    return torch.Generator(device=DEV).manual_seed(int(seed))


def normal(shape, seed):
    return torch.randn(shape, generator=rng(seed), device=DEV, dtype=DT)


def fp32(x):
    return x.float().double()


def upward32(x):
    z = x.float()
    z = torch.where(z.double() < x, torch.nextafter(z, torch.full_like(z, math.inf)), z)
    return z.double()


def rate(snr):
    return .5*math.log2(1+10**(snr/10))


def allocation(coeff, total):
    # Exact discrete optimum for separable c/r with equal-size blocks.
    vals = coeff.detach().cpu().tolist()
    reps = [1]*len(vals)
    if total < len(vals):
        raise ValueError('budget cannot pay for one transmission per block')
    for _ in range(total-len(vals)):
        j = max(range(len(vals)), key=lambda q: vals[q]/(reps[q]*(reps[q]+1)))
        reps[j] += 1
    return torch.tensor(reps, device=DEV, dtype=DT)


def ledger(d, r, snr, hr, kind, repeats):
    b = r//WIDTH
    if kind == 'LocalNewton':
        hcoords, hmeta, hp = 0, 0, 0
    else:
        hcoords = r if kind == 'Diagonal' else r*(r+1)//2
        hmeta, hp = K*32+64, K*8
    choices = 2 if kind == 'Switch' else 1
    # Each bound is per-client, includes data weight and estimated channel gain.
    meta = 256+hmeta+b*(choices*K*32+64)
    basis = 0 if kind in ['LocalNewton', 'Diagonal', 'StatsRetrain'] else r*r*32
    eig = r*32 if kind in ['V1', 'Switch', 'SubspaceV1', 'Diagonal'] else 0
    mode = b if kind == 'Switch' else 0
    useed = 64 if r < d else 0
    final = d*C*32
    digital = meta+basis+eig+mode+useed+final
    analog_h = hcoords*hr
    correction = WIDTH*C*sum(repeats)
    pilots = hp+b*K*8
    total = CP*(analog_h+correction+pilots+digital/rate(snr))
    # Ideal-statistic source training accounting, one H and one C aggregate.
    prep_analog = d*(d+1)//2+d*C
    prep_bits = 2*(10*32+64)+final+256
    prep = CP*(prep_analog+160+prep_bits/rate(snr))
    return dict(total=total, analog_H=analog_h, analog_correction=correction,
                pilot_real=pilots, digital_bits=digital, metadata_bits=meta,
                basis_bits=basis, eigen_bits=eig, selection_bits=mode,
                public_subspace_seed_bits=useed, final_model_bits=final,
                cold_total=total+CP*final/rate(snr), source_preparation=prep,
                lifecycle=prep+total, cp=CP, digital_rate=rate(snr),
                repeats=[int(x) for x in repeats], H_repeats=hr)


def repeat_budget(d, r, snr, hr, kind, ceiling):
    fixed = ledger(d, r, snr, hr, kind, [])['total']
    return math.floor((ceiling-fixed)/(CP*WIDTH*C)+1e-8)


def channel(kind, seed):
    if kind == 'unit':
        h = torch.ones(K, device=DEV, dtype=DT)
    else:
        h = normal((K, 2), seed).square().sum(1).div(2).sqrt()
    estimated = h*(normal((K,), seed+1)*.05).exp() if kind == 'rayleigh_csi5' else h.clone()
    return h, estimated


def prepare(payload, h, estimated):
    # payload: clients x blocks x coordinates. Already includes p_i.
    bounds = torch.maximum(payload.square().mean(-1).sqrt(), payload.abs().amax(-1)/2)
    declared = upward32(bounds/estimated[:, None])
    scale = declared.amax(0).clamp_min(1e-30)
    tx = payload/(scale[None, :, None]*estimated[:, None, None])
    rx = (h[:, None, None]*tx).sum(0)*scale[:, None]
    power = dict(mean=float(tx.square().mean(-1).max()), peak=float(tx.square().max()))
    assert power['mean'] <= 1+1e-10 and power['peak'] <= 4+1e-10
    return rx, scale, declared, power, tx


def basis(hs, h, he, snr, hr, seed, diagonal=False):
    r = hs.shape[-1]
    ix = torch.triu_indices(r, r, device=DEV)
    data = hs.diagonal(dim1=-2, dim2=-1) if diagonal else hs[:, ix[0], ix[1]]
    mean, sc, declared, pp, _ = prepare((data/K)[:, None, :], h, he)
    got = mean[0]+normal((data.shape[1],), seed)*sc[0]/math.sqrt(10**(snr/10)*hr)
    if diagonal:
        eig = fp32(got.clamp_min(LAM))
        q = torch.eye(r, device=DEV, dtype=DT)
    else:
        hh = torch.zeros((r, r), device=DEV, dtype=DT)
        hh[ix[0], ix[1]] = got
        hh[ix[1], ix[0]] = got
        eig, q = torch.linalg.eigh(hh)
        eig = fp32(eig.clamp_min(LAM))
        q = torch.linalg.qr(fp32(q)).Q
    return eig, q, dict(scale=float(sc[0]), min_eigen=float(eig.min()), max_eigen=float(eig.max()),
                        power=pp, min_actual_gain=float(h.min()), max_gain_ratio_error=float((h/he-1).abs().max()))


def dataset(path):
    def part(prefix):
        x = np.fromfile(path/(prefix+'-images-idx3-ubyte'), dtype=np.uint8, offset=16).copy().reshape(-1, 784)
        y = np.fromfile(path/(prefix+'-labels-idx1-ubyte'), dtype=np.uint8, offset=8).copy()
        return torch.tensor(x, device=DEV, dtype=torch.float32)/255., torch.tensor(y, device=DEV, dtype=torch.long)
    return (*part('train'), *part('t10k'))


def partition(y, seed):
    g = rng(seed)
    pools = [[] for _ in range(10)]
    for label in range(10):
        ix = torch.where(y == label)[0]
        ix = ix[torch.randperm(len(ix), device=DEV, generator=g)[:1000]]
        pools[label].append(ix[:500])
        pools[(label-1)%10].append(ix[500:])
    pools = [torch.cat(p) for p in pools]
    assert len(torch.unique(torch.cat(pools))) == 10000
    return pools


def js(p, q):
    m = (p+q)/2
    return .5*((p*(p.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum(-1)
               +(q*(q.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum(-1))


def utility(ws, zt, yt, zf, refp, js0):
    ws = fp32(ws[:2])
    acc = (torch.einsum('nd,sdc->snc', zt, ws).argmax(-1) == yt[None]).double().mean()
    prob = torch.softmax(torch.einsum('nd,sdc->snc', zf, ws), -1)
    divergence = js(prob, refp[None]).mean()
    return dict(test_accuracy=float(acc), forget_JS=float(divergence), forget_JS_ratio=float(divergence)/js0)


def evaluate_method(kind, d, r, eig, q, u, local_h, local_b, local_c, w0, wr,
                    ga, h, he, snr, hr, seed, ceiling, eval_data, nominal=False):
    bcount = r//WIDTH
    if kind == 'LocalNewton':
        payload = torch.linalg.solve(local_h, local_b)/K
        transform = torch.eye(d, device=DEV, dtype=DT)
        exact_corr = payload.sum(0)
        mean, scale, _, pp, tx = prepare(payload.reshape(K, bcount, WIDTH*C), h, he)
        coeff = scale.square()*WIDTH*C/10**(snr/10)
        selected = None
    elif kind == 'StatsRetrain':
        payload = local_c/K
        transform = (q/eig)@q.T
        exact_corr = transform@payload.sum(0)
        mean, scale, _, pp, tx = prepare(payload.reshape(K, bcount, WIDTH*C), h, he)
        weight = transform.square().sum(0).reshape(bcount, WIDTH).sum(1)*C
        coeff = scale.square()*weight/10**(snr/10)
        selected = None
    else:
        bb = torch.einsum('ij,kjc->kic', q.T, local_b)/K
        raw = bb.reshape(K, bcount, WIDTH*C)
        pre = (bb/eig[None, :, None]).reshape_as(raw)
        mr, sr, dr, pr, tr = prepare(raw, h, he)
        mp, sp, dp, pp0, tp = prepare(pre, h, he)
        c_raw = sr.square()*eig.reshape(bcount, WIDTH).pow(-2).sum(1)*C/10**(snr/10)
        c_pre = sp.square()*WIDTH*C/10**(snr/10)
        if kind == 'Original':
            selected = torch.zeros(bcount, device=DEV, dtype=torch.bool)
        elif kind == 'Switch':
            selected = c_pre < c_raw
        else:
            selected = torch.ones(bcount, device=DEV, dtype=torch.bool)
        mean = torch.where(selected[:, None], mp, mr)
        scale = torch.where(selected, sp, sr)
        coeff = torch.where(selected, c_pre, c_raw)
        tx = torch.where(selected[None, :, None], tp, tr)
        pp = dict(mean=float(tx.square().mean(-1).max()), peak=float(tx.square().max()))
        post = torch.where(selected.repeat_interleave(WIDTH), torch.ones_like(eig), 1/eig)
        transform = (u@q)*post
        exact_corr = u@q@((q.T@local_b.mean(0))/eig[:, None])
    total_reps = 8*bcount if nominal else repeat_budget(d, r, snr, hr, kind, ceiling)
    if total_reps < bcount:
        return dict(method=kind, status='infeasible', budget=ceiling)
    reps = allocation(coeff, total_reps)
    cost = ledger(d, r, snr, hr, kind, reps.cpu().tolist())
    assert cost['total'] <= ceiling+1e-7
    v = (scale.square()/10**(snr/10)/reps).repeat_interleave(WIDTH)
    rawmean = mean.reshape(r, C)
    mean_w = transform@rawmean if kind == 'StatsRetrain' else w0-transform@rawmean
    ideal_mean = exact_corr if kind == 'StatsRetrain' else w0-exact_corr
    standard = normal((MC, r, C), seed)
    noise_w = torch.einsum('dr,src->sdc', transform, standard*v.sqrt()[None, :, None])
    ws = mean_w[None]+noise_w
    gap = float((w0-wr).square().sum())
    bias = float((mean_w-wr).square().sum())
    variance = float((transform.square().sum(0)*v).sum())*C
    empirical = (ws-wr[None]).square().sum((1, 2))
    met = dict(MSE_ratio=float(empirical.mean())/gap, theoretical_MSE_ratio=(bias+variance)/gap,
               bias_ratio=bias/gap, variance_ratio=variance/gap,
               CSI_mean_shift_norm=float((mean_w-ideal_mean).norm()),
               MonteCarlo_SE_ratio=float(empirical.std())/math.sqrt(MC)/gap,
               deployed_MSE_ratio=float((fp32(ws)-wr[None]).square().sum((1, 2)).mean())/gap)
    met.update(utility(ws, *eval_data))
    # Evaluator-only attack, using only objects available to the declared server.
    attack = None
    if kind in ['Original', 'V1', 'Switch', 'SubspaceV1', 'Diagonal']:
        delta = w0[None]-ws
        invsmall = (q*eig)@q.T
        projected = -K*torch.einsum('ri,sic->src', invsmall@u.T, delta)
        truth_projected = u.T@ga
        lifted = torch.einsum('dr,src->sdc', u, projected)
        attack = dict(full_gradient_relative_MSE=float((lifted-ga[None]).square().sum((1, 2)).mean()/ga.square().sum()),
                      projected_gradient_relative_MSE=float((projected-truth_projected[None]).square().sum((1, 2)).mean()/truth_projected.square().sum()),
                      unobserved_linear_dimensions=(d-r)*C,
                      interpretation='specific stationarity attack; failure is not a privacy certificate')
    return dict(method=kind+('-nominal' if nominal else ''), status='ok', rank=r,
                metrics=met, cost=cost, budget=ceiling, power=pp,
                normalized_tx_energy=float((tx.square().sum((0, 2))*reps).sum()),
                chosen_pre_blocks=None if selected is None else selected.int().tolist(), attack=attack)


@torch.no_grad()
def math_checks():
    eye = torch.eye(12, device=DEV, dtype=DT)
    hh = normal((10, 12, 12), 1201)
    hh = hh@hh.transpose(-1, -2)+eye
    cc = normal((10, 12, 3), 1202)
    w0 = torch.linalg.solve(hh.mean(0), cc.mean(0))
    hr, cr = hh[1:].mean(0), cc[1:].mean(0)
    bb = hh[1:]@w0-cc[1:]
    delta = torch.linalg.solve(hr, bb.mean(0))
    wr = torch.linalg.solve(hr, cr)
    identity = float((w0-delta-wr).abs().max())
    ga = hh[0]@w0-cc[0]
    disclosure = float((-9*hr@delta-ga).abs().max())
    assert identity < 1e-12 and disclosure < 1e-12
    # Stronger indistinguishable-transcript example includes source and metadata.
    u = torch.linalg.qr(normal((12, 8), 1203)).Q
    null = (eye-u@u.T)@normal((12, 3), 1204)
    cc2 = cc.clone()
    cc2[0] += null
    cc2[1:] -= null/9
    w02 = torch.linalg.solve(hh.mean(0), cc2.mean(0))
    b2 = hh[1:]@w02-cc2[1:]
    projected_change = float((u.T@bb-u.T@b2).abs().max())
    _, d1, m1, _, _ = prepare((u.T@bb/9).reshape(9, 1, -1), torch.ones(9, device=DEV), torch.ones(9, device=DEV))
    _, d2, m2, _, _ = prepare((u.T@b2/9).reshape(9, 1, -1), torch.ones(9, device=DEV), torch.ones(9, device=DEV))
    assert projected_change < 1e-12 and torch.equal(m1, m2)
    # Different actual coefficients and unrounded physical MAC.
    payload = normal((9, 3, 24), 1210)*.1
    h, he = channel('rayleigh', 1211)
    mean, scale, _, pp, tx = prepare(payload, h, he)
    mac = float((mean-payload.sum(0)).abs().max())
    assert mac < 1e-12
    r = torch.tensor([2, 5, 9], device=DEV)
    nn = normal((20000, 9, 3), 1212)/10
    avg = torch.stack([nn[:, :int(r[j]), j].mean(1)*scale[j] for j in range(3)], 1)
    expected = scale.square()/100/r
    err = float((avg.var(0)/expected-1).abs().max())
    assert err < .04
    coeff = torch.tensor([1., 3., 7.], device=DEV)
    rr = allocation(coeff, 9)
    exhaustive = min(sum(float(coeff[j])/x[j] for j in range(3))
                     for x in itertools.product(range(1, 8), repeat=3) if sum(x) == 9)
    assert abs(float((coeff/rr).sum())-exhaustive) < 1e-10
    # Scalar pre/post scaling must cancel under a common power budget.
    scalar_p, scalar = 17., normal((9, 1, 16), 1220)
    _, a, _, _, _ = prepare(scalar, h, he)
    _, ap, _, _, _ = prepare(scalar*scalar_p, h, he)
    cancellation = abs(float((ap/a/scalar_p)**2)-1)
    assert cancellation < 3e-7
    # Power-limited preconditioning can help or hurt (mean-power-only example).
    p = torch.diag(torch.tensor([10., 1.], device=DEV, dtype=DT))
    def risk(v):
        a2 = float(v.square().mean())
        pre2 = float((p@v).square().mean())
        return a2*float(p.square().sum()), 2*pre2
    good = risk(torch.tensor([0., 1.], device=DEV, dtype=DT))
    bad = risk(torch.tensor([1., 0.], device=DEV, dtype=DT))
    assert good[1] < good[0] and bad[1] > bad[0]
    result = dict(passed=True, quadratic_identity_error=identity, full_gradient_reconstruction_error=disclosure,
                  subspace_example=dict(source_change=float((w02-w0).abs().max()),
                                        projected_payload_change=projected_change, metadata_identical=bool(torch.equal(m1, m2)),
                                        target_full_gradient_change=float((cc2[0]-cc[0]).norm()),
                                        guarantee='nonidentifiability example only; not DP'),
                  physical_MAC_error=mac, repetition_variance_relative_error=err, power=pp,
                  greedy_optimal=True, scalar_cancellation_error=cancellation,
                  pre_better_risks=good, pre_worse_risks=bad,
                  code_sha256=sha(__file__), protocol_sha256=sha(ROOT/'PROTOCOL_KO.md'))
    write(OUT/'math_checks.json', result)
    print(json.dumps(result), flush=True)


@torch.no_grad()
def run(paths):
    check = json.loads((OUT/'math_checks.json').read_text())
    assert check['passed'] and check['code_sha256'] == sha(__file__)
    start = time.perf_counter()
    rows_count = 0
    manifests = {}
    torch.cuda.reset_peak_memory_stats()
    for ds_index, (name, path) in enumerate(paths.items()):
        x, y, xt, yt = dataset(path)
        manifests[name] = {p.name: sha(p) for p in sorted(path.glob('*ubyte'))}
        for d in DIMS:
            proj = torch.randn((784, d-1), generator=rng(202609280+d), device=DEV)/math.sqrt(784)
            z = torch.cat([F.relu((x-.5)@proj), torch.ones(len(x), 1, device=DEV)], 1).double()
            zt = torch.cat([F.relu((xt-.5)@proj), torch.ones(len(xt), 1, device=DEV)], 1).double()
            yy = F.one_hot(y, 10).double()
            eye = torch.eye(d, device=DEV, dtype=DT)
            rank = 3*d//4
            u = torch.linalg.qr(normal((d, rank), 2026092800+d)).Q
            for seed in SEEDS:
                rows = []
                pools = partition(y, seed)
                hs = torch.stack([z[p].T@z[p]/len(p)+LAM*eye for p in pools])
                cs = torch.stack([z[p].T@yy[p]/len(p) for p in pools])
                w0 = fp32(torch.linalg.solve(hs.mean(0), cs.mean(0)))
                wr = torch.linalg.solve(hs[1:].mean(0), cs[1:].mean(0))
                bs = hs[1:]@w0-cs[1:]
                ga = hs[0]@w0-cs[0]
                subh, subb = u.T@hs[1:]@u, u.T@bs
                zf = z[pools[0]]
                refp = torch.softmax(zf@wr, -1)
                js0 = max(float(js(torch.softmax(zf@w0, -1), refp).mean()), 1e-30)
                eval_data = (zt, yt, zf, refp, js0)
                references = dict(source_accuracy=float(((zt@w0).argmax(-1) == yt).double().mean()),
                                  retained_accuracy=float(((zt@wr).argmax(-1) == yt).double().mean()),
                                  initial_parameter_gap=float((w0-wr).square().sum()), initial_forget_JS=js0,
                                  source_stationarity_norm=float((hs@w0-cs).mean(0).norm()))
                conditions = [('main', snr, ch, 32) for snr in SNRS for ch in CHANNELS]
                if d == 64:
                    conditions += [('curvature', 20, 'rayleigh', hr) for hr in [8, 128]]
                for section, snr, ch, hr in conditions:
                    ceiling = ledger(d, d, snr, hr, 'Original', [8]*(d//WIDTH))['total']
                    for draw in range(H_DRAWS):
                        ns = seed*100000+ds_index*10000+d*10+snr*7+draw*200+CHANNELS.index(ch)*2000
                        h, he = channel(ch, ns)
                        eig, q, hdiag = basis(hs[1:], h, he, snr, hr, ns+2)
                        se, sq, sdiag = basis(subh, h, he, snr, hr, ns+3)
                        de, dq, ddiag = basis(hs[1:], h, he, snr, hr, ns+4, True)
                        for method in ['Original', 'V1', 'Switch', 'LocalNewton', 'Diagonal', 'StatsRetrain', 'SubspaceV1', 'SubspaceV1-nominal']:
                            small = method.startswith('Subspace')
                            diagonal = method == 'Diagonal'
                            rr = rank if small else d
                            ee, qq = (se, sq) if small else ((de, dq) if diagonal else (eig, q))
                            row = evaluate_method(method.replace('-nominal', ''), d, rr, ee, qq,
                                      u if small else eye, subh if small else hs[1:], subb if small else bs,
                                      cs[1:], w0, wr, ga, h, he, snr, hr, ns+20, ceiling, eval_data,
                                      nominal=method.endswith('-nominal'))
                            row.update(dataset=name, D=d, seed=seed, SNR=snr, channel=ch, H_draw=draw,
                                       section=section, H_repeats=hr, curvature=(sdiag if small else ddiag if diagonal else hdiag)
                                       if method != 'LocalNewton' else None)
                            rows.append(row)
                    print(json.dumps(dict(dataset=name, D=d, seed=seed, SNR=snr, channel=ch,
                                          H_repeats=hr, rows=len(rows))), flush=True)
                rows_count += len(rows)
                write(OUT/f'{name}_D{d}_seed{seed}.json', dict(references=references, rows=rows, complete=True))
                del hs, cs, bs, subh, subb
            del z, zt, yy
        del x, y, xt, yt
    torch.cuda.synchronize()
    complete = dict(complete=True, rows=rows_count, GPU_seconds=time.perf_counter()-start,
                    peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
                    torch_version=torch.__version__, CUDA=torch.version.cuda, GPU=torch.cuda.get_device_name(),
                    seeds=SEEDS, MC=MC, H_draws=H_DRAWS, data=manifests,
                    code_sha256=sha(__file__), protocol_sha256=sha(ROOT/'PROTOCOL_KO.md'))
    write(OUT/'completion.json', complete)
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['math', 'run'], required=True)
    ap.add_argument('--fashion', type=Path)
    ap.add_argument('--mnist', type=Path)
    args = ap.parse_args()
    assert torch.cuda.is_available(), 'CUDA required for this frozen study'
    if args.stage == 'math':
        math_checks()
    else:
        assert args.fashion and args.mnist
        run({'FashionMNIST': args.fashion, 'MNIST': args.mnist})
