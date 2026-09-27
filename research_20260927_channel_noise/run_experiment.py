"""Frozen CUDA pilot: channel-noise matching and Gaussian-risk shrinkage.

Only NumPy/PyTorch are required. No private data or checkpoint is written.
The receiver sees aggregate sufficient statistics/corrections in this simulator;
the evaluator alone has the exact retained solution.
"""
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
OUT = ROOT / 'results'
DEV = 'cuda'
DT = torch.float64
SEEDS = [202609271, 202609272, 202609273]
SNRS = [10, 20, 30]
TAUS = [.001, .003, .01]
D, C, K, B, M = 65, 10, 9, 5, 130
MC, H_DRAWS = 128, 8
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    Path(p).parent.mkdir(exist_ok=True, parents=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def gen(seed):
    return torch.Generator(device=DEV).manual_seed(int(seed))


def randn(shape, seed):
    return torch.randn(shape, generator=gen(seed), device=DEV, dtype=DT)


def f32(x):
    return x.float().to(x.dtype)


def greedy(coeff, total):
    vals = coeff.detach().cpu().tolist()
    rs = [1] * len(vals)
    for _ in range(total-len(vals)):
        j = max(range(len(vals)), key=lambda k: vals[k]/(rs[k]*(rs[k]+1)))
        rs[j] += 1
    return torch.tensor(rs, device=DEV, dtype=DT)


def rate(snr):
    return .5*math.log2(1+10**(snr/10))


def cost(rs, snr, *, deletion=True):
    # 1 curvature phase + 5 correction phases, shared pilot across repetitions.
    analog = 2145*32 + M*float(rs.sum())
    metadata = 6*K*32 + 6*32 + B*32 + 256 + 32
    dl = D*D*32 + D*32 + D*C*32
    pilot = 6*K*8
    total = 1.125*(analog+pilot+(dl+metadata)/rate(snr))
    prep = 1.125*(2795+160+(D*C*32+32+640+64+128+32)/rate(snr))
    return dict(analog_real=analog, pilot_real=pilot, digital_bits=dl+metadata,
                control_metadata_bits=metadata, model_and_basis_dl_bits=dl,
                deletion_cached_real_uses=total,
                deletion_cold_real_uses=total+1.125*D*C*32/rate(snr),
                source_preparation_real_uses=prep, lifecycle_real_uses=total+prep,
                repeats=rs.int().tolist(), correction_real=M*float(rs.sum()),
                cp_factor=1.125, rate_bits_per_real=rate(snr))


def source_cost(snr):
    return cost(torch.ones(B, device=DEV), snr)['source_preparation_real_uses']


def bound_scale(payload):
    # payload: clients x blocks x within-block coordinates, weights included.
    return torch.maximum(payload.square().mean(2).sqrt(), payload.abs().amax(2)/2).amax(0)


def power(payload, scale, repeats):
    tx = payload/scale[None, :, None]
    avg = float(tx.square().mean(2).max())
    peak = float(tx.square().max())
    assert avg <= 1+1e-10 and peak <= 4+1e-10
    return dict(max_mean_power=avg, max_peak_power=peak,
                tx_symbol_energy_normalized=float((tx.square().sum((0, 2))*repeats).sum()))


def js(p, q):
    mid = (p+q)*.5
    return .5*((p*(p.clamp_min(1e-30).log()-mid.clamp_min(1e-30).log())).sum(-1)
               +(q*(q.clamp_min(1e-30).log()-mid.clamp_min(1e-30).log())).sum(-1))


def utility(ws, zt, yt, zf, refprob, js0):
    # Only first 4 payload-noise samples are used, fixed before execution.
    ws = f32(ws[:4])
    scores = torch.einsum('nd,sdc->snc', zt, ws)
    acc = (scores.argmax(-1)==yt[None]).double().mean(1)
    pf = torch.softmax(torch.einsum('nd,sdc->snc', zf, ws), -1)
    jj = js(pf, refprob[None]).mean(1)
    return dict(test_accuracy=float(acc.mean()), forget_js=float(jj.mean()),
                forget_normalized_js=float(jj.mean())/max(js0, 1e-30))


def load_data(data):
    def read(prefix):
        xx = np.fromfile(data/(prefix+'-images-idx3-ubyte'), dtype=np.uint8, offset=16).copy().reshape(-1, 784)
        yy = np.fromfile(data/(prefix+'-labels-idx1-ubyte'), dtype=np.uint8, offset=8).copy()
        return torch.as_tensor(xx, device=DEV, dtype=torch.float32)/255., torch.as_tensor(yy, device=DEV, dtype=torch.long)
    return (*read('train'), *read('t10k'))


def partition(y, seed):
    gg = gen(seed)
    pools = [[] for _ in range(10)]
    for label in range(10):
        ix = torch.where(y==label)[0]
        ix = ix[torch.randperm(len(ix), generator=gg, device=DEV)[:1000]]
        pools[label].append(ix[:500])
        pools[(label-1)%10].append(ix[500:])
    pools = [torch.cat(p) for p in pools]
    assert len(torch.unique(torch.cat(pools))) == 10000
    return pools


def curvature(hs, snr, seed, oracle):
    ix = torch.triu_indices(D, D, device=DEV)
    payload = hs[:, ix[0], ix[1]]/K
    scale = max(float(payload.square().mean(1).sqrt().max()), float(payload.abs().max())/2)
    noise = randn((2145,), seed)*scale/math.sqrt(10**(snr/10)*32)
    vec = payload.sum(0) + (0 if oracle else noise)
    hh = torch.zeros((D, D), device=DEV, dtype=DT)
    hh[ix[0], ix[1]] = vec
    hh[ix[1], ix[0]] = vec
    eig, q = torch.linalg.eigh(hh)
    eig = f32(eig.clamp_min(.01))
    # Public, deterministic postprocessing of the actually broadcast FP32 basis.
    q = torch.linalg.qr(f32(q)).Q
    return eig, q, dict(curvature_scale=scale, oracle=oracle,
                       broadcast_orthogonality_error=float((q.T@q-torch.eye(D, device=DEV)).abs().max()))


def gaussian_stats(mu, wr, variance, tau, gap):
    bias = float((mu-wr).square().sum())
    ratio = variance/tau**2
    covkl = float((M*(ratio-1-ratio.log())).sum())/2
    kl = bias/(2*tau*tau)+covkl
    matched = bool(torch.allclose(variance, torch.full_like(variance, tau*tau), rtol=1e-10, atol=1e-18))
    tv = math.erf(math.sqrt(bias)/(2*math.sqrt(2)*tau)) if matched else min(1., math.sqrt(kl/2))
    return dict(conditional_KL=kl, conditional_covariance_KL=covkl,
                conditional_TV_upper=tv, exact_conditional_TV=matched,
                mean_bias_squared=bias, mean_bias_ratio=bias/gap,
                theoretical_point_MSE_ratio=(bias+M*float(variance.sum()))/gap,
                matched_covariance=matched)


@torch.no_grad()
def math_checks():
    aa = randn((4, 12, 12), 9101)
    hs = aa@aa.transpose(-1, -2)+torch.eye(12, device=DEV)*.5
    cs = randn((4, 12, 3), 9102)
    w0 = randn((12, 3), 9103)
    hh, cc = hs.mean(0), cs.mean(0)
    correct = torch.linalg.solve(hh, (hs@w0-cs).mean(0))
    exact = float((w0-correct-torch.linalg.solve(hh, cc)).abs().max())
    assert exact < 1e-12
    payload = randn((9, 5, 130), 9104)*.08
    amin = bound_scale(payload)
    tau, snr = .01, 20
    vmin = amin.square()/100
    reps = (vmin/tau**2).ceil().clamp_min(1)
    scale = tau*(reps*100).sqrt()
    pp = power(payload, scale, reps)
    assert bool(torch.all(amin <= scale*(1+1e-14)))
    assert bool(torch.all((reps==1) | (vmin/(reps-1)>tau**2)))
    # Explicit channel superposition, before adding receiver AWGN.
    h = torch.linspace(.65, 1.3, 9, device=DEV, dtype=DT)
    ah = torch.maximum((payload.square().mean(2).sqrt()/h[:, None]),
                       payload.abs().amax(2)/(2*h[:, None])).amax(0)
    rx = (h[:, None, None]*(payload/(ah[None, :, None]*h[:, None, None]))).sum(0)*ah[:, None]
    mac = float((rx-payload.sum(0)).abs().max())
    assert mac < 1e-14
    # Verify physical repetition averaging, not just a Gaussian formula.
    n = randn((20000, int(reps.max()), 5), 9105)/10
    averaged = torch.stack([n[:, :int(reps[j]), j].mean(1)*scale[j] for j in range(5)], 1)
    cov = torch.cov(averaged.T)/tau**2
    err = float((cov-torch.eye(5, device=DEV)).abs().max())
    assert err < .05
    # Zero-mean noise adds its variance to deterministic target squared error.
    bias = randn((30,), 9106)*.1
    zz = randn((50000, 30), 9107)*.03
    mse = float((bias+zz).square().sum(1).mean())
    theory = float(bias.square().sum())+30*.03**2
    assert abs(mse-theory)/theory < .01
    # JS theorem only for unbiased Gaussian observations of the desired mean.
    theta = randn((130,), 9110)*.02
    obs = theta+randn((30000, 130), 9111)*.03
    est = obs*(1-128*.03**2/obs.square().sum(1).clamp_min(1e-30)).clamp_min(0)[:, None]
    rawrisk = float((obs-theta).square().sum(1).mean())
    jsrisk = float((est-theta).square().sum(1).mean())
    assert jsrisk < rawrisk
    # Direct log densities match analytic KL for equal covariance.
    shift = randn((10,), 9112)*.01
    zz = randn((100000, 10), 9113)*.03
    sample = shift+zz
    empirical_kl = float(((sample.square()-zz.square()).sum(1)/(2*.03**2)).mean())
    analytic_kl = float(shift.square().sum())/(2*.03**2)
    assert abs(empirical_kl-analytic_kl) < .015
    result = dict(passed=True, quadratic_identity_max_error=exact, MAC_max_error=mac,
                  covariance_relative_max_error=err, integer_repetitions_minimal=True,
                  power=pp, noise_only_MSE=mse, noise_only_theory=theory,
                  JS_unbiased_Gaussian_raw_MSE=rawrisk, JS_unbiased_Gaussian_shrunk_MSE=jsrisk,
                  empirical_KL=empirical_kl, analytic_KL=analytic_kl,
                  code_sha256=sha(__file__), protocol_sha256=sha(ROOT/'PROTOCOL_KO.md'))
    write(OUT/'math_checks.json', result)
    print(json.dumps(dict(stage='math', **result)), flush=True)


@torch.no_grad()
def run(data):
    tstart = time.perf_counter()
    x, y, xt, yt = load_data(data)
    p = torch.randn((784, 64), generator=gen(20260922), device=DEV)/math.sqrt(784)
    z = torch.cat([F.relu((x-.5)@p), torch.ones(len(x), 1, device=DEV)], 1).double()
    zt = torch.cat([F.relu((xt-.5)@p), torch.ones(len(xt), 1, device=DEV)], 1).double()
    yy = F.one_hot(y, 10).double()
    eye = torch.eye(D, device=DEV, dtype=DT)
    data_manifest = {f.name: sha(f) for f in sorted(data.glob('*idx*')) if f.is_file() and not f.name.endswith('.gz')}
    write(OUT/'environment.json', dict(torch=torch.__version__, numpy=np.__version__,
          gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, one_worker=True,
          seeds=SEEDS, snr_db=SNRS, taus=TAUS, H_draws=H_DRAWS, payload_draws=MC,
          utility_draws=4, code_sha256=sha(__file__), protocol_sha256=sha(ROOT/'PROTOCOL_KO.md'),
          data_sha256=data_manifest))
    for si, seed in enumerate(SEEDS):
        tick = time.perf_counter()
        pools = partition(y, seed)
        hs = torch.stack([z[ix].T@z[ix]/len(ix)+.01*eye for ix in pools])
        cs = torch.stack([z[ix].T@yy[ix]/len(ix) for ix in pools])
        wclean = torch.linalg.solve(hs.mean(0), cs.mean(0))
        wr = torch.linalg.solve(hs[1:].mean(0), cs[1:].mean(0))
        zf = z[pools[0]]
        pref = torch.softmax(zf@wr, -1)
        source_random = randn((D, C), seed+10)
        rows, points, references = [], [], []
        for ti, tau in enumerate(TAUS):
            w0 = f32(wclean+tau*source_random)
            znoise = randn((MC, D, C), seed+30+ti)
            p0 = torch.softmax(zf@w0, -1)
            js0 = float(js(p0, pref).mean())
            rutil = utility(wr[None]+tau*znoise, zt, yt, zf, pref, js0)
            references.append(dict(tau=tau, source_accuracy=utility(w0[None], zt, yt, zf, pref, js0)['test_accuracy'],
                                   randomized_retrain=rutil, clean_retrain_accuracy=float(((zt@wr).argmax(1)==yt).double().mean()),
                                   source_gap_squared=float((w0-wr).square().sum())))
        for snr in SNRS:
            s = 10**(snr/10)
            for oracle in [False, True]:
                # The oracle is a math diagnostic; costs are hypothetical only.
                for hn in range(H_DRAWS if not oracle else 1):
                    ns = seed*10000+snr*100+hn*10
                    eig, q, hdiag = curvature(hs[1:], snr, ns, oracle)
                    noise = randn((MC, B, M), ns+1)
                    noise2 = randn((MC, B, M), ns+2)
                    for ti, tau in enumerate(TAUS):
                        w0 = f32(wclean+tau*source_random)
                        gap = float((w0-wr).square().sum())
                        b = hs[1:]@w0-cs[1:]
                        payload = ((q.T@b)/eig[None, :, None]/K).reshape(K, B, M)
                        center = payload.sum(0)
                        amin = bound_scale(payload)
                        vmin = amin.square()/s
                        rmin = (vmin/tau**2).ceil().clamp_min(1)
                        r40 = greedy(M*vmin, 40)
                        mu = w0-q@center.reshape(D, C)
                        js0 = float(js(torch.softmax(zf@w0, -1), pref).mean())
                        for method in ['NM-Air', 'Uniform-match', 'Aware-fixed40', 'DS-fixed40', 'Naive-add40', 'Noise-only']:
                            if method=='NM-Air':
                                rs = rmin
                                scale = tau*(rs*s).sqrt()
                                var = torch.full_like(rs, tau*tau)
                                rec = center+tau*noise
                            elif method=='Uniform-match':
                                rs = torch.ones_like(rmin)*rmin.max()
                                scale = tau*(rs*s).sqrt()
                                var = torch.full_like(rs, tau*tau)
                                rec = center+tau*noise
                            elif method=='Aware-fixed40':
                                rs = torch.maximum(rmin, r40)
                                scale = amin
                                physical = vmin/rs
                                var = torch.full_like(rs, tau*tau)
                                rec = center+physical.sqrt()[None, :, None]*noise+(tau*tau-physical).clamp_min(0).sqrt()[None, :, None]*noise2
                            elif method in ['DS-fixed40', 'Naive-add40']:
                                rs, scale = r40, amin
                                var = vmin/rs
                                rec = center+var.sqrt()[None, :, None]*noise
                                if method=='Naive-add40':
                                    var = var+tau*tau
                                    rec = rec+tau*noise2
                            else:
                                rs = torch.ones_like(rmin)
                                scale = amin
                                var = torch.full_like(rs, tau*tau)
                                rec = tau*noise
                            ws = w0[None]-q@rec.reshape(MC, D, C)
                            cmu = w0 if method=='Noise-only' else mu
                            gs = gaussian_stats(cmu, wr, var, tau, gap)
                            row = dict(seed=seed, snr_db=snr, tau=tau, curvature='oracle' if oracle else 'OTA32',
                                H_noise=hn, method=method, **gs,
                                empirical_point_MSE_ratio=float((ws-wr).square().sum((1, 2)).mean())/gap,
                                **utility(ws, zt, yt, zf, pref, js0))
                            if method!='Noise-only':
                                row.update(cost=cost(rs, snr), power=power(payload, scale, rs),
                                           physical_variance=(scale.square()/s/rs).tolist(),
                                           output_variance=var.tolist(), hessian=hdiag)
                            else:
                                # Software-only diagnostic, no deletion communication.
                                row.update(cost=None, diagnostic_only=True)
                            rows.append(row)
                    # Deterministic reference branch: block James--Stein shrinkage.
                    w0 = f32(wclean)
                    gap = float((w0-wr).square().sum())
                    b = hs[1:]@w0-cs[1:]
                    payload = ((q.T@b)/eig[None, :, None]/K).reshape(K, B, M)
                    center = payload.sum(0)
                    vmin = bound_scale(payload).square()/s
                    js0 = float(js(torch.softmax(zf@w0, -1), pref).mean())
                    for budget in [5, 10, 20, 40]:
                        rs = greedy(M*vmin, budget)
                        variance = vmin/rs
                        rec = center+variance.sqrt()[None, :, None]*noise
                        shrink = (1-(M-2)*variance[None]/rec.square().sum(-1).clamp_min(1e-30)).clamp_min(0)
                        for method, rr in [('DS-raw', rec), ('JS-Air', rec*shrink[:, :, None])]:
                            ws = w0[None]-q@rr.reshape(MC, D, C)
                            points.append(dict(seed=seed, snr_db=snr, curvature='oracle' if oracle else 'OTA32',
                                H_noise=hn, budget=budget, method=method,
                                empirical_point_MSE_ratio=float((ws-wr).square().sum((1, 2)).mean())/gap,
                                mean_shrink_factor=float(shrink.mean()) if method=='JS-Air' else 1.,
                                **utility(ws, zt, yt, zf, pref, js0), cost=cost(rs, snr)))
                print(json.dumps(dict(stage='data', seed=seed, snr=snr, oracle=oracle,
                                      gaussian_rows=len(rows), point_rows=len(points))), flush=True)
        torch.cuda.synchronize()
        write(OUT/f'seed{seed}.json', dict(complete=True, seed=seed, rows=rows, point_rows=points,
              references=references, wall_seconds=time.perf_counter()-tick,
              source_rule='A_tau(D)=exact ridge(D)+tau*Gaussian then FP32 broadcast',
              source_preparation='ideal sufficient-statistic construction; same assumption as earlier DS pilot',
              code_sha256=sha(__file__), protocol_sha256=sha(ROOT/'PROTOCOL_KO.md')))
    write(OUT/'completion.json', dict(complete=True, wall_seconds=time.perf_counter()-tstart,
          max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), new_model_data_seeds=SEEDS,
          real_RF_measurement=False, worker_count=1))


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['math', 'run', 'all'], default='all')
    ap.add_argument('--data', type=Path, required=False)
    args = ap.parse_args()
    assert torch.cuda.is_available(), 'A CUDA GPU is required for this experiment.'
    if args.stage in ['math', 'all']:
        math_checks()
    if args.stage in ['run', 'all']:
        checks = json.loads((OUT/'math_checks.json').read_text(encoding='utf-8'))
        assert checks['passed'] and checks['code_sha256']==sha(__file__)
        assert checks['protocol_sha256']==sha(ROOT/'PROTOCOL_KO.md')
        assert args.data is not None and args.data.is_dir()
        run(args.data)
