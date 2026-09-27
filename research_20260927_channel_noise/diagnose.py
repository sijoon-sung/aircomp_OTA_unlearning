"""Post-run diagnosis, fixed main setting only; never tunes the main experiment."""
from pathlib import Path
import argparse
import math
import time
import torch
import torch.nn.functional as F
import run_experiment as e


def auc(positive, negative):
    # P(score_U > score_reference)+.5*P(tie); no trainable discriminator.
    neg = negative.sort().values
    lo = torch.searchsorted(neg, positive, right=False)
    hi = torch.searchsorted(neg, positive, right=True)
    return float(((lo+hi).double()*.5/len(neg)).mean())


@torch.no_grad()
def run(data):
    start = time.perf_counter()
    x, y, _, _ = e.load_data(data)
    proj = torch.randn((784, 64), generator=e.gen(20260922), device=e.DEV)/math.sqrt(784)
    z = torch.cat([F.relu((x-.5)@proj), torch.ones(len(x), 1, device=e.DEV)], 1).double()
    yy = F.one_hot(y, 10).double()
    eye = torch.eye(65, device=e.DEV, dtype=e.DT)
    rows = []
    tau = .003
    for seed in e.SEEDS:
        pools = e.partition(y, seed)
        hs = torch.stack([z[p].T@z[p]/len(p)+.01*eye for p in pools])
        cs = torch.stack([z[p].T@yy[p]/len(p) for p in pools])
        w0 = e.f32(torch.linalg.solve(hs.mean(0), cs.mean(0))+tau*e.randn((65, 10), seed+10))
        wr = torch.linalg.solve(hs[1:].mean(0), cs[1:].mean(0))
        scores = {key: [] for key in ['OTA32', 'oracle', 'noise_only']}
        for hn in range(128):
            base = 700000000+seed*1000+hn*5
            for oracle in [False, True]:
                eig, q, _ = e.curvature(hs[1:], 20, base, oracle)
                center = (((q.T@(hs[1:]@w0-cs[1:]))/eig[None, :, None])/9).sum(0)
                w = w0[None]-q@(center[None]+tau*e.randn((16, 65, 10), base+1))
                score = (e.f32(w)-wr).square().sum((1, 2))
                scores['oracle' if oracle else 'OTA32'].append(score)
            w = w0[None]+tau*e.randn((16, 65, 10), base+2)
            scores['noise_only'].append((e.f32(w)-wr).square().sum((1, 2)))
        ref = e.f32(wr[None]+tau*e.randn((2048, 65, 10), seed+88700))
        refscore = (ref-wr).square().sum((1, 2))
        for key, vals in scores.items():
            vals = torch.cat(vals)
            rows.append(dict(seed=seed, sampler=key, sampler_draws=2048, reference_draws=2048,
                independent_H_draws=128 if key=='OTA32' else None,
                radius_discriminator_AUC=auc(vals, refscore),
                radius_squared_mean=float(vals.mean()), reference_radius_squared_mean=float(refscore.mean()),
                interpretation='sampler-vs-randomized-retrain discrimination, NOT membership inference'))
    # Nominal/true variance ratio changes covariance even with exact center.
    sensitivity = []
    for ratio in [.8, 1., 1.2]:
        n = e.randn((10000, 650), 88900+int(ratio*100))*math.sqrt(ratio)*tau
        empirical = float(n.square().mean())/(tau*tau)
        kl = 650/2*(ratio-1-math.log(ratio))
        sensitivity.append(dict(actual_over_nominal_noise_variance=ratio,
            empirical_output_variance_ratio=empirical, covariance_only_KL=kl,
            Pinsker_TV_upper=min(1., math.sqrt(kl/2))))
    calibration = []
    for n in [64, 256, 1024, 4096, 16384]:
        # Receiver-only silent slots. No clipping, no extra data transmission.
        samples = e.randn((2048, n), 99900+n)
        estimate = samples.square().mean(1)
        ratio = estimate.reciprocal()
        kl = 650/2*(ratio-1-ratio.log())
        calibration.append(dict(silent_real_samples=n, added_real_uses=1.125*n,
            variance_estimator_relative_std=math.sqrt(2/n),
            covariance_KL_mean=float(kl.mean()), covariance_KL_p95=float(torch.quantile(kl, .95)),
            TV_Pinsker_bound_mean=float((kl/2).sqrt().clamp_max(1).mean()),
            overhead_per_deletion_if_shared_over_100=1.125*n/100))
    e.write(e.OUT/'diagnosis.json', dict(posthoc_diagnostic=True,
        main_hyperparameters_unchanged=True, source_code_sha256=e.sha(e.__file__),
        diagnostic_code_sha256=e.sha(__file__), rows=rows,
        variance_miscalibration=sensitivity, silent_calibration_cost=calibration,
        seconds=time.perf_counter()-start,
        caveat='AUC is a finite-sample diagnostic, not a worst-case unlearning or privacy certificate. Shared Hessian draws form clusters.'))
    print({'diagnosis_complete': True, 'rows': rows, 'seconds': time.perf_counter()-start})


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    run(ap.parse_args().data)
