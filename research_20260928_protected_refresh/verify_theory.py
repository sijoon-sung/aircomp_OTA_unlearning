"""Independent, bounded theory/PHY checks; NOT a real-data accuracy benchmark."""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SEED = 202609289
B, VAR, NU2, RHO_CAP = 1., 8., .01, .5
RHO0 = (2*B)**2/(2*VAR)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def rho(beta):
    return RHO0/(1-beta**2), 2*RHO0/(1+beta)


def interval():
    return max(0., 2*RHO0/RHO_CAP-1), math.sqrt(1-RHO0/RHO_CAP)


def beta_energy(a, r):
    lo, hi = interval()
    return float(np.clip(2*r/(2*r+a+math.sqrt(a*a+4*a*r)), lo, hi))


def energy(beta, a, r):
    return NU2/VAR*(beta*beta*a+(1-beta)**2*r)/(1-beta*beta)


def repeats(beta, h):
    coef = np.array([-beta]+[1-beta]*(len(h)-1))
    scale_min = float(np.max(np.abs(coef)*B/h))
    return max(1, math.ceil(NU2*scale_min**2/((1-beta**2)*VAR)-1e-12))


def cost(beta, h, q=296, d=16, c=10):
    r = repeats(beta, h)
    active = len(h) if beta > 0 else len(h)-1
    rate = .5*math.log2(101)
    control, final_bits, pilots = 384, d*c*32, active*8
    total = 1.125*(q*r+pilots+(control+final_bits)/rate)
    r0 = max(1, math.ceil(NU2*float(np.max(B/h))**2/VAR-1e-12))
    prep = 1.125*(q*r0+len(h)*8+(control+final_bits)/rate)
    a, retained = B*B/h[0]**2, float(np.sum(B*B/h[1:]**2))
    return dict(beta=beta, repeats=r, payload_real_uses=q*r,
                pilot_real_uses=pilots, control_bits=control,
                model_broadcast_bits=final_bits, total_deletion_real_uses=total,
                source_preparation_real_uses=prep, lifecycle_real_uses=total+prep,
                payload_tx_energy_upper_bound=energy(beta,a,retained),
                rho_A=rho(beta)[0], rho_R=rho(beta)[1])


def main():
    assert torch.cuda.is_available(), 'This run intentionally requires CUDA.'
    torch.set_default_dtype(torch.float64)
    dev = 'cuda'
    generator = torch.Generator(device=dev).manual_seed(SEED)
    def rand(*shape):
        return torch.randn(shape, device=dev, generator=generator)
    start = time.perf_counter()
    k, d, c = 10, 4, 2
    ix = torch.triu_indices(d,d,device=dev)
    qh, q = len(ix[0]), len(ix[0])+d*c
    x, w = rand(k,32,d), rand(d,c)
    y = x@w + .1*rand(k,32,c)
    h = x.transpose(-1,-2)@x/32
    cc = x.transpose(-1,-2)@y/32
    statistics = torch.cat((h[:,ix[0],ix[1]],cc.reshape(k,-1)),dim=1)
    statistics = statistics/torch.maximum(statistics.norm(dim=1,keepdim=True)/B,
                                         torch.ones((k,1),device=dev))
    retained, target = statistics[1:].sum(0), statistics[0]
    def decode(v):
        matrix = torch.zeros((*v.shape[:-1],d,d),device=dev)
        matrix[...,ix[0],ix[1]]=v[...,:qh]/k
        matrix[...,ix[1],ix[0]]=v[...,:qh]/k
        eigen, basis = torch.linalg.eigh(matrix)
        matrix = (basis*eigen.clamp_min(0).unsqueeze(-2))@basis.transpose(-1,-2)
        return torch.linalg.solve(matrix+.2*torch.eye(d,device=dev),
                                  v[...,qh:].reshape(*v.shape[:-1],d,c)/k)
    # Public normalization by fixed k; active count does not vary across neighbors.
    z0 = math.sqrt(VAR)*rand(1024,q)
    src = retained+target+z0
    source_cf_model_mse = float((decode(src)-decode(retained-target+z0)).square().mean())
    assert source_cf_model_mse > 1e-10, 'A must actually affect the learned source.'
    checks = []
    for beta in [0.,.25,.5,math.sqrt(.5),.9,.99]:
        z1 = math.sqrt((1-beta**2)*VAR)*rand(1024,q)
        aggregate = (1-beta)*retained-beta*target+z1
        refresh = beta*src+aggregate
        coupled_reference = retained+beta*z0+z1
        changed_target = -target
        changed_source = retained+changed_target+z0
        changed_refresh = beta*changed_source+(1-beta)*retained-beta*changed_target+z1
        abs_error = float((refresh-coupled_reference).abs().max())
        model_error = float((decode(refresh)-decode(coupled_reference)).abs().max())
        cf_error = float((refresh-changed_refresh).abs().max())
        cov = VAR*np.array([[1,beta],[beta,1]])
        precision = np.linalg.inv(cov)
        shift_a = np.array([2*B,0.])
        shift_r = np.array([2*B,2*B])
        direct_a = .5*shift_a@precision@shift_a
        direct_r = .5*shift_r@precision@shift_r
        assert abs_error < 1e-12 and model_error < 1e-11 and cf_error < 1e-12
        assert np.allclose([direct_a,direct_r],rho(beta),rtol=1e-12)
        checks.append(dict(beta=beta,statistic_error=abs_error,decoder_error=model_error,
                           counterfactual_A_error=cf_error,rho_A=direct_a,rho_R=direct_r))
    beta = .5
    zz0,zz1 = rand(100000,2),rand(100000,2)
    refreshed_noise = beta*zz0+math.sqrt(1-beta**2)*zz1
    joint = torch.stack((zz0[:,0],refreshed_noise[:,0]),dim=0)
    cov_error = float((torch.cov(joint)-torch.tensor([[1.,beta],[beta,1.]],device=dev)).abs().max())
    mean_error = float(refreshed_noise.mean(0).abs().max())
    assert cov_error < .025 and mean_error < .025
    # beta=1 makes noise cancel from source-output difference, exposing A exactly.
    leak_error = float(((src-(retained+z0))-target).abs().max())
    assert leak_error < 1e-12
    eps = lambda rr: rr+2*math.sqrt(rr*math.log(1e5))
    grid = np.linspace(*interval(),100001)
    broad_grid = np.linspace(0,.99999,100000)
    rr = np.maximum(RHO0/(1-broad_grid**2),2*RHO0/(1+broad_grid))
    best_minmax = float(broad_grid[np.argmin(rr)])
    assert abs(best_minmax-.5) < 2e-5
    nrng = np.random.default_rng(SEED)
    max_opt_gap = 0.
    max_repeat_excess = 0
    for _ in range(200):
        hh = np.exp(nrng.uniform(-5,2,10))
        a, r = B*B/hh[0]**2,float(np.sum(B*B/hh[1:]**2))
        bb = beta_energy(a,r)
        eg = energy(grid,a,r)
        gap = energy(bb,a,r)-float(eg.min())
        assert gap <= 1e-10*max(1,float(eg.min()))
        max_opt_gap=max(max_opt_gap,gap)
        aa, ar = B/hh[0],float(np.max(B/hh[1:]))
        bp = float(np.clip(ar/(aa+ar),*interval()))
        vgrid = NU2*np.maximum(grid**2*aa**2,(1-grid)**2*ar**2)/(VAR*(1-grid**2))
        rg = int(np.maximum(1,np.ceil(vgrid-1e-12)).min())
        excess = repeats(bp,hh)-rg
        assert excess <= 0
        max_repeat_excess=max(max_repeat_excess,excess)
    # Actual simultaneous transmissions, not individually revealed at receiver.
    gains = torch.tensor([.8,1.1,.6,1.2,.9,1.3,.7,1.,1.5,.95],device=dev)
    r = repeats(beta,gains.cpu().numpy())
    scale = math.sqrt(r*(1-beta**2)*VAR/NU2)
    coef = torch.tensor([-beta]+[1-beta]*(k-1),device=dev)
    tx = coef[:,None]*statistics/(scale*gains[:,None])
    mean = (gains[:,None]*tx).sum(0)
    raw = mean[None,None,:]+math.sqrt(NU2)*rand(10000,r,q)
    received = scale*raw.mean(1)
    required = (1-beta)*retained-beta*target
    mean_mapping_error=float((scale*mean-required).abs().max())
    empirical_var=float((received-required).square().mean())
    max_tx_block_energy=float(tx.square().sum(1).max())
    actual_energy=float(r*tx.square().sum())
    af=float(statistics[0].square().sum()/gains[0]**2)
    rf=float((statistics[1:].square().sum(1)/gains[1:]**2).sum())
    assert max_tx_block_energy <= 1+1e-12 and mean_mapping_error<1e-12
    assert abs(empirical_var/((1-beta**2)*VAR)-1)<.02
    assert abs(actual_energy-energy(beta,af,rf))<1e-12
    cases={
        'unit':np.ones(10),
        'weak_deleted':np.array([.01]+[1.]*9),
        'weak_retained':np.array([1.,.01]+[1.]*8),
    }
    costs={}
    for name,hh in cases.items():
        a, r = B*B/hh[0]**2,float(np.sum(B*B/hh[1:]**2))
        aa, ar = B/hh[0],float(np.max(B/hh[1:]))
        costs[name]=[cost(0.,hh),cost(.5,hh),cost(beta_energy(a,r),hh),
                     cost(float(np.clip(ar/(aa+ar),*interval())),hh)]
    torch.cuda.synchronize()
    result=dict(scope='THEORY_AND_SYNTHETIC_PHY_ONLY; no real-data accuracy measurement',
                seed=SEED,device=torch.cuda.get_device_name(),torch_version=torch.__version__,
                elapsed_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                rho0=RHO0,joint_rho_cap=RHO_CAP,delta=1e-5,epsilon_cap=eps(RHO_CAP),
                beta_interval=interval(),beta_half_rho=max(rho(.5)),beta_half_epsilon=eps(max(rho(.5))),
                checks=checks,mc_covariance_abs_error=cov_error,mc_mean_abs_error=mean_error,
                source_counterfactual_A_model_mse=source_cf_model_mse,
                beta_one_exact_difference_leak_error=leak_error,minmax_beta_grid=best_minmax,
                optimizer_max_energy_excess=max_opt_gap,optimizer_max_repeat_excess=max_repeat_excess,
                phy=dict(mean_error=mean_mapping_error,empirical_update_variance=empirical_var,
                         expected_update_variance=(1-beta**2)*VAR,max_tx_block_energy=max_tx_block_energy,
                         actual_energy=actual_energy,formula_energy=energy(beta,af,rf)),
                costs=costs,
                utility_risk=[dict(K=kk,q=qq,expected_noise_norm_squared_over_B_squared=qq*VAR/kk**2)
                              for kk in [10,100,1000] for qq in [296,2720]],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL_KO.md').read_bytes()).hexdigest(),
                all_assertions_passed=True)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['device','elapsed_seconds','peak_cuda_bytes','beta_half_epsilon','mc_covariance_abs_error','all_assertions_passed']},indent=2))
    for name,rows in costs.items():
        print(name,json.dumps(rows[:2]))


if __name__=='__main__':
    main()
