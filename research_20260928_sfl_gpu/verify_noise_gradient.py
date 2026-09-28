"""GPU audit of gradient correction for signal-dependent receiver noise."""
import json
import math
from pathlib import Path
import torch

torch.set_num_threads(2)
torch.manual_seed(8101)
torch.backends.cuda.matmul.allow_tf32 = False
dev, dt = "cuda", torch.float64
K, n, r, draws = 3, 12, 7, 100000
u = torch.randn(K, n, device=dev, dtype=dt)
h = torch.tensor([1., .3, .8], device=dev, dtype=dt)
sigma2 = .2
v = u.square().mean(1)/h.square()
winner = int(v.argmax())
tau = (sigma2*v[winner]).sqrt()
M = torch.randn(r, n, device=dev, dtype=dt)/math.sqrt(n)
target = torch.randn(r, device=dev, dtype=dt)
tr = M.square().sum()
delta0 = (u.sum(0)@M.T-target)@M
correction = sigma2*tr*u[winner]/(n*h[winner]**2)
truth = delta0+correction
eps = torch.randn(draws, n, device=dev, dtype=dt)
z = u.sum(0)[None, :]+tau*eps
delta = (z@M.T-target)@M
dtau_du = math.sqrt(sigma2)*u[winner]/(n*h[winner]**2*v[winner].sqrt())
pathwise = delta+(delta*eps).sum(1, keepdim=True)*dtau_du
stein = delta+correction
probe = torch.randint(0, 2, (draws, n), device=dev).to(dt)*2-1
trace_estimates = ((probe@M.T@M)*probe).sum(1)
stein_hutch = delta+sigma2*trace_estimates[:, None]*u[winner]/(n*h[winner]**2)
checks = {}
for name, values in [("nominal", delta), ("oracle_pathwise", pathwise), ("scalar_stein", stein),
                     ("scalar_stein_one_probe", stein_hutch)]:
    estimate = values.mean(0)
    err = estimate-truth
    se = values.std(0, unbiased=True)/math.sqrt(draws)
    checks[name] = dict(error_norm=float(err.norm()), relative_error=float(err.norm()/truth.norm()),
                        max_error_in_standard_errors=float((err.abs()/se.clamp_min(1e-15)).max()),
                        estimate=estimate.tolist())
assert checks["oracle_pathwise"]["max_error_in_standard_errors"] < 6
assert checks["scalar_stein"]["max_error_in_standard_errors"] < 6
assert checks["scalar_stein_one_probe"]["max_error_in_standard_errors"] < 6
assert checks["nominal"]["error_norm"] > 10*checks["scalar_stein"]["error_norm"]
# Check the derivative of the expected loss by exact differentiable algebra.
ud = u.clone().requires_grad_(True)
var = sigma2*(ud.square().mean(1)/h.square()).max()
expected_loss = .5*((ud.sum(0)@M.T-target).square().sum()+var*tr)
direct = torch.autograd.grad(expected_loss, ud)[0]
assert torch.allclose(direct[winner], truth, atol=1e-12, rtol=1e-12)
for k in range(K):
    if k != winner:
        assert torch.allclose(direct[k], delta0, atol=1e-12, rtol=1e-12)
# Validate the real/complex factor of two using the actual radio implementation.
import run_sfl
actual, _ = run_sfl.ota([u[k].repeat(draws, 1) for k in range(K)],
                        h.cpu().numpy(), h.cpu().numpy(), -10*math.log10(sigma2), 9941)
observed_variance = float((actual-u.sum(0)).square().mean())
assert abs(observed_variance/float(tau**2)-1) < .01
obj = dict(passed=True, gpu=torch.cuda.get_device_name(0), draws=draws, dimension=n,
           winner=winner, sigma2=sigma2, receiver_real_variance=float(tau**2),
           trace=float(tr), probe_trace_mean=float(trace_estimates.mean()),
           radio_empirical_real_variance=observed_variance,
           true_gradient=truth.tolist(), estimators=checks,
           scope="quadratic smooth suffix, exact CSI, continuous backward, block-power normalization",
           caveat="Unbiasedness of expected noisy objective gradient, not equality to noiseless gradient.")
Path(__file__).with_name("noise_gradient_validation.json").write_text(
    json.dumps(obj, indent=2)+"\n", encoding="utf-8")
print(json.dumps({k: v for k, v in obj.items() if k not in ["true_gradient"]}))
