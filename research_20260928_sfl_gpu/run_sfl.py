"""Cohort split-federated training with explicit forward OTA and digital backward.

Original implementation of the equations in PROTOCOL_KO.md, not official code
of SegOTA, SplitFC, or a published low-rank SFL algorithm.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda"
K, GROUPS, BATCH, LOCAL, HIDDEN = 4, 4, 64, 5, 64
CP, BANDWIDTH, PWR = 1.125, 1e6, 1.
CONFIGS = {
    "Feature8-W8": dict(kind="feature", q=64, width=8, bits=8),
    "Latent8-Q16-W8": dict(kind="digital", q=16, width=8, bits=8),
    "Latent4-Q16-W8": dict(kind="digital", q=16, width=8, bits=4),
    "OTA-Q64-W8": dict(kind="ota", q=64, width=8, bits=8),
    "OTA-Q16-W8": dict(kind="ota", q=16, width=8, bits=8),
    "OTA-Q16-W4": dict(kind="ota", q=16, width=4, bits=8),
    "Latent4-Q16-W4": dict(kind="digital", q=16, width=4, bits=4),
    "Ideal-Q16-W8": dict(kind="ideal", q=16, width=8, bits=32),
    "OTA-Stein-Q16-W8": dict(kind="ota", q=16, width=8, bits=8, correction="stein"),
    "OTA-Reg-Q16-W8": dict(kind="ota", q=16, width=8, bits=8, correction="fixed"),
    "OTA-Q16-HET": dict(kind="ota", q=16, width=[4, 8, 12, 4], bits=8),
    "Latent4-Q16-HET": dict(kind="digital", q=16, width=[4, 8, 12, 4], bits=4),
    "OTA-Stein-Q16-HET": dict(kind="ota", q=16, width=[4, 8, 12, 4], bits=8, correction="stein"),
    "OTA-Reg-Q16-HET": dict(kind="ota", q=16, width=[4, 8, 12, 4], bits=8, correction="fixed"),
}
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quant(x, bits):
    if bits >= 32:
        return x
    bound = x.detach().abs().amax().clamp_min(1e-12)
    levels = 2**(bits-1)-1
    return (x / bound * levels).round().clamp(-levels, levels) * (bound / levels)


class Wire(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, forward_bits, backward_bits):
        ctx.backward_bits = backward_bits
        return quant(x, forward_bits)

    @staticmethod
    def backward(ctx, g):
        return quant(g, ctx.backward_bits), None, None


def channels(seed, condition="rician", csi=0.):
    rng = np.random.default_rng(seed)
    z = (rng.normal(size=K)+1j*rng.normal(size=K))/math.sqrt(2)
    if condition == "rician":
        z = (math.sqrt(5)+z)/math.sqrt(6)
    elif condition == "unit":
        z = np.ones(K, dtype=complex)
    h = z * np.sqrt(np.array([1., .8, .6, .4]))
    # Additive complex CSI error with E|e|^2/E|h|^2 = csi^2.
    err = (rng.normal(size=K)+1j*rng.normal(size=K))/math.sqrt(2)
    he = h + csi*np.sqrt(np.array([1., .8, .6, .4]))*err
    return h, he


def rates(h, snr):
    return np.maximum(1e-9, np.minimum(6., np.log2(1+10**(snr/10)*np.abs(h)**2)))


class Ledger:
    def __init__(self):
        self.x = {k: 0. for k in ["forward_uses", "backward_uses", "control_uses", "pilot_uses",
                                    "sync_ul_uses", "sync_dl_uses", "ul_energy", "dl_energy"]}
        self.max_block_power = 0.
        self.max_symbol_power = 0.
        self.noise_nmse_sum = 0.
        self.calls = 0

    def frame(self, config, batch, feature_dim, h, snr, ota_energy=None, signal_dim=None):
        kind, q, bits = config["kind"], config["q"], config["bits"]
        if kind == "ideal":
            return
        rr = rates(h, snr)
        # Assume equal training UL/DL gains. Broadcast limited by the worst link.
        down = float(rr.min())
        if kind == "feature":
            ul = sum((batch*np.array(feature_dim)*bits+32)/rr)
            dl = sum((batch*np.array(feature_dim)*8+32)/rr)
            dim = feature_dim
        else:
            dim = q
            ul = batch*q/2 if kind == "ota" else sum((batch*q*bits+32)/rr)
            dl = (batch*q*8+32)/down  # strong digital baseline uses common gradient too
        # Shared batch IDs and one label source. Data already synchronized within cohort.
        ctrl_dl = (batch*14+96)/down
        ctrl_ul = (batch*4+32)/rr[0]
        if kind == "ota":
            ctrl_ul += sum(32/rr)
            ctrl_dl += (32+64)/down  # K scale reports and common scale
        if config.get("correction"):
            ctrl_dl += 34/down  # trace (float32) and bottleneck sensor index (2 bits)
        ctrl = ctrl_ul+ctrl_dl
        pilots = K*8+8
        self.x["forward_uses"] += CP*float(ul)
        self.x["backward_uses"] += CP*float(dl)
        self.x["control_uses"] += CP*float(ctrl)
        self.x["pilot_uses"] += CP*pilots
        ul_e = float(ota_energy) if kind == "ota" else float(ul)
        self.x["ul_energy"] += CP*(ul_e+K*8+ctrl_ul)
        self.x["dl_energy"] += CP*(float(dl)+8+ctrl_dl)
        self.calls += 1

    def sync(self, counts, seed, snr, initial=False):
        hs = np.stack([channels(seed+g)[0] for g in range(GROUPS)])
        rr = rates(hs, snr)
        ul = 0. if initial else float((np.array(counts)[None, :]*32/rr).sum())
        dl = float((np.array(counts)*32/rr.min(axis=0)).sum())
        ctrl_ul = 0. if initial else float((96/rr).sum())
        ctrl_dl = float((96/rr.min(axis=0)).sum())
        pilot_ul, pilot_dl = (0 if initial else GROUPS*K*8), K*8
        self.x["sync_ul_uses"] += CP*ul
        self.x["sync_dl_uses"] += CP*dl
        self.x["control_uses"] += CP*(ctrl_ul+ctrl_dl)
        self.x["pilot_uses"] += CP*(pilot_ul+pilot_dl)
        self.x["ul_energy"] += CP*(ul+ctrl_ul+pilot_ul)
        self.x["dl_energy"] += CP*(dl+ctrl_dl+pilot_dl)

    def result(self):
        out = dict(self.x)
        out["total_uses"] = sum(v for k, v in self.x.items() if k.endswith("_uses"))
        out["modeled_radio_seconds"] = out["total_uses"]/BANDWIDTH
        out["modeled_ul_joules_at_1W_1MHz"] = self.x["ul_energy"]/BANDWIDTH
        out["modeled_dl_joules_at_1W_1MHz"] = self.x["dl_energy"]/BANDWIDTH
        out["max_block_power"] = self.max_block_power
        out["max_symbol_power"] = self.max_symbol_power
        out["mean_forward_noise_nmse"] = self.noise_nmse_sum/max(1, self.calls)
        return out


def ota(us, h, he, snr, seed, ledger=None):
    # K x B x q, paired real coordinates -> complex channel symbols.
    stack = torch.stack(us)
    packed = torch.complex(stack[..., 0::2], stack[..., 1::2])
    hh = torch.tensor(h, device=DEVICE, dtype=packed.dtype)[:, None, None]
    est = torch.tensor(he, device=DEVICE, dtype=packed.dtype)[:, None, None]
    rms2 = packed.detach().abs().square().mean((1, 2)).clamp_min(1e-12)
    alpha = torch.sqrt(est[:, 0, 0].abs().square()/rms2).min().detach()
    tx = alpha*packed/est
    rg = torch.Generator(device=DEVICE).manual_seed(seed)
    noise = torch.randn((*packed.shape[1:], 2), device=DEVICE, generator=rg)
    noise = torch.complex(noise[..., 0], noise[..., 1])*math.sqrt(10**(-snr/10)/2)
    rx = (hh*tx).sum(0)/alpha + noise/alpha
    actual = torch.stack((rx.real, rx.imag), -1).flatten(-2)
    ideal = stack.sum(0)
    # Perfect CSI: exact derivative holding the transmitted scale fixed.
    # Imperfect CSI stress: nominal/common backward ignores unknown residual gain.
    out = ideal+(actual-ideal).detach()
    energy = float(tx.detach().abs().square().sum())
    if ledger is not None:
        ledger.max_block_power = max(ledger.max_block_power, float(tx.detach().abs().square().mean((1, 2)).max()))
        ledger.max_symbol_power = max(ledger.max_symbol_power, float(tx.detach().abs().square().max()))
        ledger.noise_nmse_sum += float((actual-ideal).detach().square().sum() / ideal.detach().square().sum().clamp_min(1e-12))
    return out, energy


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        q = cfg["q"]
        ws = cfg["width"] if isinstance(cfg["width"], list) else [cfg["width"]]*K
        self.ws = ws
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(w, 2*w, 3, padding=1), nn.ReLU(), nn.Flatten())
            for w in ws])
        self.cs = [98*w for w in ws]
        self.projections = nn.ModuleList([nn.Linear(c, q, bias=False) for c in self.cs])
        self.shared = nn.Identity() if q == HIDDEN else nn.Linear(q, HIDDEN, bias=False)
        self.bias = nn.Parameter(torch.zeros(HIDDEN))
        # Smooth suffix is needed for the ordinary Gaussian Stein/Hessian identity.
        self.head = nn.Sequential(nn.SiLU(), nn.Linear(HIDDEN, 32), nn.SiLU(), nn.Linear(32, 10))

    def forward(self, x, seed, ledger=None, snr=20., condition="rician", csi=0.):
        views = [x[..., :14, :14], x[..., :14, 14:], x[..., 14:, :14], x[..., 14:, 14:]]
        hs = [enc(v.contiguous()) for enc, v in zip(self.encoders, views)]
        kind, bits = self.cfg["kind"], self.cfg["bits"]
        h, he = channels(seed, condition, csi)
        if kind == "feature":
            hs = [Wire.apply(z, bits, 8) for z in hs]
        us = [p(z) for p, z in zip(self.projections, hs)]
        energy = None
        if kind == "ota":
            z, energy = ota(us, h, he, snr, seed+817, ledger)
        elif kind == "digital":
            z = sum(Wire.apply(u, bits, 32) for u in us)
        else:
            z = sum(us)
        if kind not in ["feature", "ideal"]:
            z = Wire.apply(z, 32, 8)
        self.latest_z, self.latest_us, self.latest_h = z, us, h
        if ledger is not None:
            ledger.frame(self.cfg, len(x), self.cs, h, snr, energy)
        return self.head(self.shared(z)+self.bias)

    def correction_proxy(self, loss, snr, seed):
        mode = self.cfg.get("correction")
        if not mode:
            return loss.new_zeros(()), 0.
        z = self.latest_z
        if mode == "stein":
            delta = torch.autograd.grad(loss, z, create_graph=True, retain_graph=True)[0]
            rg = torch.Generator(device=DEVICE).manual_seed(seed+411)
            v = torch.randint(0, 2, z.shape, device=DEVICE, generator=rg).to(z.dtype)*2-1
            hv = torch.autograd.grad((delta*v).sum(), z, retain_graph=True)[0]
            trace = (hv*v).sum().detach()
        else:
            # Constant curvature proxy. Deliberately simple regularization baseline.
            trace = loss.new_tensor(1.)
        variances = torch.stack([u.detach().square().mean() for u in self.latest_us])
        gains = torch.tensor(np.abs(self.latest_h)**2, device=DEVICE)
        winner = int((variances/gains).argmax())
        u = self.latest_us[winner]
        coefficient = (10**(-snr/10)*trace/(u.numel()*gains[winner])).detach()
        # This expression is a client-side gradient carrier, not a loss value to report.
        return .5*coefficient*u.square().sum(), float(trace)

    def client_counts(self):
        counts = []
        for e, p in zip(self.encoders, self.projections):
            n = sum(t.numel() for t in e.parameters())
            if self.cfg["kind"] != "feature":
                n += sum(t.numel() for t in p.parameters())
            counts.append(n)
        return counts

    def compute(self):
        q = self.cfg["q"]
        encs = [196*9*w+49*18*w*w for w in self.ws]
        projs = [0 if self.cfg["kind"] == "feature" else c*q for c in self.cs]
        macs = [a+b for a, b in zip(encs, projs)]
        server_proj = sum(self.cs)*q if self.cfg["kind"] == "feature" else 0
        if q != HIDDEN:
            server_proj += q*HIDDEN
        return dict(client_forward_MAC_per_sample=max(macs), client_MACs=macs,
                    encoder_MAC=encs, projection_MAC=projs,
                    server_forward_MAC_per_sample=server_proj+64*32+32*10,
                    client_parameters=max(self.client_counts()),
                    client_parameter_gradient_Adam_bytes=16*max(self.client_counts()),
                    client_cut_activation_bytes_batch=4*BATCH*max(self.cs),
                    global_parameters=sum(p.numel() for p in self.parameters()))


def data(path, seed):
    def read(pre):
        x = np.fromfile(path/(pre+"-images-idx3-ubyte"), dtype=np.uint8, offset=16).reshape(-1, 1, 28, 28)
        y = np.fromfile(path/(pre+"-labels-idx1-ubyte"), dtype=np.uint8, offset=8).astype(np.int64)
        return x, y
    x, y = read("train")
    xt, yt = read("t10k")
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in range(10):
        ix = rng.permutation(np.flatnonzero(y == c))
        tr.extend(ix[:1200]); va.extend(ix[1200:1400])
        te.extend(rng.permutation(np.flatnonzero(yt == c))[:300])
    tr, va, te = np.array(tr), np.array(va), np.array(te)
    tx = torch.tensor(x[tr].copy(), device=DEVICE, dtype=torch.float32)/255.
    ty = torch.tensor(y[tr], device=DEVICE)
    vx = torch.tensor(x[va].copy(), device=DEVICE, dtype=torch.float32)/255.
    vy = torch.tensor(y[va], device=DEVICE)
    ex = torch.tensor(xt[te].copy(), device=DEVICE, dtype=torch.float32)/255.
    ey = torch.tensor(yt[te], device=DEVICE)
    order = np.argsort(y[tr], kind="stable")
    shards = np.array_split(order, 12)
    perm = rng.permutation(12)
    ids = [np.concatenate([shards[j] for j in perm[3*g:3*g+3]]) for g in range(GROUPS)]
    meta = dict(train=12000, validation=2000, test=3000,
                cohort_histograms=[np.bincount(y[tr][i], minlength=10).tolist() for i in ids],
                train_index_sha=hashlib.sha256(tr.tobytes()).hexdigest(),
                validation_index_sha=hashlib.sha256(va.tobytes()).hexdigest(),
                test_index_sha=hashlib.sha256(te.tobytes()).hexdigest(),
                raw_sha={p.name: sha(p) for p in path.glob("*ubyte")})
    return tx, ty, vx, vy, ex, ey, ids, meta


@torch.no_grad()
def evaluate(model, x, y, seed, snr=20., condition="rician", csi=0., chunk=100):
    model.eval()
    correct, loss = 0, 0.
    for i in range(0, len(y), chunk):
        o = model(x[i:i+chunk], seed+i, snr=snr, condition=condition, csi=csi)
        correct += int((o.argmax(1) == y[i:i+chunk]).sum())
        loss += float(F.cross_entropy(o, y[i:i+chunk], reduction="sum"))
    return dict(accuracy=correct/len(y), loss=loss/len(y))


def run_one(name, cfg, ds, seed, rounds, out, train_snr=20.):
    x, y, xv, yv, xt, yt, ids, meta = ds
    torch.manual_seed(seed)
    global_model = Model(cfg).to(DEVICE)
    locals_ = [copy.deepcopy(global_model) for _ in range(GROUPS)]
    opts = [torch.optim.Adam(m.parameters(), lr=.001) for m in locals_]
    ledger = Ledger()
    hist = []
    trace_values = []
    correction_events = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); start = time.perf_counter()
    counts = global_model.client_counts()
    if cfg["kind"] != "ideal":
        ledger.sync(counts, seed*100000+54000, train_snr, initial=True)
    for r in range(rounds):
        state = global_model.state_dict()
        for g, (m, opt) in enumerate(zip(locals_, opts)):
            m.load_state_dict(state); m.train()
            # Same indices and channels for paired architecture/method comparisons.
            rng = np.random.default_rng(seed*100003+r*GROUPS+g)
            for step in range(LOCAL):
                ix = torch.tensor(rng.choice(ids[g], BATCH, replace=False), device=DEVICE)
                key = seed*10000000+r*1000+g*100+step
                opt.zero_grad(set_to_none=True)
                pred = m(x[ix], key, ledger, snr=train_snr)
                loss = F.cross_entropy(pred, y[ix])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite {name} r={r}")
                if cfg.get("correction"):
                    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    ev0.record()
                proxy, trace = m.correction_proxy(loss, train_snr, key)
                if cfg.get("correction"):
                    ev1.record()
                    correction_events.append((ev0, ev1))
                    trace_values.append(trace)
                (loss+proxy).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.)
                opt.step()
        merged = {key: sum(m.state_dict()[key] for m in locals_)/GROUPS for key in state}
        global_model.load_state_dict(merged)
        if cfg["kind"] != "ideal":
            ledger.sync(counts, seed*100000+r*GROUPS+55000, train_snr)
        if (r+1) % 10 == 0 or r+1 == rounds:
            val = evaluate(global_model, xv, yv, seed*700000+r*10000, snr=train_snr)
            row = dict(round=r+1, validation=val, ledger=ledger.result())
            hist.append(row)
            print(json.dumps(dict(event="progress", method=name, seed=seed, round=r+1,
                                  val=round(val["accuracy"], 4))), flush=True)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-start
    checks = {}
    for condition, snr, csi in [("rician", 20, 0), ("rician", 10, 0),
                                ("rayleigh", 20, 0), ("rician", 20, .05)]:
        vals = [evaluate(global_model, xt, yt, seed*900000+rep*50000,
                         snr, condition, csi) for rep in range(3)]
        checks[f"{condition}_{snr}dB_csi{csi}"] = vals
    result = dict(method=name, config=cfg, seed=seed, rounds=rounds, batch=BATCH, local_steps=LOCAL,
                  groups=GROUPS, sensors_per_group=K, dataset=meta, history=hist, tests=checks,
                  ledger=ledger.result(), compute=global_model.compute(), gpu_wall_seconds=elapsed,
                  process_peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  train_snr=train_snr,
                  correction_gpu_milliseconds=sum(a.elapsed_time(b) for a, b in correction_events),
                  trace_mean=float(np.mean(trace_values)) if trace_values else None,
                  trace_negative_fraction=float(np.mean(np.array(trace_values)<0)) if trace_values else None,
                  code_sha=sha(__file__))
    if cfg["kind"] == "ota":
        assert result["ledger"]["max_block_power"] <= 1.00001
    save(out/f"{name}_seed{seed}.json", result)
    print(json.dumps(dict(event="finished", method=name, seed=seed, wall_seconds=round(elapsed, 2),
                         accuracy=np.mean([v["accuracy"] for v in checks["rician_20dB_csi0"]]))), flush=True)
    del global_model, locals_, opts
    torch.cuda.empty_cache()


def verify():
    torch.manual_seed(441)
    dt = torch.float64
    hs = [torch.randn(5, c, device=DEVICE, dtype=dt, requires_grad=True) for c in [3, 5, 7, 4]]
    aa = [torch.randn(6, h.shape[1], device=DEVICE, dtype=dt, requires_grad=True) for h in hs]
    bb = torch.randn(8, 6, device=DEVICE, dtype=dt, requires_grad=True)
    lhs = sum(h@a.T for h, a in zip(hs, aa))@bb.T
    rhs = torch.cat(hs, 1) @ torch.cat([bb@a for a in aa], 1).T
    delta = torch.randn_like(lhs)
    params = hs+aa+[bb]
    gl = torch.autograd.grad((lhs*delta).sum(), params, retain_graph=True)
    gr = torch.autograd.grad((rhs*delta).sum(), params)
    ferr = float((lhs-rhs).detach().abs().max())
    berr = max(float((a-b).abs().max()) for a, b in zip(gl, gr))
    assert ferr < 1e-11 and berr < 1e-11
    # Independent examples do not share the same loss after logit summation.
    logits = torch.tensor([[3., -3.], [-3., 3.]], device=DEVICE, dtype=dt)
    independent = float(F.cross_entropy(logits, torch.tensor([0, 1], device=DEVICE)))
    wrongly_mixed = float(F.cross_entropy(logits.sum(0, keepdim=True), torch.tensor([0], device=DEVICE)))
    assert wrongly_mixed > independent+.5
    ll = Ledger()
    us = [torch.randn(2000, 16, device=DEVICE, dtype=dt) for _ in range(K)]
    h, he = channels(912, "unit")
    noisy, energy = ota(us, h, he, 20., 776, ll)
    assert ll.max_block_power < 1.00001 and energy > 0
    obj = dict(forward_max_error=ferr, backward_max_error=berr,
               separate_loss=independent, wrongly_summed_loss=wrongly_mixed,
               max_average_symbol_power=ll.max_block_power, passed=True)
    save(ROOT/"math_validation.json", obj)
    print(json.dumps(obj), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT/"results")
    ap.add_argument("--seeds", default="311,312,313")
    ap.add_argument("--rounds", type=int, default=80)
    ap.add_argument("--snr", type=float, default=20.)
    ap.add_argument("--methods", default=",".join(CONFIGS))
    args = ap.parse_args()
    if args.verify:
        verify()
    else:
        assert args.data and torch.cuda.is_available()
        for seed in map(int, args.seeds.split(",")):
            ds = data(args.data, seed)
            for name in args.methods.split(","):
                file = args.out/f"{name}_seed{seed}.json"
                if file.exists():
                    print("SKIP completed", file.name, flush=True)
                    continue
                run_one(name, CONFIGS[name], ds, seed, args.rounds, args.out, args.snr)
