"""코드 분할 다중 shard AirComp: 모든 shard 가 같은 자원에 동시에 보내고, 서버가 shard 코드로 합을 분리한다.

송신 (shard k 의 client i, 라운드 t)
  shard 마다 길이 L 의 ±1 코드 c_k 를 둔다. client i 는 업데이트 심볼 하나를 L 칩으로 펼쳐 보낸다:
      chip_l = b_i * x_i * c_k[l],  l = 0..L-1,   b_i = 1 / (n_k |h_i| alpha_k)   (shard 안 전력 정렬)
  칩당 평균 전력 <= P. 자원: 심볼당 L 칩 (모든 shard 가 공유).
  client 마다 칩 단위 타이밍 오차 delta_i ~ U[0, delta_max] (정적). 사각 칩 펄스에서 지연된 코드는
      c_i^eff[l] = (1 - delta_i) c_k[l] + delta_i c_k[l-1]    (순환)
채널: Y[l] = sum_i |h_i| * chip_l(i) + Z[l],  Z[l] ~ N(0, sigma2)   (모든 shard 의 client 가 한 신호로 더해짐)
수신 (shard k): 코드 c_k 로 역확산 후 scale
      r_k = alpha_k * (1/L) sum_l c_k[l] Y[l]
         = sum_{i in S_k} g_{ik} x_i / n_k                         (자기 shard, 타이밍 오차가 없으면 g=1 -> 평균)
         + sum_{j != k} (alpha_k / alpha_j) sum_{i in S_j} g_{ik} x_i / n_j   (다른 shard 에서 새는 간섭)
         + alpha_k * (1/L) c_k . Z                                    (잡음, 분산 alpha_k^2 sigma2 / L)
      g_{ik} = (1/L) sum_l c_i^eff[l] c_k[l]
코드 종류: walsh (Hadamard 행, 동기면 완전 직교) / pn (무작위 ±1, 상호상관 ~ 1/sqrt(L)).
측정용으로 간섭 항과 자기 항의 에너지, 실제 자기 항 가중치의 평균 대비 편차를 장부에 남긴다.
"""
import math
from dataclasses import dataclass
import numpy as np
import torch
from .radio import Ledger
from .fl import key, CLIP

@dataclass
class CdmaConfig:
    sigma2: float = 0.01
    P: float = 1.0
    C: float = 1.0
    code: str = 'walsh'      # 'walsh' | 'pn'
    L: int = 4
    delay_max: float = 0.0   # 칩 단위 타이밍 오차 상한

def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H

def make_codes(cfg: CdmaConfig, K, seed):
    if cfg.code == 'walsh':
        L = 1 << max(0, math.ceil(math.log2(max(cfg.L, K))))
        H = hadamard(L)
        return H[:K].copy()                       # [K, L]
    if cfg.code == 'pn':
        return np.stack([np.random.default_rng(key(seed, 'pn', k, cfg.L)).choice([-1.0, 1.0], cfg.L) for k in range(K)])
    raise ValueError(cfg.code)

def effective_gain(codes, shard_of_client, delays):
    """g[i, k] = (1/L) sum_l c_i^eff[l] c_k[l]. codes [K, L], delays [N]."""
    K, L = codes.shape; N = len(shard_of_client)
    g = np.zeros((N, K))
    for i in range(N):
        c = codes[shard_of_client[i]]; ce = (1 - delays[i]) * c + delays[i] * np.roll(c, 1)
        g[i] = codes @ ce / L
    return g

class CdmaSystem:
    """한 system(동시에 보내는 shard 묶음)의 코드·지연·이득."""
    def __init__(self, cfg: CdmaConfig, n_shards_total, seed, delays):
        self.cfg = cfg; self.codes = make_codes(cfg, n_shards_total, seed); self.L = self.codes.shape[1]; self.delays = delays

    def transmit(self, shard_ids, X_list, h_list, D, noise_key, device):
        """shard_ids: 이번 라운드 동시에 보내는 shard 번호들, X_list[s]: [n_s, D] clip 된 업데이트, h_list[s]: [n_s] 채널 진폭 (client id 와 함께).
        반환: {shard: 수신 추정 [D]}, {shard: Ledger}, 진단 dict."""
        cfg = self.cfg; L = self.L
        alphas = {}; coef = {}
        for s, (X, (ids, hs)) in zip(shard_ids, zip(X_list, h_list)):
            n = X.shape[0]; hmin = float(np.min(hs))
            alphas[s] = cfg.C / (n * hmin * math.sqrt(cfg.P * D))           # shard 안 전력 정렬 (칩당 전력 <= P)
        # 칩 신호 합성: Y = sum_i |h_i| b_i x_i (x) c_i^eff  -> 메모리 절약을 위해 역확산 결과를 직접 계산
        # r_k = alpha_k [ sum_i g_ik |h_i| b_i x_i + (1/L) c_k . Z ]
        g = torch.Generator(device=device).manual_seed(int(noise_key))
        Z = torch.randn(L, D, generator=g, device=device) * math.sqrt(cfg.sigma2)   # 칩별 잡음 (실제 생성)
        codes_t = torch.as_tensor(self.codes, dtype=Z.dtype, device=device)
        # 각 client 의 송신 진폭 |h_i| b_i x_i  (= x_i / (n alpha))
        contrib = []; meta = []
        for s, X, (ids, hs) in zip(shard_ids, X_list, h_list):
            n = X.shape[0]; h = torch.as_tensor(hs, dtype=X.dtype, device=device)
            b = 1.0 / (n * h * alphas[s]); S = X * b[:, None]
            pr = float(((S * S).mean(1) / cfg.P).max()); assert pr <= 1 + 1e-6, pr
            contrib.append((h[:, None] * S)); meta.append((s, ids, pr, float((S * S).sum()) * L))
        out, leds, diag = {}, {}, {}
        for s in shard_ids:
            own = torch.zeros(D, device=device, dtype=Z.dtype); leak = torch.zeros(D, device=device, dtype=Z.dtype)
            for (s2, ids, _, _), A in zip(meta, contrib):
                gk = torch.as_tensor([self._g(i, s2, s) for i in ids], dtype=Z.dtype, device=device)
                term = (gk[:, None] * A).sum(0)
                if s2 == s: own += term
                else: leak += term
            noise = (codes_t[s][:, None] * Z).sum(0) / L
            r = alphas[s] * (own + leak + noise)
            out[s] = r
            ownv = alphas[s] * own; leakv = alphas[s] * leak
            diag[s] = dict(own_energy=float((ownv * ownv).sum()), leak_energy=float((leakv * leakv).sum()),
                           noise_energy=float((alphas[s] * noise).pow(2).sum()))
            e = next(m for m in meta if m[0] == s)
            leds[s] = Ledger(ul_symbols=L * D, ul_time=L * D, dl_bits=32 * D, energy=e[3], repeats=1, rounds=1, tx_count=len(e[1]),
                             mse_sum=float(D * alphas[s] ** 2 * cfg.sigma2 / L), max_power_ratio=e[2])
        return out, leds, diag

    def _g(self, i, s_tx, s_rx):
        c = self.codes[s_tx]; d = self.delays[i]; ce = (1 - d) * c + d * np.roll(c, 1)
        return float(self.codes[s_rx] @ ce / self.L)

def run_rounds_cdma(tr, jobs, h, seed, cdma_cfg_of, delays, log=None, log_every=0):
    """jobs: fl.ShardJob 목록. 같은 system 의 job 들은 같은 자원에 동시에 보내며 코드로 분리된다.
    job.tape 를 shard 코드 번호로 쓴다. cdma_cfg_of(system) -> CdmaConfig.
    반환: W, ledgers, diagnostics(job 별 own/leak/noise 에너지 합)."""
    dev = tr.device; D = tr.D
    W = torch.stack([j.w0.to(dev).clone() for j in jobs])
    leds = [Ledger() for _ in jobs]; diags = [dict(own_energy=0.0, leak_energy=0.0, noise_energy=0.0, rounds=0) for _ in jobs]
    systems = {}
    for ji, j in enumerate(jobs):
        systems.setdefault(j.system, []).append(ji)
    cs = {name: CdmaSystem(cdma_cfg_of(name), 1 + max(jobs[ji].tape for ji in idx), seed, delays) for name, idx in systems.items()}
    t_lo = min(j.t0 for j in jobs); t_hi = max(j.T for j in jobs)
    for t in range(t_lo, t_hi):
        live = [ji for ji, j in enumerate(jobs) if j.t0 <= t < j.T]
        pairs = [(ji, i) for ji in live for i in jobs[ji].members]
        if not pairs:
            continue
        jidx = torch.tensor([p[0] for p in pairs], device=dev)
        U, _ = tr.updates(W[jidx], [p[1] for p in pairs], t, [jobs[p[0]].salt for p in pairs])
        rows_of = {}
        for k, (ji, _) in enumerate(pairs):
            rows_of.setdefault(ji, []).append(k)
        for name, idx in systems.items():
            act = [ji for ji in idx if ji in rows_of]
            if not act:
                continue
            shard_ids = [jobs[ji].tape for ji in act]
            X_list = [U[rows_of[ji]] for ji in act]
            h_list = [(jobs[ji].members, h[t, jobs[ji].members]) for ji in act]
            nk = key(seed, t, 'cdma_noise', jobs[act[0]].salt)   # src/ref/rep 이 같은 칩 잡음을 공유(짝 비교), alt 만 다름
            out, ld, dg = cs[name].transmit(shard_ids, X_list, h_list, D, nk, dev)
            for ji in act:
                s = jobs[ji].tape; W[ji] += out[s]; ld[s].compute_calls = len(jobs[ji].members); leds[ji].add(ld[s])
                for kk in ('own_energy', 'leak_energy', 'noise_energy'):
                    diags[ji][kk] += dg[s][kk]
                diags[ji]['rounds'] += 1
        if log_every and log and (t + 1) % log_every == 0:
            log(f'  round {t + 1}/{t_hi} pairs={len(pairs)}')
    assert torch.isfinite(W).all()
    return W, leds, diags
