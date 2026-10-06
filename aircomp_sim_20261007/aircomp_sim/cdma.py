"""코드 분할 다중 shard AirComp: 모든 shard 가 같은 자원에 동시에 보내고, 서버가 shard 코드로 합을 분리한다.

송신 (shard k 의 client i, 라운드 t)
  shard 마다 길이 L 의 ±1 코드 c_k 를 둔다. client i 는 업데이트 심볼 하나를 L 칩으로 펼쳐 보낸다:
      chip_l = b_i * x_i * c_k[l],  l = 0..L-1,   b_i = 1 / (|h_i| beta_k)
  beta_k 는 shard k 의 member 가 서버에 도착하는 크기를 정한다 (수신 진폭 = x_i / beta_k, shard 안 전력 정렬).
  칩당 평균 전력 <= P 이므로 beta_k >= beta_min_k = C / (min_{S_k}|h| sqrt(PD)).
  client 마다 칩 단위 타이밍 오차 delta_i ~ U[0, delta_max] (정적). 사각 칩 펄스에서 지연된 코드는
      c_i^eff[l] = (1 - delta_i) c_k[l] + delta_i c_k[l-1]    (순환)
채널: Y[l] = sum_i |h_i| * chip_l(i) + Z[l],  Z[l] ~ N(0, sigma2)   (모든 shard 의 client 가 한 신호로 더해짐)
수신 (shard k): 코드 c_k 로 역확산 후 scale alpha_k = beta_k / n_k
      r_k = alpha_k * (1/L) sum_l c_k[l] Y[l]
         = sum_{i in S_k} g_{ik} x_i / n_k                                  (자기 shard, 타이밍 오차가 없으면 g=1 -> 평균)
         + sum_{j != k} (beta_k / beta_j) sum_{i in S_j} g_{ik} x_i / n_k      (다른 shard 에서 새는 간섭)
         + alpha_k * (1/L) c_k . Z                                             (잡음, 분산 alpha_k^2 sigma2 / L)
      g_{ik} = (1/L) sum_l c_i^eff[l] c_k[l]
  간섭에 곱해지는 beta_k / beta_j 가 shard 사이의 near-far 비다.

전력 정렬 규칙 (align)
  maxpow : beta_k = beta_min_k. 최약 member 가 전력 상한에 닿는다. 기존 다중 그룹 AirComp 와 같고, 이전 P0-A 의 설정.
           beta_k / beta_j = min_{S_j}|h| / min_{S_k}|h| -> 채널이 약한 shard 일수록 다른 shard 의 신호를 크게 받는다.
  common : 모든 shard 가 공통 공개 상수 beta0 로 도착한다. beta0 = n_nom sqrt(target L / (D sigma2))
           (명목 인원 n_nom 인 shard 의 평균 추정 오차가 target 이 되는 크기). beta_k / beta_j = 1.
           전력 상한 때문에 beta0 를 못 맞추는 shard (beta_min_k > beta0) 만 beta_min_k 를 쓴다.
  beta0 는 다른 shard 의 채널이나 인원에 의존하지 않는다. member 의 수신 크기도 자기 shard 인원과 무관하다.
반복 (eps > 0 일 때)
  전력 상한으로 정한 beta_min_k 의 1회 집계 오차 MSE1 = D (beta_min_k / n_k)^2 sigma2 / L 이 eps 보다 크면
  R_k = ceil(MSE1 / eps) 번 반복한다. 한 라운드는 R_max = max_k R_k 개 슬롯을 쓰고, 슬롯 r 에는 R_k > r 인 shard 만 보낸다.
  슬롯마다 칩 잡음이 새로 생기고, shard k 는 자기 R_k 개 슬롯을 평균한다. 그래서 shard j 의 간섭은 min(R_j, R_k) / R_k 배로 준다.
코드 종류: walsh (Hadamard 행, 동기면 완전 직교) / pn (무작위 ±1, 상호상관 ~ 1/sqrt(L)).
측정용으로 자기 항, 간섭 항(보낸 shard 별), 추적 client 한 명이 새는 양, 잡음 항의 에너지와 beta, 반복 수를 장부에 남긴다.
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
    align: str = 'maxpow'    # 'maxpow' | 'common'
    target: float = 10.0     # align='common': 명목 인원 shard 의 평균 추정 목표 오차
    n_nom: float = 0.0       # align='common': beta0 를 정하는 명목 shard 인원 (공개 상수). 0 이면 그 shard 의 인원
    eps: float = 0.0         # 허용 집계 오차. > 0 이면 전력 상한 때문에 이를 넘는 shard 는 반복 전송 (0 이면 반복 없음)

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
        self._gc = {}; self._gt = {}

    def plan(self, n, hmin, D):
        """shard 하나의 member 수신 크기 beta 와 반복 수 R. 수신 scale 은 alpha = beta / n."""
        cfg = self.cfg; L = self.L
        beta_min = cfg.C / (hmin * math.sqrt(cfg.P * D))          # 최약 member 가 칩당 전력 상한 P 에 닿는 크기
        if cfg.align == 'common':
            nn = cfg.n_nom if cfg.n_nom > 0 else n
            beta0 = nn * math.sqrt(cfg.target * L / (D * cfg.sigma2))
            if beta_min <= beta0:
                return beta0, 1
        elif cfg.align != 'maxpow':
            raise ValueError(cfg.align)
        mse1 = D * (beta_min / n) ** 2 * cfg.sigma2 / L
        R = max(1, math.ceil(mse1 / cfg.eps - 1e-9)) if cfg.eps > 0 else 1
        return beta_min, R

    def transmit(self, shard_ids, X_list, h_list, D, noise_key, device, track=()):
        """shard_ids: 이번 라운드 동시에 보내는 shard 번호들, X_list[s]: [n_s, D] clip 된 업데이트, h_list[s]: (client id 목록, 채널 진폭).
        track: 다른 shard 로 새는 양을 따로 잴 client id 들.
        반환: {shard: 수신 추정 [D]}, {shard: Ledger}, 진단 dict."""
        cfg = self.cfg; L = self.L
        plans = {s: self.plan(X.shape[0], float(np.min(hs)), D) for s, X, (ids, hs) in zip(shard_ids, X_list, h_list)}
        Rmax = max(R for _, R in plans.values())
        # 칩 잡음은 슬롯마다 실제로 생성한다. 슬롯 0 은 noise_key 그대로 (반복이 없으면 이전 P0-A 와 같은 잡음)
        Zs = []
        for r in range(Rmax):
            g = torch.Generator(device=device).manual_seed(int(noise_key if r == 0 else key(noise_key, 'slot', r)))
            Zs.append(torch.randn(L, D, generator=g, device=device) * math.sqrt(cfg.sigma2))
        dt = Zs[0].dtype
        codes_t = torch.as_tensor(self.codes, dtype=dt, device=device)
        # 각 client 의 수신 진폭 |h_i| b_i x_i (= x_i / beta)
        contrib, meta = {}, {}
        for s, X, (ids, hs) in zip(shard_ids, X_list, h_list):
            beta, R = plans[s]
            h = torch.as_tensor(hs, dtype=X.dtype, device=device)
            b = 1.0 / (h * beta); S = X * b[:, None]
            pr = float(((S * S).mean(1) / cfg.P).max()); assert pr <= 1 + 1e-6, pr
            contrib[s] = h[:, None] * S
            meta[s] = (list(ids), X.shape[0], pr, float((S * S).sum()) * L * R)
        out, leds, diag = {}, {}, {}
        for s in shard_ids:
            beta, R = plans[s]; ids, n, pr, energy = meta[s]; alpha = beta / n
            own = torch.zeros(D, device=device, dtype=dt); leak = torch.zeros(D, device=device, dtype=dt)
            leak_by, uleak = {}, {}
            for s2 in shard_ids:
                ids2 = meta[s2][0]; A = contrib[s2]; gk = self._gvec(ids2, s2, s, dt, device)
                term = (gk[:, None] * A).sum(0)
                if s2 == s:
                    own += term
                    continue
                w = min(plans[s2][1], R) / R
                leak += w * term
                leak_by[s2] = float((alpha * w * term).pow(2).sum())
                for u in track:
                    if u in ids2:
                        q = ids2.index(u)
                        uleak[u] = float((alpha * w * gk[q] * A[q]).pow(2).sum())
            noise = torch.zeros(D, device=device, dtype=dt)
            for r in range(R):
                noise += (codes_t[s][:, None] * Zs[r]).sum(0)
            noise /= L * R
            out[s] = alpha * (own + leak + noise)
            ownv = alpha * own; leakv = alpha * leak
            diag[s] = dict(own_energy=float((ownv * ownv).sum()), leak_energy=float((leakv * leakv).sum()),
                           noise_energy=float((alpha * noise).pow(2).sum()), alpha=alpha, beta=beta, repeats=float(R), slots=float(Rmax),
                           leak_by_src=leak_by, u_leak=uleak)
            leds[s] = Ledger(ul_symbols=L * D * Rmax, ul_time=L * D * Rmax, dl_bits=32 * D, energy=energy, repeats=R, rounds=1, tx_count=len(ids),
                             mse_sum=float(D * alpha ** 2 * cfg.sigma2 / (L * R)), max_power_ratio=pr)
        return out, leds, diag

    def _g(self, i, s_tx, s_rx):
        k = (i, s_tx, s_rx)
        if k not in self._gc:
            c = self.codes[s_tx]; d = self.delays[i]; ce = (1 - d) * c + d * np.roll(c, 1)
            self._gc[k] = float(self.codes[s_rx] @ ce / self.L)
        return self._gc[k]

    def _gvec(self, ids, s_tx, s_rx, dt, device):
        k = (tuple(ids), s_tx, s_rx)
        if k not in self._gt:
            self._gt[k] = torch.as_tensor([self._g(i, s_tx, s_rx) for i in ids], dtype=dt, device=device)
        return self._gt[k]

DIAG_SUM = ('own_energy', 'leak_energy', 'noise_energy', 'alpha', 'beta', 'repeats', 'slots')
DIAG_DICT = ('leak_by_src', 'u_leak')

def run_rounds_cdma(tr, jobs, h, seed, cdma_cfg_of, delays, log=None, log_every=0, track_of=None):
    """jobs: fl.ShardJob 목록. 같은 system 의 job 들은 같은 자원에 동시에 보내며 코드로 분리된다.
    job.tape 를 shard 코드 번호로 쓴다. cdma_cfg_of(system) -> CdmaConfig. track_of(system) -> 간섭을 따로 잴 client id 들.
    반환: W, ledgers, diagnostics(job 별 라운드 합. 진단 dict 의 값은 라운드 수로 나누면 평균)."""
    dev = tr.device; D = tr.D
    W = torch.stack([j.w0.to(dev).clone() for j in jobs])
    leds = [Ledger() for _ in jobs]
    diags = [dict({kk: 0.0 for kk in DIAG_SUM}, rounds=0, **{kk: {} for kk in DIAG_DICT}) for _ in jobs]
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
            out, ld, dg = cs[name].transmit(shard_ids, X_list, h_list, D, nk, dev, track_of(name) if track_of else ())
            for ji in act:
                s = jobs[ji].tape; W[ji] += out[s]; ld[s].compute_calls = len(jobs[ji].members); leds[ji].add(ld[s])
                d = diags[ji]
                for kk in DIAG_SUM:
                    d[kk] += dg[s][kk]
                for kk in DIAG_DICT:
                    for a, v in dg[s][kk].items():
                        d[kk][a] = d[kk].get(a, 0.0) + v
                d['rounds'] += 1
        if log_every and log and (t + 1) % log_every == 0:
            log(f'  round {t + 1}/{t_hi} pairs={len(pairs)}')
    assert torch.isfinite(W).all()
    return W, leds, diags
