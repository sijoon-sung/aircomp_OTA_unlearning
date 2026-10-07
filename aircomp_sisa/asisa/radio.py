"""AirComp 물리 계층 (실수 기저대역 등가, 위상은 송신 전에 보상).

한 shard 의 한 라운드
  member i 는 clip 된 update x_i (||x_i|| <= C) 를 s_i = x_i / (|h_i| beta) 로 보낸다.
  -> 서버에 도착하는 진폭은 x_i / beta (모든 member 가 같은 크기로 도착: 채널 역보상).
  서버는 받은 합에 alpha = beta / n 을 곱한다:  r = (1/n) sum_i x_i + alpha z.
  1회 집계 오차 MSE1 = D alpha^2 sigma2 / L  (L = 코드 길이, 직교 블록이면 1).

전력 정렬 (align)
  beta_min = C / (min_i|h_i| sqrt(P_eff D))  : 가장 약한 member 가 전력 상한에 닿는 크기 (더 작게는 못 함)
  maxpow : beta = beta_min. 잡음이 가장 작다.
  common : beta = beta0 = n_nom sqrt(target L / (D sigma2)). 모든 shard 가 같은 공개 상수로 도착하고,
           명목 인원 n_nom 인 shard 의 집계 오차가 target 이 된다. beta_min > beta0 인 shard 만 beta_min 을 쓴다.
           n_nom 을 비우면 그 shard 의 인원을 쓴다 (= 집계 오차를 target 에 맞추는 '전력 낮춤').
반복: 전력 상한으로 정한 beta_min 의 MSE1 이 eps 를 넘으면 R = ceil(MSE1 / eps) 번 보내 평균한다.

다중화 (mux)
  orth  : shard 마다 직교 자원 블록 하나. shard 사이 간섭 없음. blocks = b 이면 블록 b 개를 동시에 써서
          시간이 1/b, 총 전력 P 를 나누므로 블록당 전력 P/b.
  code  : 모든 shard 가 같은 자원에 동시에 보낸다. shard k 는 길이 L 의 ±1 코드 c_k 로 심볼을 펼치고,
          서버는 c_k 로 역확산해 shard 합을 분리한다. client 마다 칩 타이밍 오차 delta_i ~ U[0, delay_max] (정적, 칩 단위 cyclic prefix 가정):
              c_i^eff[l] = (1 - delta_i) c_k[l] + delta_i c_k[l-1]
              r_k = sum_{i in S_k} g_ik x_i / n_k + sum_{j != k} (beta_k / beta_j) sum_{i in S_j} g_ik x_i / n_k + alpha_k (1/L) c_k . Z
              g_ik = (1/L) c_i^eff . c_k
          가운데 항이 shard 간 간섭이고, beta_k / beta_j 가 shard 사이 near-far 비다.
          반복이 있으면 한 라운드에 max_k R_k 개 슬롯을 쓰고 슬롯 r 에는 R_k > r 인 shard 만 보낸다.
          shard k 는 자기 R_k 슬롯을 평균하므로 shard j 의 간섭은 min(R_j, R_k)/R_k 배가 된다.
  ideal : 채널 없이 평균에 제곱합 기대값 noise_mse 인 잡음만 더한다 (허용 잡음 측정용).

코드 분할의 코드 (code)
  walsh : Hadamard 행. 시간 오차가 없으면 완전 직교.
  pn    : 무작위 ±1. 상호상관 ~ 1/sqrt(L).
  zcz   : 길이 L 의 walsh 칩 사이마다 0 칩을 gap 개 넣는다 (칩 수 L(gap+1)). 수신기는 0..gap 칩 늦게 온 신호를 모두 모으는
          창(window) 템플릿 t_k = sum_{m=0}^{gap} roll(c_k, m) 으로 역확산한다. 시간 오차가 gap 칩 미만이면
          다른 shard 와의 상관이 정확히 0 (영 상관 구간, zero correlation zone) 이고 자기 신호는 그대로 모인다.
          대가: 칩 수 (gap+1) 배, 0 칩 구간의 잡음도 모이므로 잡음 (gap+1) 배.
  일반식: 수신 템플릿 t, 송신 코드 c 일 때 g_ik = t_k . c_i^eff / (t_k . c_k), 잡음 분산 = sigma2 |t_k|^2 / (t_k . c_k)^2.
  칩 타이밍 오차 delta = m + f (정수 m, 소수 f):  c^eff = (1 - f) roll(c, m) + f roll(c, m + 1).
전력 정렬 global (orth): 같은 system 의 모든 shard 가 system 전체 최약 member 의 한계 beta 로 보낸다 (기존 AirComp 관행을 shard 마다 그대로 쓴 경우).
  삭제 대상이 그 최약 member 이면 '처음부터 없었다면' 의 세계에서 모든 shard 의 전력과 잡음이 달라진다.
스케줄링 sched=('global', q): system 전체 채널의 하위 q 분위수 미만 client 는 송신하지 않는다. 한 명이 빠지면 문턱이 바뀌어 다른 shard 의 참여자가 바뀔 수 있다.
전력 정렬 weakest (code 전용): 같은 자원을 쓰는 shard 들이 모두 가장 약한 shard 의 최대 크기 beta = max_k beta_min_k 로 도착한다.
  도착 크기가 같아지는 선택 중 잡음이 가장 작지만, 각 shard 의 전력이 다른 shard 의 채널에 의존한다.
"""
import math
from dataclasses import dataclass, field
import numpy as np
import torch
from .config import CLIP, P_MAX, EPS
from .util import key

@dataclass
class RadioConfig:
    sigma2: float = 0.01        # 수신 잡음 분산 (실수 심볼·칩당). 명목 SNR = P / sigma2
    eps: float = EPS            # 반복 기준 허용 집계 오차. 0 이면 반복하지 않음
    align: str = 'maxpow'       # 'maxpow' | 'common' | 'weakest'(code) | 'global'(orth: 같은 system 전체의 최약 member 기준)
    sched: tuple = None         # None | ('shard'|'global', q): 이번 라운드 채널이 (shard 안 | system 전체) 하위 q 분위수 미만이면 송신하지 않음
    target: float = None        # common 의 목표 집계 오차 (None 이면 eps)
    n_nom: float = None         # common 의 명목 인원 (None 이면 그 shard 의 인원)
    mux: str = 'orth'           # 'orth' | 'code' | 'ideal'
    code: str = 'walsh'         # mux='code': 'walsh' | 'pn' | 'zcz'
    L: int = 4                  # mux='code': 코드 길이 (zcz 는 바탕 walsh 길이)
    gap: int = 1                # code='zcz': walsh 칩 사이에 넣는 0 칩 수 (견디는 시간 오차 = gap 칩 미만)
    delay_max: float = 0.0      # mux='code': 칩 타이밍 오차 상한
    blocks: float = 1.0         # mux='orth': 동시에 쓰는 자원 블록 수
    noise_mse: float = 0.0      # mux='ideal': 더하는 잡음의 제곱합 기대값
    P: float = P_MAX
    C: float = CLIP

@dataclass
class Ledger:
    """자원 장부. 시간 단위는 블록 하나에서 실수 심볼 하나를 보내는 시간."""
    ul_symbols: float = 0.0     # 업링크 실수 심볼(칩) 수, 반복 포함
    ul_time: float = 0.0
    dl_bits: float = 0.0        # 모델 broadcast 비트
    energy: float = 0.0         # 송신 에너지 합 sum ||s_i||^2 (반복 포함)
    repeats: float = 0.0
    rounds: int = 0
    tx_count: int = 0           # client-라운드 송신 횟수
    compute_calls: int = 0      # 로컬 학습 횟수
    mse_sum: float = 0.0        # 달성한 집계 오차의 합
    max_power_ratio: float = 0.0
    def add(self, o):
        for k in self.__dataclass_fields__:
            v = max(self.max_power_ratio, o.max_power_ratio) if k == 'max_power_ratio' else getattr(self, k) + getattr(o, k)
            setattr(self, k, v)
        return self
    def asdict(self):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d['mean_mse'] = self.mse_sum / max(1, self.rounds)
        return d

def new_diag():
    return dict(own_energy=0.0, leak_energy=0.0, noise_energy=0.0, alpha=0.0, beta=0.0, repeats=0.0, slots=0.0,
                leak_by_src={}, u_leak={})

def plan(cfg, n, hmin, D, L=1, beta_force=None):
    """shard 하나의 (beta, 반복 R). 수신 scale 은 alpha = beta / n. L 은 처리 이득 (직교 블록 1, 코드 분할은 코드에 따라).
    beta_force: align='weakest' 에서 system 이 정한 공통 beta."""
    P_eff = cfg.P / cfg.blocks
    beta_min = cfg.C / (hmin * math.sqrt(P_eff * D))
    if cfg.align == 'common':
        target = cfg.eps if cfg.target is None else cfg.target
        beta0 = (cfg.n_nom or n) * math.sqrt(target * L / (D * cfg.sigma2))
        if beta_min <= beta0:
            return beta0, 1
    elif cfg.align in ('weakest', 'global'):
        beta_min = max(beta_min, beta_force)
    elif cfg.align != 'maxpow':
        raise ValueError(cfg.align)
    mse1 = D * (beta_min / n) ** 2 * cfg.sigma2 / L
    return beta_min, (max(1, math.ceil(mse1 / cfg.eps - 1e-9)) if cfg.eps > 0 else 1)

def _gaussian(noise_key, shape, device, dtype, std):
    g = torch.Generator(device=device).manual_seed(int(noise_key))
    return torch.randn(*shape, generator=g, device=device, dtype=dtype) * std

def transmit_orth(cfg, X, hs, noise_key, beta_force=None):
    """직교 블록 하나로 shard 평균을 받는다. X [n, D], hs [n]. beta_force: align='global' 에서 system 이 정한 beta.
    반환: 추정 [D], Ledger, 진단."""
    n, D = X.shape
    beta, R = plan(cfg, n, float(np.min(hs)), D, 1, beta_force)
    alpha = beta / n
    h = torch.as_tensor(hs, dtype=X.dtype, device=X.device)
    S = X / (h[:, None] * beta)
    pr = float(((S * S).mean(1) / (cfg.P / cfg.blocks)).max()); assert pr <= 1 + 1e-6, f'전력 상한 위반 {pr}'
    own = (h[:, None] * S).sum(0)
    z = _gaussian(noise_key, (R, D), X.device, X.dtype, math.sqrt(cfg.sigma2)).mean(0)
    d = new_diag()
    d.update(own_energy=float((alpha * own).pow(2).sum()), noise_energy=float((alpha * z).pow(2).sum()), alpha=alpha, beta=beta,
             repeats=float(R), slots=float(R))
    led = Ledger(ul_symbols=R * D, ul_time=R * D / cfg.blocks, dl_bits=32 * D, energy=float((S * S).sum()) * R, repeats=R, rounds=1,
                 tx_count=n, mse_sum=D * alpha * alpha * cfg.sigma2 / R, max_power_ratio=pr)
    return alpha * (own + z), led, d

def transmit_ideal(cfg, X, noise_key):
    n, D = X.shape
    r = X.mean(0)
    if cfg.noise_mse > 0:
        r = r + _gaussian(noise_key, (D,), X.device, X.dtype, math.sqrt(cfg.noise_mse / D))
    d = new_diag(); d['own_energy'] = float(X.mean(0).pow(2).sum())
    return r, Ledger(rounds=1, tx_count=n, mse_sum=cfg.noise_mse), d

def _walsh(L, K):
    L = 1 << max(0, math.ceil(math.log2(max(L, K))))
    H = np.array([[1.0]])
    while H.shape[0] < L:
        H = np.block([[H, H], [H, -H]])
    return H[:K].copy()

def make_codes(cfg, K, seed):
    """반환: (송신 코드 [K, 칩 수], 수신 템플릿 [K, 칩 수])."""
    if cfg.code == 'walsh':
        c = _walsh(cfg.L, K); return c, c
    if cfg.code == 'pn':
        c = np.stack([np.random.default_rng(key(seed, 'pn', k, cfg.L)).choice([-1.0, 1.0], cfg.L) for k in range(K)]); return c, c
    if cfg.code == 'zcz':
        w = _walsh(cfg.L, K); g = cfg.gap
        c = np.zeros((K, w.shape[1] * (g + 1))); c[:, ::g + 1] = w
        return c, sum(np.roll(c, m, axis=1) for m in range(g + 1))
    raise ValueError(cfg.code)

class CodeSystem:
    """같은 자원에 동시에 보내는 shard 묶음 하나 (mux='code')."""
    def __init__(self, cfg, n_codes, seed, delays):
        self.cfg = cfg; self.codes, self.tmpl = make_codes(cfg, n_codes, seed); self.delays = delays
        self.chips = self.codes.shape[1]
        self.norm = float(self.tmpl[0] @ self.codes[0])                       # t_k . c_k (모든 코드에서 같음)
        self.pgain = self.norm ** 2 / float(self.tmpl[0] @ self.tmpl[0])     # 처리 이득: 잡음 분산 = sigma2 / pgain
        self.cenergy = float(self.codes[0] @ self.codes[0])                  # 심볼 하나를 펼친 칩 에너지 배수
        self._g = {}; self._gt = {}

    def gain(self, i, s_tx, s_rx):
        """client i (코드 s_tx) 의 신호가 템플릿 s_rx 로 역확산될 때의 이득 g."""
        k = (i, s_tx, s_rx)
        if k not in self._g:
            c = self.codes[s_tx]; d = self.delays[i]; m = int(math.floor(d)); f = d - m
            self._g[k] = float(self.tmpl[s_rx] @ ((1 - f) * np.roll(c, m) + f * np.roll(c, m + 1)) / self.norm)
        return self._g[k]

    def _gains(self, ids, s_tx, s_rx, dtype, device):
        k = (tuple(ids), s_tx, s_rx)
        if k not in self._gt:
            self._gt[k] = torch.as_tensor([self.gain(i, s_tx, s_rx) for i in ids], dtype=dtype, device=device)
        return self._gt[k]

    def transmit(self, shards, noise_key, track=()):
        """shards: [(코드 번호, X [n, D], client id 목록, 채널 진폭)]. track: 다른 shard 로 새는 양을 따로 잴 client.
        반환: {코드: 추정 [D]}, {코드: Ledger}, {코드: 진단}."""
        cfg = self.cfg
        D = shards[0][1].shape[1]; dev = shards[0][1].device; dt = shards[0][1].dtype
        bf = max(cfg.C / (float(np.min(hs)) * math.sqrt(cfg.P * D)) for _, _, _, hs in shards) if cfg.align in ('weakest', 'global') else None
        plans = {s: plan(cfg, X.shape[0], float(np.min(hs)), D, self.pgain, bf) for s, X, ids, hs in shards}
        Rmax = max(R for _, R in plans.values())
        Z = [_gaussian(noise_key if r == 0 else key(noise_key, 'slot', r), (self.chips, D), dev, dt, math.sqrt(cfg.sigma2)) for r in range(Rmax)]
        tmpl = torch.as_tensor(self.tmpl, dtype=dt, device=dev)
        arrive, info = {}, {}
        for s, X, ids, hs in shards:
            beta, R = plans[s]
            h = torch.as_tensor(hs, dtype=dt, device=dev)
            S = X / (h[:, None] * beta)
            pr = float(((S * S).mean(1) / cfg.P).max()); assert pr <= 1 + 1e-6, f'전력 상한 위반 {pr}'
            arrive[s] = h[:, None] * S                                   # 도착 진폭 x_i / beta
            info[s] = (list(ids), X.shape[0], pr, float((S * S).sum()) * self.cenergy * R)
        out, leds, diags = {}, {}, {}
        for s, X, ids, hs in shards:
            beta, R = plans[s]; n = X.shape[0]; alpha = beta / n
            own = torch.zeros(D, device=dev, dtype=dt); leak = torch.zeros(D, device=dev, dtype=dt)
            d = new_diag()
            for s2, A in arrive.items():
                ids2 = info[s2][0]; g = self._gains(ids2, s2, s, dt, dev)
                term = (g[:, None] * A).sum(0)
                if s2 == s:
                    own += term
                    continue
                w = min(plans[s2][1], R) / R
                leak += w * term
                d['leak_by_src'][s2] = float((alpha * w * term).pow(2).sum())
                for u in track:
                    if u in ids2:
                        q = ids2.index(u); d['u_leak'][u] = float((alpha * w * g[q] * A[q]).pow(2).sum())
            noise = sum((tmpl[s][:, None] * Z[r]).sum(0) for r in range(R)) / (self.norm * R)
            out[s] = alpha * (own + leak + noise)
            d.update(own_energy=float((alpha * own).pow(2).sum()), leak_energy=float((alpha * leak).pow(2).sum()),
                     noise_energy=float((alpha * noise).pow(2).sum()), alpha=alpha, beta=beta, repeats=float(R), slots=float(Rmax))
            diags[s] = d
            leds[s] = Ledger(ul_symbols=self.chips * D * Rmax, ul_time=self.chips * D * Rmax, dl_bits=32 * D, energy=info[s][3], repeats=R, rounds=1,
                             tx_count=n, mse_sum=D * alpha * alpha * cfg.sigma2 / (self.pgain * R), max_power_ratio=info[s][2])
        return out, leds, diags

def digital_ledger(cfg, hs, D):
    """비교용: client 마다 직교 자원으로 32 bit x D 를 Shannon 용량 log2(1 + P|h|^2/sigma2) 로 보낼 때."""
    led = Ledger(rounds=1, tx_count=len(hs), dl_bits=32 * D)
    for h in hs:
        uses = 32 * D / math.log2(1 + cfg.P * h * h / cfg.sigma2)
        led.ul_symbols += uses; led.ul_time += uses; led.energy += cfg.P * uses
    return led
