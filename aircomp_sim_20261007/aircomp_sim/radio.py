"""라운드 단위 AirComp 무선 시뮬레이터 (실수 기저대역 등가 모델).

매 라운드, 매 shard 전송마다 다음이 실제로 실행된다.
  1. 전력 제어: 송신 계수 b_i 와 수신 scale alpha 를 계산한다.
  2. 송신: 각 client 가 s_i = b_i * x_i 를 보낸다 (심볼 D 개, 필요하면 R 번 반복).
  3. 채널: y = sum_i |h_i| s_i + z,  z ~ N(0, sigma2) per real symbol (위상은 송신 전에 보상됐다고 가정).
  4. 수신: r = alpha * y  ->  (1/n) sum_i x_i + alpha z.  반복 R 번이면 평균.
  5. 장부: 심볼 수, client 별 송신 에너지(실제 ||s_i||^2), 반복 수, 집계 오차.

전력 정렬 (채널 역보상)
  b_i = 1 / (n * |h_i| * alpha),  alpha >= alpha_phys = C / (n * min_i|h_i| * sqrt(P D))
  alpha = alpha_phys 이면 가장 약한 client 가 전력 상한 P 에 닿는다 (공개 상한 C 기준).
  집계 오차(1회) MSE1 = D alpha^2 sigma2.
전송 규칙
  maxpow : alpha = alpha_phys, MSE1 > eps 이면 R = ceil(MSE1/eps) 번 반복
  minen  : MSE1 <= eps 이면 alpha 를 키워(전력을 낮춰) MSE = eps 로 맞춘다. 아니면 maxpow 와 같다
무선 제어 공유 (문제 상황 재현용)
  align='shard'  : alpha 를 shard 안의 최약 채널로 계산 (무선 격리)
  align='shared' : 모든 shard 가 전체 최약 채널 기준 공통 alpha 를 쓴다
  sched=None     : 전원 송신
  sched=('shard'|'shared', q) : 이번 라운드 채널이 (shard 안 | 전체) 하위 q 분위수 미만인 client 는 송신하지 않는다
자원
  shard 마다 직교 자원 블록 1개. 삭제 재학습 중 reuse_idle=True 이면 비어 있는 블록을 재학습 shard 가 모두 써서
  R 번 반복이 병렬로 나가고(지연 1/b), 총 전력 제약(power='total')이면 심볼당 전력이 P/b 로 준다.
디지털 기준선
  client 마다 직교 자원으로 32 bit x D 를 보낸다. 사용량 = 32 D / log2(1 + P|h_i|^2/sigma2) real uses, 에너지 = P x 사용량.
"""
import math
from dataclasses import dataclass, field
import numpy as np
import torch

@dataclass
class Ledger:
    ul_symbols: float = 0.0      # 업링크 아날로그 real 심볼 수 (반복 포함)
    ul_time: float = 0.0         # 블록 1개 기준 정규화 시간 (병렬 사용 시 1/b)
    dl_bits: float = 0.0         # 다운링크 모델 broadcast 비트
    energy: float = 0.0          # 송신 에너지 합 (정규화 단위, 실제 ||s_i||^2 x R)
    repeats: float = 0.0
    rounds: int = 0
    tx_count: int = 0            # client-round 송신 횟수
    compute_calls: int = 0       # 로컬 학습 호출 수
    mse_sum: float = 0.0         # 달성한 집계 오차(벡터 제곱합) 합
    max_power_ratio: float = 0.0 # 심볼당 전력 / P 의 최댓값 (<=1 이어야 함)
    def add(self, o):
        for k in self.__dataclass_fields__:
            if k == 'max_power_ratio':
                self.max_power_ratio = max(self.max_power_ratio, o.max_power_ratio)
            else:
                setattr(self, k, getattr(self, k) + getattr(o, k))
        return self
    def asdict(self):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d['mean_mse'] = self.mse_sum / max(1, self.rounds)
        return d

@dataclass
class RadioConfig:
    sigma2: float = 0.01      # 수신 잡음 분산 (real 심볼당). 명목 SNR = P/sigma2
    P: float = 1.0            # 심볼당 평균 전력 상한
    C: float = 1.0            # 공개 update 크기 상한 (clipping)
    eps: float = 10.0         # 허용 집계 오차 (벡터 제곱합)
    mode: str = 'maxpow'      # 'maxpow' | 'minen'
    align: str = 'shard'      # 'shard' | 'shared'
    sched: tuple = None       # None | ('shard'|'shared', q)
    power: str = 'symbol'     # 'symbol' | 'total'  (대역을 b 배 쓸 때 심볼당 전력 유지 / 총전력 유지)
    dl_bits_per_param: int = 32

class Radio:
    def __init__(self, cfg: RadioConfig, D: int, seed: int, device: str):
        self.cfg, self.D, self.seed, self.device = cfg, D, seed, device
        self._noise_key = 0

    # ---------------- 전력 제어
    def alpha_phys(self, n, hmin):
        return self.cfg.C / (n * hmin * math.sqrt(self.cfg.P * self.D))

    def plan(self, n, hmin, bw=1.0):
        """alpha, 반복 R, 심볼당 유효 전력 상한. 반환 dict."""
        P_eff = self.cfg.P / bw if self.cfg.power == 'total' else self.cfg.P
        a0 = self.cfg.C / (n * hmin * math.sqrt(P_eff * self.D))
        mse1 = self.D * a0 * a0 * self.cfg.sigma2
        if self.cfg.mode == 'minen' and mse1 <= self.cfg.eps:
            alpha = math.sqrt(self.cfg.eps / (self.D * self.cfg.sigma2)); R = 1
        else:
            alpha = a0; R = max(1, math.ceil(mse1 / self.cfg.eps - 1e-9))
        return dict(alpha=alpha, R=R, P_eff=P_eff, mse=self.D * alpha * alpha * self.cfg.sigma2 / R)

    # ---------------- 스케줄링
    def active(self, members, h_t, pool_h_t):
        """이번 라운드 송신할 client. pool_h_t: 문턱 계산에 쓰는 채널 집합 (shard 또는 전체)."""
        if self.cfg.sched is None:
            return list(members)
        _, q = self.cfg.sched
        thr = float(np.quantile(pool_h_t, q))
        act = [i for i in members if h_t[i] >= thr]
        return act if act else [max(members, key=lambda i: h_t[i])]

    # ---------------- 한 shard 의 한 라운드 전송
    def transmit(self, X, hs, hmin_align, bw=1.0, noise_key=None):
        """X: [n, D] 송신 벡터(이미 clip), hs: [n] 이번 라운드 채널 진폭, hmin_align: 전력 정렬에 쓰는 최약 채널.
        반환 (수신 평균 추정 [D], Ledger)."""
        n = X.shape[0]; pl = self.plan(n, hmin_align, bw)
        alpha, R, P_eff = pl['alpha'], pl['R'], pl['P_eff']
        h = torch.as_tensor(hs, dtype=X.dtype, device=X.device)
        b = 1.0 / (n * h * alpha)                               # 송신 계수
        S = X * b[:, None]                                      # 송신 심볼 [n, D]
        sym_power = (S * S).mean(1)                             # 심볼당 평균 전력
        pr = float((sym_power / P_eff).max())
        assert pr <= 1.0 + 1e-6, f'power cap violated: {pr}'
        g = torch.Generator(device=X.device).manual_seed(int(noise_key) if noise_key is not None else self._next_key())
        y = (h[:, None] * S).sum(0)                             # 채널 합 (위상 보상 가정)
        z = torch.randn(R, self.D, generator=g, device=X.device, dtype=X.dtype) * math.sqrt(self.cfg.sigma2)
        r = alpha * (y[None, :] + z).mean(0)                    # 수신 scale 후 R 회 평균
        led = Ledger(ul_symbols=R * self.D, ul_time=R * self.D / bw, dl_bits=self.cfg.dl_bits_per_param * self.D,
                     energy=float((S * S).sum()) * R, repeats=R, rounds=1, tx_count=n, mse_sum=pl['mse'], max_power_ratio=pr)
        return r, led

    def _next_key(self):
        self._noise_key += 1
        return (self.seed * 1000003 + self._noise_key) % (2 ** 62)

    # ---------------- 디지털 기준선 장부
    def digital_ledger(self, hs, n_params=None):
        D = n_params or self.D; led = Ledger(rounds=1, tx_count=len(hs), dl_bits=self.cfg.dl_bits_per_param * D)
        for hi in hs:
            cap = math.log2(1 + self.cfg.P * hi * hi / self.cfg.sigma2)
            uses = 32 * D / cap
            led.ul_symbols += uses; led.ul_time += uses; led.energy += self.cfg.P * uses
        return led
