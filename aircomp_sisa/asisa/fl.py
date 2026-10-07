"""shard 단위 AirComp 연합학습의 라운드 루프.

shard 모델 여러 개(학습 경로 하나씩)를 라운드마다 함께 진행한다.
  1. 이번 라운드 송신할 member 를 정한다: 이탈(참여 못 함) -> 스케줄링 (radio.sched) -> 최소 인원 규칙 (Shard.min_present).
  2. 송신할 (shard, member) 쌍의 로컬 update 를 vmap 으로 한 번에 계산한다.
  3. system 마다 무선 전송을 실제로 실행한다. 같은 system 의 shard 들은 같은 라운드에 보내고,
     mux='code' 면 같은 자원을 공유해 서로 간섭한다. mux='orth' 면 각자 블록을 쓴다.
     align='global' (orth) 이면 system 전체의 최약 송신 member 로 전력을 정한다.
  4. 각 shard 모델에 수신한 평균 추정을 더한다. hook 이 있으면 (shard 번호, 라운드, 송신 member, 수신값, member update) 를 넘긴다.

난수 짝 맞춤
  minibatch 는 (seed, client, 라운드, step, salt), 잡음은 (seed, 라운드, salt[, 슬롯 번호]), 이탈은 (seed, client, 라운드) 로 정해진다.
  그래서 salt 가 같은 두 학습 경로(예: u 있음 / u 없이 처음부터)는 같은 배치·잡음·이탈을 쓰고, 차이는 바꾼 조건에서만 나온다.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .radio import Ledger, new_diag, transmit_orth, transmit_ideal, CodeSystem
from .util import key
from .config import N_CLIENTS

@dataclass
class Shard:
    name: str
    members: list
    system: str               # 같은 system 의 shard 는 함께 전송한다
    slot: int = 0             # code: 코드 번호 / orth·ideal: 잡음 키에 쓰는 번호
    salt: str = ''
    T: int = 160
    save_at: tuple = ()       # 이 라운드 시작 직전 모델을 저장
    w0: torch.Tensor = None   # 시작 모델 (None 이면 train 의 w0)
    drop: float = 0.0         # 라운드마다 client 가 참여하지 못할 확률 (client·라운드로 정해짐, 같은 seed 면 모든 경로에서 같음)
    absent: dict = None       # {라운드: 그 라운드에 빠지는 client 집합} (지정 이탈)
    min_present: int = 0      # 송신 인원이 이보다 적으면 그 라운드는 보내지 않는다 (shard 자기 정보만 쓰는 규칙)
    noise_salt: str = None    # 잡음 난수만 따로 바꿀 때 (None 이면 salt). 배치는 같고 채널 잡음은 독립인 두 전송을 만들 때 쓴다

DIAG_SUM = ('own_energy', 'leak_energy', 'noise_energy', 'alpha', 'beta', 'repeats', 'slots')

def chip_delays(seed, delay_max):
    return delay_max * np.array([np.random.default_rng(key(seed, 'chipdelay', i)).uniform() for i in range(N_CLIENTS)])

def nsalt(s):
    return s.salt if s.noise_salt is None else s.noise_salt

def is_absent(seed, i, t, p):
    return p > 0 and np.random.default_rng(key(seed, 'avail', i, t)).random() < p

def train(tr, shards, h, w0, seed, radio_of, track_of=None, log=None, log_every=0, hook=None):
    """shards: Shard 목록, h: [T, N] 채널, radio_of(system) -> RadioConfig, track_of(system) -> 간섭을 따로 잴 client 들.
    반환: W [J, D] 최종 모델, Ledger 목록, 진단 목록(라운드 합; rounds 로 나누면 평균, skipped = 보내지 않은 라운드 수), 체크포인트 목록."""
    dev = tr.device
    W = torch.stack([(s.w0 if s.w0 is not None else w0).to(dev).clone() for s in shards])
    leds = [Ledger() for _ in shards]; diags = [dict(new_diag(), rounds=0, skipped=0, sent_sum=0) for _ in shards]; ckpts = [dict() for _ in shards]
    systems = {}
    for j, s in enumerate(shards):
        systems.setdefault(s.system, []).append(j)
    code_sys = {}
    for name, idx in systems.items():
        cfg = radio_of(name)
        if cfg.mux == 'code':
            code_sys[name] = CodeSystem(cfg, 1 + max(shards[j].slot for j in idx), seed, chip_delays(seed, cfg.delay_max))
    T_all = max(s.T for s in shards)
    for t in range(T_all):
        live = [j for j, s in enumerate(shards) if t < s.T]
        for j in live:
            if t in shards[j].save_at:
                ckpts[j][t] = W[j].clone()
        present = {j: [i for i in shards[j].members if not is_absent(seed, i, t, shards[j].drop)
                       and not (shards[j].absent and i in shards[j].absent.get(t, ()))] for j in live}
        act = dict(present)
        for name, idx in systems.items():
            cfg = radio_of(name); lj = [j for j in idx if j in present]
            if cfg.sched is not None and lj:
                scope, q = cfg.sched
                pool_all = [i for j in lj for i in present[j]]
                for j in lj:
                    pool = present[j] if scope == 'shard' else pool_all
                    if not pool or not present[j]:
                        continue
                    thr = float(np.quantile(h[t, pool], q))
                    a = [i for i in present[j] if h[t, i] >= thr]
                    act[j] = a if a else [max(present[j], key=lambda i: h[t, i])]
        for j in live:
            if len(act[j]) < max(1, shards[j].min_present):
                if act[j] or shards[j].members:
                    diags[j]['skipped'] += 1
                act[j] = []
        pairs = [(j, i) for j in live for i in act[j]]
        if not pairs:
            continue
        U = tr.updates(W[torch.tensor([p[0] for p in pairs], device=dev)], [p[1] for p in pairs], t, [shards[p[0]].salt for p in pairs])
        rows = {}
        for k, (j, _) in enumerate(pairs):
            rows.setdefault(j, []).append(k)
        for name, idx in systems.items():
            ja = [j for j in idx if j in rows]
            if not ja:
                continue
            cfg = radio_of(name); res = {}
            if cfg.mux == 'code':
                out, ld, dg = code_sys[name].transmit(
                    [(shards[j].slot, U[rows[j]], act[j], h[t, act[j]]) for j in ja],
                    key(seed, t, 'noise', nsalt(shards[ja[0]])), track_of(name) if track_of else ())
                res = {j: (out[shards[j].slot], ld[shards[j].slot], dg[shards[j].slot]) for j in ja}
            else:
                bf = None
                if cfg.mux == 'orth' and cfg.align == 'global':
                    hg = float(min(h[t, i] for j in ja for i in act[j]))
                    bf = cfg.C / (hg * np.sqrt(cfg.P / cfg.blocks * tr.D))
                for j in ja:
                    nk = key(seed, t, 'noise', nsalt(shards[j]), shards[j].slot)
                    res[j] = transmit_ideal(cfg, U[rows[j]], nk) if cfg.mux == 'ideal' else transmit_orth(cfg, U[rows[j]], h[t, act[j]], nk, bf)
            for j, (r, led, dg) in res.items():
                W[j] += r
                led.compute_calls = len(act[j]); leds[j].add(led)
                d = diags[j]
                for kk in DIAG_SUM:
                    d[kk] += dg[kk]
                for kk in ('leak_by_src', 'u_leak'):
                    for a, v in dg[kk].items():
                        d[kk][a] = d[kk].get(a, 0.0) + v
                d['rounds'] += 1; d['sent_sum'] += len(act[j])
                if hook is not None:
                    hook(j, t, act[j], r, U[rows[j]])
        if log and log_every and (t + 1) % log_every == 0:
            log(f'  라운드 {t + 1}/{T_all}  (shard, member) 쌍={len(pairs)}')
    for j, s in enumerate(shards):
        if s.T in s.save_at:
            ckpts[j][s.T] = W[j].clone()
    assert torch.isfinite(W).all(), '모델에 유한하지 않은 값'
    return W, leds, diags, ckpts

def per_round(d, field):
    return d[field] / max(1, d['rounds'])
