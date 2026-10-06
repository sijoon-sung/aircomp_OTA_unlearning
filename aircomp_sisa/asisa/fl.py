"""shard 단위 AirComp 연합학습의 라운드 루프.

shard 모델 여러 개(학습 경로 하나씩)를 라운드마다 함께 진행한다.
  1. 모든 (shard, member) 쌍의 로컬 update 를 vmap 으로 한 번에 계산한다.
  2. system 마다 무선 전송을 실제로 실행한다. 같은 system 의 shard 들은 같은 라운드에 보내고,
     mux='code' 면 같은 자원을 공유해 서로 간섭한다. mux='orth' 면 각자 블록을 쓴다.
  3. 각 shard 모델에 수신한 평균 추정을 더한다.

난수 짝 맞춤
  minibatch 는 (seed, client, 라운드, step, salt), 잡음은 (seed, 라운드, salt[, 슬롯 번호]) 로 정해진다.
  그래서 salt 가 같은 두 학습 경로(예: u 있음 / u 없이 처음부터)는 같은 배치와 같은 잡음을 쓰고,
  차이는 바꾼 조건에서만 나온다. salt 를 바꾸면 '학습 난수 변동'의 기준선이 된다.
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

DIAG_SUM = ('own_energy', 'leak_energy', 'noise_energy', 'alpha', 'beta', 'repeats', 'slots')

def chip_delays(seed, delay_max):
    return delay_max * np.array([np.random.default_rng(key(seed, 'chipdelay', i)).uniform() for i in range(N_CLIENTS)])

def train(tr, shards, h, w0, seed, radio_of, track_of=None, log=None, log_every=0):
    """shards: Shard 목록, h: [T, N] 채널, radio_of(system) -> RadioConfig, track_of(system) -> 간섭을 따로 잴 client 들.
    반환: W [J, D] 최종 모델, Ledger 목록, 진단 목록(라운드 합; rounds 로 나누면 평균), 체크포인트 목록."""
    dev = tr.device
    W = w0.to(dev)[None].repeat(len(shards), 1)
    leds = [Ledger() for _ in shards]; diags = [dict(new_diag(), rounds=0) for _ in shards]; ckpts = [dict() for _ in shards]
    systems = {}
    for j, s in enumerate(shards):
        systems.setdefault(s.system, []).append(j)
    code_sys = {}
    for name, idx in systems.items():
        cfg = radio_of(name)
        if cfg.mux == 'code':
            code_sys[name] = CodeSystem(cfg, 1 + max(shards[j].slot for j in idx), seed, chip_delays(seed, cfg.delay_max))
    for t in range(max(s.T for s in shards)):
        live = [j for j, s in enumerate(shards) if t < s.T]
        for j in live:
            if t in shards[j].save_at:
                ckpts[j][t] = W[j].clone()
        pairs = [(j, i) for j in live for i in shards[j].members]
        if not pairs:
            continue
        U = tr.updates(W[torch.tensor([p[0] for p in pairs], device=dev)], [p[1] for p in pairs], t, [shards[p[0]].salt for p in pairs])
        rows = {}
        for k, (j, _) in enumerate(pairs):
            rows.setdefault(j, []).append(k)
        for name, idx in systems.items():
            act = [j for j in idx if j in rows]
            if not act:
                continue
            cfg = radio_of(name); res = {}
            if cfg.mux == 'code':
                out, ld, dg = code_sys[name].transmit(
                    [(shards[j].slot, U[rows[j]], shards[j].members, h[t, shards[j].members]) for j in act],
                    key(seed, t, 'noise', shards[act[0]].salt), track_of(name) if track_of else ())
                res = {j: (out[shards[j].slot], ld[shards[j].slot], dg[shards[j].slot]) for j in act}
            else:
                for j in act:
                    nk = key(seed, t, 'noise', shards[j].salt, shards[j].slot)
                    res[j] = transmit_ideal(cfg, U[rows[j]], nk) if cfg.mux == 'ideal' else transmit_orth(cfg, U[rows[j]], h[t, shards[j].members], nk)
            for j, (r, led, dg) in res.items():
                W[j] += r
                led.compute_calls = len(shards[j].members); leds[j].add(led)
                d = diags[j]
                for kk in DIAG_SUM:
                    d[kk] += dg[kk]
                for kk in ('leak_by_src', 'u_leak'):
                    for a, v in dg[kk].items():
                        d[kk][a] = d[kk].get(a, 0.0) + v
                d['rounds'] += 1
        if log and log_every and (t + 1) % log_every == 0:
            log(f'  라운드 {t + 1}/{max(s.T for s in shards)}  (shard, member) 쌍={len(pairs)}')
    for j, s in enumerate(shards):
        if s.T in s.save_at:
            ckpts[j][s.T] = W[j].clone()
    assert torch.isfinite(W).all(), '모델에 유한하지 않은 값'
    return W, leds, diags, ckpts

def per_round(d, field):
    return d[field] / max(1, d['rounds'])
