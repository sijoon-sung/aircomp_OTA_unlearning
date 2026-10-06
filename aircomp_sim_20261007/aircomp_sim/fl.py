"""shard 단위 AirComp 연합학습 — 라운드 루프와 삭제 절차.

여러 shard 작업(job)을 한 라운드씩 함께 진행한다. 로컬 학습은 vmap 으로 묶어서 계산하고(실제 SGD),
무선 전송은 job 마다 radio.transmit 을 실제로 호출한다. 전역 전력 정렬(align='shared')과 전역 스케줄링은
같은 system 에 속한 job 들의 채널을 모아 계산한다.
"""
import sys
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'sisa_arch_20261006'))
from sisa_arch.common import (load_split, channels, slow_channel, partition, shard_of, init_vector, key, N_CLIENTS, CLIP,
                              write_json, read_json, Logger, environment, PowerMeter, ensure_dataset, setup_torch, pick_device)
from sisa_arch.trainer import LocalTrainer, Evaluator, class_metrics, disagreement, kl
from .radio import Radio, RadioConfig, Ledger

@dataclass
class ShardJob:
    name: str
    members: list              # 송신 가능한 client
    w0: torch.Tensor
    T: int
    t0: int = 0
    system: str = 'default'    # 같은 system 의 job 들끼리 무선 제어를 공유할 수 있음
    tape: int = 0              # 잡음 난수 tape (shard 번호)
    salt: str = ''
    save_at: tuple = ()
    bw: float = 1.0            # 이 job 이 쓰는 자원 블록 수(삭제 중 빈 블록 재사용)
    radio: RadioConfig = None  # None 이면 run 의 기본 설정
    record_history: bool = False

def run_rounds(tr, jobs, h, seed, radio_default: RadioConfig, log=None, log_every=0):
    """반환: W [J, D], checkpoints list[dict t->tensor], ledgers list[Ledger], histories list[dict t->tensor] (record_history 인 job 만)."""
    dev = tr.device; D = tr.D
    W = torch.stack([j.w0.to(dev).clone() for j in jobs])
    radios = {}
    for j in jobs:
        cfg = j.radio or radio_default
        if id(cfg) not in radios:
            radios[id(cfg)] = Radio(cfg, D, seed, dev)
    leds = [Ledger() for _ in jobs]; cks = [dict() for _ in jobs]; hist = [dict() for _ in jobs]
    t_lo = min(j.t0 for j in jobs); t_hi = max(j.T for j in jobs)
    for t in range(t_lo, t_hi):
        live = [ji for ji, j in enumerate(jobs) if j.t0 <= t < j.T]
        for ji in live:
            j = jobs[ji]
            if t in j.save_at:
                cks[ji][t] = W[ji].clone()
            if j.record_history:
                hist[ji][t] = W[ji].clone()
        # 스케줄링 (system 별 공유 가능)
        active = {}
        for ji in live:
            j = jobs[ji]; cfg = j.radio or radio_default; rd = radios[id(cfg)]
            if cfg.sched is not None and cfg.sched[0] == 'shared':
                pool = [i for jk in live if jobs[jk].system == j.system for i in jobs[jk].members]
            else:
                pool = j.members
            active[ji] = rd.active(j.members, h[t], h[t, pool])
        pairs = [(ji, i) for ji in live for i in active[ji]]
        if not pairs:
            continue
        jidx = [p[0] for p in pairs]; clients = [p[1] for p in pairs]; salts = [jobs[p[0]].salt for p in pairs]
        U, nrm = tr.updates(W[torch.tensor(jidx, device=dev)], clients, t, salts)
        # 전력 정렬 기준 (system 별 공유 가능)
        for ji in live:
            j = jobs[ji]; act = active[ji]
            if not act:
                continue
            cfg = j.radio or radio_default; rd = radios[id(cfg)]
            if cfg.align == 'shared':
                pool = [i for jk in live if jobs[jk].system == j.system for i in active[jk]]
                hmin = float(h[t, pool].min())
            else:
                hmin = float(h[t, act].min())
            rows = [k for k, p in enumerate(pairs) if p[0] == ji]
            X = U[rows]
            nk = key(seed, j.tape, t, 'noise', j.salt)
            r, led = rd.transmit(X, h[t, act], hmin, bw=j.bw, noise_key=nk)
            W[ji] += r
            led.compute_calls = len(act); leds[ji].add(led)
            for k in rows:
                if float(nrm[k]) > CLIP:
                    pass
        if log_every and log and (t + 1) % log_every == 0:
            log(f'  round {t + 1}/{t_hi} pairs={len(pairs)}')
    for ji, j in enumerate(jobs):
        if j.T in j.save_at:
            cks[ji][j.T] = W[ji].clone()
        if j.record_history:
            hist[ji][j.T] = W[ji].clone()
    assert torch.isfinite(W).all()
    return W, cks, leds, hist

def ensemble_metrics(ev, plist):
    return ev.ensemble(plist)
