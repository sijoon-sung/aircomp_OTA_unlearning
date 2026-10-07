"""control (E1) — shard 를 시간 슬롯(직교 블록)으로 나눠도, 전력·스케줄링을 shard 밖 정보로 정하면 정확한 언러닝이 깨지는가.

TDMA 처럼 shard 마다 직교 블록을 쓰면 신호는 섞이지 않는다. 남는 것은 '제어' 의 섞임이다.
  shard 안 정렬:    beta_k = C / (min_{S_k}|h| sqrt(PD))         (shard 자기 최약 member)
  전체 정렬:        beta_k = C / (min_{전체}|h| sqrt(PD))         (기존 AirComp 관행: 전체 최약 노드의 한계에 맞춤)
  shard 안 스케줄링: shard 안 채널 하위 20% 는 그 라운드에 송신하지 않음
  전체 스케줄링:    전체 채널 하위 20% 는 송신하지 않음
삭제 대상이 전체 최약 노드 u 이면, 전체 정렬에서는 'u 가 처음부터 없었다면' 모든 shard 의 beta (전력·잡음) 가 달랐다.
전체 스케줄링에서는 문턱이 바뀌어 다른 shard 의 참여자가 바뀐다. 그런데 SISA 는 u 의 shard 만 재학습한다.
설정: 무작위 배정 4 shard, 직교 블록, SNR 20 / 0 / -10dB, 삭제 대상 = 전체 최약 노드 / 채널 중간 노드.
측정 (deletion.py): 미삭제 shard 의 파라미터 차이 (u 있음 vs u 없이 처음부터 같은 제어 규칙으로) 와 그 난수 기준선,
  SISA 앙상블 (u 의 shard 만 재학습) vs 전체 재학습 앙상블의 예측 불일치와 확률 거리, 비용 (에너지, 슬롯, 집계 오차), 정확도.
예상: shard 안 정렬·스케줄링은 미삭제 shard 의 차이가 0 (직교라 같은 계산). 전체 정렬은 최약 노드 삭제 때만 0 이 아니고, 잡음이 클수록 크다.
"""
import numpy as np
from . import seed_context, db
from .deletion import add_system, measure
from ..channel import partition
from ..config import N_CLIENTS
from ..fl import train
from ..radio import RadioConfig
from ..util import write_json, fmt, pct, pp, table, select, mean, paired, pm

NAME = 'control'
K = 4
CTRLS = {'local': dict(align='maxpow'), 'global_align': dict(align='global'),
         'local_sched': dict(align='maxpow', sched=('shard', .2)), 'global_sched': dict(align='maxpow', sched=('global', .2))}
CTRL_KO = {'local': 'shard 안 정렬', 'global_align': '전체 정렬 (기존 관행)', 'local_sched': 'shard 안 스케줄링', 'global_sched': '전체 스케줄링'}
SNRS = [0.01, 1.0, 10.0]
TARGET_KO = {'weakest': '전체 최약 노드', 'median': '채널 중간 노드'}

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    T, eps = cfg['T'], cfg['eps']
    snrs = [1.0] if cfg['quick'] else SNRS
    ctrls = ['local', 'global_align', 'global_sched'] if cfg['quick'] else list(CTRLS)
    rows, sysrows = [], []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed)
        groups = partition('random_split', range(N_CLIENTS), c.base, K, seed)
        hmin = [float(min(c.base[i] for i in g)) for g in groups]
        order = np.argsort(c.base)
        targets = {'weakest': int(order[0]), 'median': int(order[N_CLIENTS // 2])}
        slot_of = {k: k for k in range(K)}
        for s2 in snrs:
            shards, radio, track = [], {}, {}
            for cn in ctrls:
                add_system(shards, radio, track, cn, groups, slot_of, targets, RadioConfig(sigma2=s2, eps=eps, **CTRLS[cn]), T)
            log(f'[control] seed {seed} {db(s2)}: shard 모델 {len(shards)}개')
            W, leds, diags, _ = train(c.tr, shards, c.h, c.w0, seed, lambda s: radio[s], log=log, log_every=T // 4)
            P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
            for cn in ctrls:
                r, sr = measure(c, W, P, leds, diags, ix, cn, groups, slot_of, targets, hmin, None, dict(seed=seed, ctrl=cn, sigma2=s2))
                rows += r; sysrows.append(sr)
            del W, P
        log(f'[control] seed {seed} 끝')
    write_json(out / 'results.json', dict(rows=rows, systems=sysrows, setting=dict(K=K, T=T, eps=eps, seeds=list(cfg['seeds']), snrs=snrs, ctrls=ctrls)))
    L = ['# control (E1) — 시간 슬롯으로 나눠도 제어를 shard 밖 정보로 정하면 정확성이 깨지는가', '',
         f'무작위 배정 {K} shard, 직교 블록 (shard 사이 신호 섞임 없음), {cfg["T"]}라운드, seed {len(cfg["seeds"])}개, eps = {fmt(cfg["eps"])}.', '',
         '열: 파라미터 차이 = 삭제 대상의 shard 가 아닌 shard 모델의 ||W(u 있음) − W(u 없이 처음부터)|| / ||W(u 없이)|| (미삭제 shard 평균·최대). '
         '난수 변동 = 같은 조건을 다른 배치·잡음으로 학습했을 때의 같은 양. 앙상블 거리 = SISA (u 의 shard 만 재학습) 와 전체 재학습 앙상블의 test 확률 분포 거리 (total variation 평균) 와 예측 불일치. '
         '앙상블 난수 기준 = 원래 학습을 다른 난수로 했을 때 두 앙상블의 같은 거리.', '']
    for tn in ['weakest', 'median']:
        L += [f'## 삭제 대상: {TARGET_KO[tn]}', '']
        tb = []
        for s2 in snrs:
            for cn in ctrls:
                rs = select(rows, ctrl=cn, sigma2=s2, target=tn); ss = select(sysrows, ctrl=cn, sigma2=s2)
                if not rs:
                    continue
                tb.append([db(s2), CTRL_KO[cn], fmt(mean(rs, 'rel'), 4), fmt(max(r['rel'] for r in rs), 4), fmt(mean(rs, 'rel_yard'), 4),
                           fmt(mean(ss, f'ens_tv_{tn}'), 4), pct(mean(ss, f'ens_dis_{tn}')), fmt(mean(ss, 'ens_yard_tv'), 4), pct(mean(ss, 'ens_yard_dis'))])
        L += [table(['SNR', '제어', '미삭제 shard 파라미터 차이 평균', '최대', '파라미터 난수 변동', '앙상블 확률 거리 (SISA vs 전체 재학습)', '앙상블 예측 불일치',
                     '앙상블 확률 거리 난수 기준', '앙상블 예측 불일치 난수 기준'], tb), '']
    L += ['## 비용', '']
    tb = []
    for s2 in snrs:
        for cn in ctrls:
            ss = select(sysrows, ctrl=cn, sigma2=s2)
            if ss:
                tb.append([db(s2), CTRL_KO[cn], fmt(mean(ss, 'energy')), fmt(mean(ss, 'ul') / c.tr.D, 2), fmt(mean(ss, 'mse'), 4), pct(mean(ss, 'acc_ens'))])
    L += [table(['SNR', '제어', '학습 에너지/라운드 (4 shard 합)', 'shard 하나의 라운드당 UL (D 단위, 반복 포함)', '평균 집계 오차', '앙상블 정확도'], tb), '',
          '## 판정 (± 는 seed 짝 차이의 2×표준오차)', '']
    for s2 in snrs:
        for cn in [x for x in ctrls if x != 'local']:
            m, se, n = paired(rows, 'rel', dict(sigma2=s2, target='weakest'), dict(ctrl=cn), dict(ctrl='local'), keys=('seed', 'shard'))
            L.append(f'- {db(s2)}, 전체 최약 노드 삭제: 파라미터 차이 ({CTRL_KO[cn]} − shard 안 정렬) = {pm(m, se, fmt)} (n={n}) → '
                     f'{"shard 밖 제어가 흔적을 남김" if (n > 1 and m > 2 * se) else "차이 확인 안 됨"}')
        m, se, n = paired(sysrows, 'acc_ens', dict(sigma2=s2), dict(ctrl='global_align'), dict(ctrl='local'), keys=('seed',))
        L.append(f'- {db(s2)}, 비용: 앙상블 정확도 (전체 정렬 − shard 안 정렬) = {pm(m, se, pp)} (n={n})')
    L += ['', '## 읽는 법', '',
          '- shard 안 정렬·스케줄링은 직교 블록이라 미삭제 shard 의 계산이 u 와 무관하다. 파라미터 차이는 부동소수점 바닥(약 1e-4) 이어야 한다.',
          '- 전체 정렬에서 전체 최약 노드를 지우면 다른 shard 의 beta 가 바뀌므로 파라미터 차이가 0 보다 커야 하고, 잡음이 큰 SNR 에서 더 커야 한다. 채널 중간 노드를 지우면 기준이 바뀌지 않으므로 0 이어야 한다.',
          '- 전체 스케줄링은 한 명이 빠지면 문턱이 바뀌어 다른 shard 의 송신자가 바뀌므로, 데이터 수준의 차이가 생긴다.',
          '- 앙상블 확률 거리가 난수 기준과 비슷하면, SISA 결과가 전체 재학습과 다른 정도가 학습 난수만큼 크다는 뜻이다 (정확하지 않음).',
          '- 비용 표에서 전체 정렬은 강한 shard 도 약한 노드 기준으로 보내므로 잡음이 커지거나 반복이 늘어난다. 즉 shard 안 정렬은 정확성을 지키면서 비용도 낮다.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
