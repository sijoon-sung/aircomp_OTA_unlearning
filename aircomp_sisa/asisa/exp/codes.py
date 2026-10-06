"""codes — 시간 오차에 강한 코드(ZCZ)로 shard 간 간섭을 구조적으로 0 으로 만들 수 있는가, 그 대가는 무엇인가.

코드 (radio.py)
  walsh L4  : 칩 4개. 시간 오차가 있으면 간섭.
  walsh L8  : 칩 8개 (ZCZ gap1 과 같은 자원). shard 4개에 길이 8 walsh 의 행 4개를 쓴다.
  zcz gap1  : walsh L4 칩 사이에 0 칩 1개 (칩 8개). 시간 오차 1 칩 미만이면 간섭 0, 잡음 2배.
  zcz gap2  : 0 칩 2개 (칩 12개). 시간 오차 2 칩 미만이면 간섭 0, 잡음 3배.
  pn L16    : 칩 16개.
시간 오차: client 마다 U[0, 상한] 칩 (정적). 상한 0.5 (ZCZ gap1 의 영 상관 구간 안) / 1.5 (gap1 밖, gap2 안).
  대조군: walsh L4, 시간 오차 0 (완전 직교).
고정: 채널 정렬 자르기 배정 (near-far 가 가장 큰 경우), 20dB, 코드 배치는 interference 와 같다 (가장 강한 shard = 코드 2, 가장 약한 shard = 코드 3).
전력 정렬: 최대 전력 / 가장 약한 shard 에 맞춤. 코드만으로 충분한지, 도착 크기 맞추기와 함께 써야 하는지를 본다.
측정: deletion.py 와 같음 (파라미터 차이와 그 난수 기준선, u 간섭/자기, 예측 불일치) + 라운드당 칩 수, 집계 오차, 정확도.
"""
import numpy as np
from . import seed_context, db
from .deletion import add_system, measure
from ..channel import partition
from ..config import N_CLIENTS
from ..fl import train
from ..radio import RadioConfig
from ..util import write_json, fmt, pct, table, select, mean, paired, pm

NAME = 'codes'
K = 4
SIGMA2 = .01
CODES = [('walsh4', 'walsh', 4, 0), ('walsh8', 'walsh', 8, 0), ('zcz4g1', 'zcz', 4, 1), ('zcz4g2', 'zcz', 4, 2), ('pn16', 'pn', 16, 0)]
CODE_KO = {'walsh4': 'walsh L4 (칩 4)', 'walsh8': 'walsh L8 (칩 8)', 'zcz4g1': 'ZCZ gap1 (칩 8)', 'zcz4g2': 'ZCZ gap2 (칩 12)', 'pn16': 'PN L16 (칩 16)'}
DELAYS = [0.5, 1.5]
ALIGNS = ['maxpow', 'weakest']
ALIGN_KO = {'maxpow': '최대 전력', 'weakest': '가장 약한 shard 에 맞춤'}

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    T, eps = cfg['T'], cfg['eps']
    codes = [CODES[0], CODES[2]] if cfg['quick'] else CODES
    delays = [0.5] if cfg['quick'] else DELAYS
    rows, sysrows = [], []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed)
        groups = partition('rank_chunk', range(N_CLIENTS), c.base, K, seed)
        hmin = [float(min(c.base[i] for i in g)) for g in groups]
        strong, weak = int(np.argmax(hmin)), int(np.argmin(hmin))
        rest = [k for k in range(K) if k not in (strong, weak)]
        slot_of = {rest[0]: 0, rest[1]: 1, strong: 2, weak: 3}
        best = lambda g: int(max(g, key=lambda i: c.base[i]))
        targets = {'in_strong': best(groups[strong]), 'in_weak': best(groups[weak])}
        focus = {'in_strong': weak, 'in_weak': strong}
        runs = [(0.0, [CODES[0]])] + [(dm, codes) for dm in delays]
        for dm, cl in runs:
            shards, radio, track, tags = [], {}, {}, []
            for name, code, L, gap in cl:
                for al in ALIGNS:
                    rc = RadioConfig(sigma2=SIGMA2, eps=eps, align=al, mux='code', code=code, L=L, gap=gap, delay_max=dm)
                    tag = f'{name}|{dm}|{al}'; tags.append((tag, name, al))
                    add_system(shards, radio, track, tag, groups, slot_of, targets, rc, T)
            log(f'[codes] seed {seed} 시간 오차 상한 {dm}: shard 모델 {len(shards)}개')
            W, leds, diags, _ = train(c.tr, shards, c.h, c.w0, seed, lambda s: radio[s], track_of=lambda s: track.get(s, ()), log=log, log_every=T // 4)
            P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
            for tag, name, al in tags:
                r, sr = measure(c, W, P, leds, diags, ix, tag, groups, slot_of, targets, hmin, focus,
                                dict(seed=seed, code=name, delay_max=dm, align=al))
                rows += r; sysrows.append(sr)
            del W, P
        log(f'[codes] seed {seed} 끝')
    write_json(out / 'results.json', dict(rows=rows, systems=sysrows, setting=dict(K=K, T=T, eps=eps, seeds=list(cfg['seeds']), codes=[x[0] for x in codes], delays=delays)))
    L = ['# codes — 시간 오차에 강한 코드(ZCZ)와 shard 간 간섭', '',
         f'채널 정렬 자르기 배정, {db(SIGMA2)}, {cfg["T"]}라운드, seed {len(cfg["seeds"])}개. 가장 강한 shard = 코드 2, 가장 약한 shard = 코드 3. '
         'ZCZ = walsh 칩 사이에 0 칩을 넣고, 수신기가 0..gap 칩 늦게 온 신호를 모두 모으는 창으로 역확산한다. 시간 오차가 gap 칩 미만이면 다른 shard 와 상관이 정확히 0.', '',
         '열: 칩 = 심볼 하나에 쓰는 칩 수 (자원). 파라미터 차이 = 흔적을 받는 shard 모델의 ||W(u 있음) − W(u 없이 처음부터)|| / ||W(u 없이)||, 파라미터 난수 변동 = 다른 배치·잡음으로 학습했을 때의 같은 양.', '']
    ctrl = mean(select(rows, delay_max=0.0), 'rel')
    for tn, title in [('in_strong', '## 1. 가장 강한 shard 의 member 를 지울 때 가장 약한 shard 에 남는 흔적'),
                      ('in_weak', '## 2. 가장 약한 shard 의 member 를 지울 때 가장 강한 shard 에 남는 흔적')]:
        L += [title, '']
        tb = []
        for dm in [0.0] + delays:
            for name, code, Lc, gap in codes:
                for al in ALIGNS:
                    rs = select(rows, code=name, delay_max=dm, align=al, target=tn, role='focus')
                    ss = select(sysrows, code=name, delay_max=dm, align=al)
                    if not rs:
                        continue
                    rl, ry = mean(rs, 'rel'), mean(rs, 'rel_yard')
                    tb.append([fmt(dm, 1), CODE_KO[name], ALIGN_KO[al], fmt(mean(ss, 'ul') / c.tr.D, 0), fmt(mean(rs, 'uleak_to_own'), 4), fmt(mean(rs, 'leak_to_own'), 4),
                               fmt(rl, 4), fmt(ry, 4), fmt(rl / ry if ry > 0 else float('nan'), 4), pct(mean(rs, 'trace')), fmt(mean(ss, 'mse'), 4), pct(mean(ss, 'acc_ens'))])
        L += [table(['시간 오차 상한(칩)', '코드', '전력 정렬', '칩', 'u 간섭/자기', '간섭/자기(전체)', '파라미터 차이', '파라미터 난수 변동', '차이÷난수',
                     '예측 불일치', '평균 집계 오차', '앙상블 정확도'], tb), '']
    L += ['## 3. 판정', '', f'- 대조군 (walsh L4, 시간 오차 0) 파라미터 차이 = {fmt(ctrl, 4)} (부동소수점 바닥).']
    for dm in delays:
        for al in ALIGNS:
            base = dict(delay_max=dm, align=al, target='in_strong', role='focus')
            for name in ['zcz4g1', 'zcz4g2']:
                if not select(rows, code=name, **base):
                    continue
                v = mean(select(rows, code=name, **base), 'rel'); lk = mean(select(rows, code=name, **base), 'leak_to_own')
                inside = (name == 'zcz4g1' and dm <= 1) or (name == 'zcz4g2' and dm <= 2)
                m, se, n = paired(rows, 'rel', base, dict(code='walsh4'), dict(code=name))
                L.append(f'- 시간 오차 ≤ {dm} 칩, {ALIGN_KO[al]}, {CODE_KO[name]}: 영 상관 구간 {"안" if inside else "밖"}. 간섭/자기 = {fmt(lk, 4)}, '
                         f'파라미터 차이 = {fmt(v, 4)} (대조군 {fmt(ctrl, 4)}), walsh L4 대비 감소 = {pm(m, se, fmt)} (n={n})')
    L += ['', '## 읽는 법', '',
          '- 시간 오차가 영 상관 구간 안이면 ZCZ 의 간섭/자기는 0 이고 파라미터 차이는 대조군 바닥 수준이어야 한다. 구간 밖이면 간섭이 다시 생긴다.',
          '- 같은 칩 수(8)의 walsh L8 과 ZCZ gap1 을 비교하면, 같은 자원을 "코드 수를 늘리는 데" 쓰는 것과 "시간 오차를 견디는 데" 쓰는 것의 차이가 보인다.',
          '- ZCZ 의 대가는 칩 수 (gap+1) 배와 잡음 (gap+1) 배다. 평균 집계 오차와 정확도 열에서 잡음 증가가 정확도를 해치는지 본다.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
