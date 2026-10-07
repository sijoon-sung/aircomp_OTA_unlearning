"""subcarrier (R1~R3) — TDMA 말고 shard 를 무선 자원에 배치하는 방법: FDMA, 채널 인식 OFDMA, TDMA+FDMA 혼합, 보호 대역.

사실 하나: shard K 개의 합을 따로 받으려면 어떤 방식이든 전체 자원은 K·D 심볼이다. 대역폭이 같으면 FDMA 도 라운드 시간은 TDMA 와 같다.
대신 FDMA 는 기기가 좁은 대역에 같은 전력을 K 배 오래 실으므로 부반송파당 전력이 K 배다 -> 약한 기기가 전력 상한에 걸리는
AirComp 에서 잡음·반복이 준다. 대가는 주파수 오차(CFO) 가 있을 때 이웃 부반송파로 새는 간섭 (shard 간 섞임) 이다.
AirComp 특유의 점: (1) shard 의 부반송파 품질은 member 중 최약 채널로 정해진다 -> 채널 인식 할당은 'shard 전원이 괜찮은 부반송파' 를 찾는다.
                  (2) 이웃 블록으로 새는 간섭에도 near-far (beta 비) 가 붙는다 -> 블록 순서 (도착 크기가 비슷한 shard 끼리 이웃) 가 설계 변수다.
R1 배치 방식 비교 (무작위 배정): TDMA / FDMA 연속 블록 / FDMA 블록 위치를 채널에 맞춤 / OFDMA 섞어 배치 / OFDMA 채널 인식 (조각남)
   x 주파수 오차 상한 {0, 0.05} x SNR {20, -10dB}
R2 보호 대역 (채널 정렬 자르기 배정 = near-far 가 큰 경우): FDMA 블록, 보호 부반송파 {0,1,2,4} x 블록 순서 {slot 순, 최약 채널 순} x 주파수 오차 {0.05, 0.1}, 20dB
R3 TDMA+FDMA 혼합 (무작위 배정): 시간 슬롯 G = 2 (shard 2개씩 FDMA), 주파수 오차 0.05, SNR {20, -10dB}. G=1 (FDMA) 과 G=4 (TDMA) 는 R1 과 같다.
고정: 부반송파 64개, 주파수 선택적 채널 (4-tap), shard 안 최대 전력 정렬 (평균 전력 기준), 잡음 허용 eps 를 넘으면 반복.
측정 (deletion.py): 라운드 시간 (D/F 심볼 단위, TDMA 반복 없음 = 4), 반복, 집계 오차, 에너지, 간섭/자기 신호,
  가장 강한 shard 의 member 를 지울 때 가장 약한 shard 의 파라미터 차이와 그 난수 기준선, 정확도.
"""
import numpy as np
from . import seed_context, db
from .deletion import add_system, measure
from ..channel import partition, RULE_KO
from ..config import N_CLIENTS
from ..fl import train
from ..radio import RadioConfig
from ..util import write_json, fmt, pct, pp, table, select, mean, paired, pm

NAME = 'subcarrier'
K = 4
LAYOUT = {'tdma': dict(groups=K), 'fdma': dict(alloc='block'), 'fdma_aware': dict(alloc='block_aware'),
          'ofdma_inter': dict(alloc='interleave'), 'ofdma_aware': dict(alloc='aware'), 'hybrid2': dict(groups=2, alloc='block')}
LAYOUT_KO = {'tdma': 'TDMA', 'fdma': 'FDMA 연속 블록', 'fdma_aware': 'FDMA 블록 위치 채널 인식', 'ofdma_inter': 'OFDMA 섞어 배치',
             'ofdma_aware': 'OFDMA 채널 인식 (조각남)', 'hybrid2': 'TDMA 2슬롯 + FDMA 2블록'}
ORDER_KO = {'fixed': 'slot 순', 'beta': '최약 채널 순 (이웃 도착 크기 비슷)'}

def configs(quick):
    if quick:
        return [('R1', 'random_split', f'{lay}|0.05|1.0', dict(LAYOUT[lay], cfo=0.05, sigma2=1.0)) for lay in ['tdma', 'fdma', 'ofdma_inter']]
    out = []
    for lay in ['tdma', 'fdma', 'fdma_aware', 'ofdma_inter', 'ofdma_aware']:
        for cfo in [0.0, 0.05]:
            for s2 in [0.01, 10.0]:
                out.append(('R1', 'random_split', f'{lay}|{cfo}|{s2}', dict(LAYOUT[lay], cfo=cfo, sigma2=s2)))
    for g in [0, 1, 2, 4]:
        for order in ['fixed', 'beta']:
            for cfo in [0.05, 0.1]:
                out.append(('R2', 'rank_chunk', f'g{g}|{order}|{cfo}', dict(alloc='block', guard=g, block_order=order, cfo=cfo, sigma2=0.01)))
    for s2 in [0.01, 10.0]:
        out.append(('R3', 'random_split', f'hybrid2|0.05|{s2}', dict(LAYOUT['hybrid2'], cfo=0.05, sigma2=s2)))
    return out

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    T, eps = cfg['T'], cfg['eps']
    confs = configs(cfg['quick'])
    rows, sysrows = [], []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed)
        lay = {}
        for rule in sorted({x[1] for x in confs}):
            groups = partition(rule, range(N_CLIENTS), c.base, K, seed)
            hmin = [float(min(c.base[i] for i in g)) for g in groups]
            strong, weak = int(np.argmax(hmin)), int(np.argmin(hmin))
            best = lambda g: int(max(g, key=lambda i: c.base[i]))
            lay[rule] = (groups, hmin, {'in_strong': best(groups[strong]), 'in_weak': best(groups[weak])}, {'in_strong': weak, 'in_weak': strong})
        chunks = [[x for x in confs if x[0] == 'R2']] + [[x for x in confs if x[0] != 'R2' and x[3]['sigma2'] == s2] for s2 in sorted({x[3]['sigma2'] for x in confs})]
        for ch in [x for x in chunks if x]:
            shards, radio, track = [], {}, {}
            for part, rule, tag, kw in ch:
                groups, hmin, targets, focus = lay[rule]
                add_system(shards, radio, track, f'{part}|{tag}', groups, {k: k for k in range(K)}, targets,
                           RadioConfig(eps=eps, mux='ofdm', **kw), T)
            log(f'[subcarrier] seed {seed}: 설정 {len(ch)}개, shard 모델 {len(shards)}개')
            W, leds, diags, _ = train(c.tr, shards, c.h, c.w0, seed, lambda s: radio[s], track_of=lambda s: track.get(s, ()), log=log, log_every=T // 4)
            P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
            for part, rule, tag, kw in ch:
                groups, hmin, targets, focus = lay[rule]
                extra = dict(seed=seed, part=part, rule=rule, tag=tag, layout=tag.split('|')[0], cfo=kw['cfo'], sigma2=kw['sigma2'],
                             guard=kw.get('guard', 0), order=kw.get('block_order', 'fixed'))
                r, sr = measure(c, W, P, leds, diags, ix, f'{part}|{tag}', groups, {k: k for k in range(K)}, targets, hmin, focus, extra)
                sr['energy_total'] = sum(leds[ix[f'{part}|{tag}|src|{k}']].energy for k in range(K)) / max(1, leds[ix[f'{part}|{tag}|src|0']].rounds)
                sr['repeats_mean'] = float(np.mean([diags[ix[f'{part}|{tag}|src|{k}']]['repeats'] / max(1, diags[ix[f'{part}|{tag}|src|{k}']]['rounds']) for k in range(K)]))
                rows += r; sysrows.append(sr)
            del W, P
        log(f'[subcarrier] seed {seed} 끝')
    write_json(out / 'results.json', dict(rows=rows, systems=sysrows, setting=dict(K=K, T=T, eps=eps, seeds=list(cfg['seeds']), configs=[x[:3] for x in confs])))
    L = ['# subcarrier (R1~R3) — shard 를 부반송파·시간에 배치하는 방법', '',
         f'부반송파 64개, 주파수 선택적 채널, shard {K}개, {cfg["T"]}라운드, seed {len(cfg["seeds"])}개, eps = {fmt(cfg["eps"])}. '
         '라운드 시간은 D/F 심볼 단위 (TDMA 에서 반복이 없으면 4). 간섭과 파라미터 차이는 가장 강한 shard 의 member 를 지울 때 가장 약한 shard 에서 잰 값.', '',
         '파라미터 차이 = 그 shard 모델의 ||W(u 있음) − W(u 없이 처음부터)|| / ||W(u 없이)||, 난수 변동 = 같은 조건을 다른 배치·잡음으로 학습했을 때의 같은 양.', '']
    hd = ['라운드 시간', '반복', '집계 오차', '에너지/라운드', '간섭/자기 (약한 shard)', '최대 간섭/자기', '파라미터 차이', '파라미터 난수 변동', '앙상블 정확도']
    def cells(rs, ss):
        return [fmt(mean(ss, 'slots'), 2), fmt(mean(ss, 'repeats_mean'), 2), fmt(mean(ss, 'mse'), 4), fmt(mean(ss, 'energy_total')), fmt(mean(rs, 'leak_to_own'), 4),
                fmt(mean(ss, 'max_leak_to_own'), 4), fmt(mean(rs, 'rel'), 4), fmt(mean(rs, 'rel_yard'), 4), pct(mean(ss, 'acc_ens'))]
    L += ['## R1. 배치 방식 (무작위 배정)', '']
    tb = []
    for s2 in sorted({x['sigma2'] for x in sysrows if x['part'] == 'R1'}):
        for cfo in sorted({x['cfo'] for x in sysrows if x['part'] == 'R1'}):
            for lay_ in ['tdma', 'fdma', 'fdma_aware', 'ofdma_inter', 'ofdma_aware']:
                ss = select(sysrows, part='R1', layout=lay_, cfo=cfo, sigma2=s2); rs = select(rows, part='R1', layout=lay_, cfo=cfo, sigma2=s2, target='in_strong', role='focus')
                if ss:
                    tb.append([db(s2), cfo, LAYOUT_KO[lay_]] + cells(rs, ss))
    L += [table(['SNR', '주파수 오차', '배치'] + hd, tb), '']
    L += ['## R2. 보호 대역과 블록 순서 (채널 정렬 자르기 배정, 20dB)', '']
    tb = []
    for cfo in sorted({x['cfo'] for x in sysrows if x['part'] == 'R2'}):
        for order in ['fixed', 'beta']:
            for g in [0, 1, 2, 4]:
                ss = select(sysrows, part='R2', guard=g, order=order, cfo=cfo); rs = select(rows, part='R2', guard=g, order=order, cfo=cfo, target='in_strong', role='focus')
                if ss:
                    tb.append([cfo, ORDER_KO[order], g] + cells(rs, ss))
    L += [table(['주파수 오차', '블록 순서', '보호 부반송파'] + hd, tb), '']
    L += ['## R3. TDMA + FDMA 혼합 (주파수 오차 0.05)', '']
    tb = []
    for s2 in sorted({x['sigma2'] for x in sysrows if x['part'] in ('R1', 'R3')}):
        for lay_, part in [('fdma', 'R1'), ('hybrid2', 'R3'), ('tdma', 'R1')]:
            ss = select(sysrows, part=part, layout=lay_, cfo=0.05, sigma2=s2); rs = select(rows, part=part, layout=lay_, cfo=0.05, sigma2=s2, target='in_strong', role='focus')
            if ss:
                tb.append([db(s2), LAYOUT_KO[lay_]] + cells(rs, ss))
    L += [table(['SNR', '배치'] + hd, tb), '', '## 판정 (± 는 seed 짝 차이의 2×표준오차)', '']
    for s2 in sorted({x['sigma2'] for x in sysrows if x['part'] == 'R1'}):
        b = dict(part='R1', sigma2=s2, cfo=0.0)
        for lay_ in ['fdma', 'fdma_aware', 'ofdma_aware']:
            m, se, n = paired(sysrows, 'mse', b, dict(layout='tdma'), dict(layout=lay_))
            if n:
                L.append(f'- {db(s2)}, 주파수 오차 0: 집계 오차 (TDMA − {LAYOUT_KO[lay_]}) = {pm(m, se, fmt)} (n={n}), '
                         f'라운드 시간 TDMA {fmt(mean(select(sysrows, layout="tdma", **b), "slots"), 2)} vs {fmt(mean(select(sysrows, layout=lay_, **b), "slots"), 2)}')
        b5 = dict(part='R1', sigma2=s2, cfo=0.05, target='in_strong', role='focus')
        for lay_ in ['fdma', 'ofdma_inter', 'ofdma_aware']:
            m, se, n = paired(rows, 'rel', b5, dict(layout=lay_), dict(layout='tdma'))
            if n:
                L.append(f'- {db(s2)}, 주파수 오차 0.05: 약한 shard 파라미터 차이 ({LAYOUT_KO[lay_]} − TDMA) = {pm(m, se, fmt)} (n={n}) → '
                         f'{"섞임이 흔적을 남김" if (n > 1 and m > 2 * se) else "차이 확인 안 됨"}')
    L += ['', '## 읽는 법', '',
          '- 같은 대역폭이면 TDMA, FDMA, OFDMA 의 라운드 시간은 같다 (보호 대역만큼 늘어남). FDMA 계열은 부반송파당 전력이 K 배라 집계 오차와 반복이 작아야 한다 (-10dB 에서 차이가 커야 함).',
          '- 채널 인식 배치는 shard 의 최약 member 가 깊은 페이딩에 빠진 부반송파를 피하므로 집계 오차가 더 작아야 한다. 대신 OFDMA 채널 인식은 조각나서 이웃 shard 가 많아 주파수 오차가 있을 때 간섭이 커질 수 있다.',
          '- 주파수 오차가 없으면 모든 배치에서 간섭은 0 이다 (check.py). 주파수 오차가 있으면 섞어 배치 > 연속 블록 > 보호 대역 순으로 간섭이 줄고, 보호 대역만큼 라운드 시간이 늘어난다.',
          '- 블록 순서를 최약 채널 순으로 하면 이웃 shard 의 도착 크기가 비슷해져, 약한 shard 가 강한 이웃의 간섭을 증폭해 받는 정도 (near-far) 가 줄어야 한다.',
          '- 혼합 (G=2) 은 이웃 블록 수가 줄어 간섭이 FDMA 와 TDMA 사이, 전력 이득도 사이에 있어야 한다.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
