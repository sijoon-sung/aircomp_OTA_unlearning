"""interference — 같은 자원에 동시에 보내는 shard 사이의 간섭과 near-far: 도착 크기를 어떻게 맞추면 흔적이 줄어드는가.

식 (radio.py, mux='code')
  shard j 의 member i 가 shard k 의 평균 추정에 새는 양 = g_ik (beta_k / beta_j) x_i / n_k
  최대 전력:                beta_k = beta_min_k (자기 최약 member 가 상한)  -> 비 = min_{S_j}|h| / min_{S_k}|h|, 약한 shard 가 증폭된 간섭을 받는다
  가장 약한 shard 에 맞춤:   모든 shard 가 max_k beta_min_k 로 도착           -> 비 = 1, 도착 크기를 같게 하는 선택 중 잡음 최소
  공통 수신 크기:            모든 shard 가 공개 상수 beta0 (eps, 명목 인원 N/K) 로 도착 -> 비 = 1, 잡음을 허용치까지 키우고 전력 최소
  u 의 신호가 다른 shard 로 새면 그 shard 도 u 에 의존하므로, u 의 shard 만 재학습해서는 정확한 언러닝이 되지 않는다.
설정
  배정: 채널 정렬 자르기 (약한 client 가 한 shard 에 몰림) / 채널 돌려 담기 (shard 마다 h_min 이 비슷) / 무작위
  코드: walsh L4 칩 오차 0 (완전 직교 = 대조군) / walsh L4 칩 오차 상한 0.3 / PN16
  SNR: 20dB (잡음 여유 있음) / -20dB (전력 상한에 걸려 반복이 필요함)
  코드 배치: 가장 강한 shard (h_min 최대) = 코드 2, 가장 약한 shard = 코드 3, 나머지 = 코드 0, 1.
    walsh 에 순환 칩 오차를 주면 간섭은 코드 2 <-> 3 사이에서만 생긴다. 그래서 두 shard 사이의 코드 간섭 g 는 어느 조건에서든 같고
    수신 크기 비만 달라진다. 코드 0·1 shard 는 같은 실행 안의 대조군이다.
  삭제 대상 2명 (둘 다 자기 shard 에서 채널이 가장 좋은 member 라서 지워도 그 shard 의 h_min 은 그대로):
    in_strong = 가장 강한 shard 의 member (-> 가장 약한 shard 가 증폭된 흔적을 받는 방향)
    in_weak   = 가장 약한 shard 의 member (-> 가장 강한 shard 가 감쇠된 흔적을 받는 방향)
측정: deletion.py 참고. 흔적의 크기는 파라미터 차이로 보고, 예측 불일치는 참고로 둔다
  (P2 실측: 예측 불일치는 간섭과 상관 0.26, 파라미터 차이는 0.87. 작은 흔적에도 예측 불일치는 몇 % 로 포화된다).
"""
import numpy as np
from . import seed_context, db
from .deletion import add_system, measure
from ..channel import partition, RULE_KO
from ..config import N_CLIENTS
from ..fl import train
from ..radio import RadioConfig
from ..util import write_json, fmt, pct, pp, table, select, mean, paired, pm

NAME = 'interference'
K = 4
ASSIGN = ['rank_chunk', 'rank_rr', 'random_split']
ALIGNS = ['maxpow', 'weakest', 'common']
ALIGN_KO = {'maxpow': '최대 전력', 'weakest': '가장 약한 shard 에 맞춤', 'common': '공통 수신 크기'}
CODES = [('walsh', 4, 0.0), ('walsh', 4, 0.3), ('pn', 16, 0.0)]
SNRS = [0.01, 100.0]

def cname(c):
    return f'{c[0]}{c[1]}_d{c[2]:g}'

def cko(c):
    return f'{c[0]} L{c[1]} 칩오차 {c[2]:g}'

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    T, eps = cfg['T'], cfg['eps']
    assigns = ['rank_chunk', 'rank_rr'] if cfg['quick'] else ASSIGN
    codes = [('walsh', 4, 0.3)] if cfg['quick'] else CODES
    snrs = [0.01] if cfg['quick'] else SNRS
    rows, sysrows, layout = [], [], []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed)
        for rule in assigns:
            groups = partition(rule, range(N_CLIENTS), c.base, K, seed)
            hmin = [float(min(c.base[i] for i in g)) for g in groups]
            strong, weak = int(np.argmax(hmin)), int(np.argmin(hmin))
            rest = [k for k in range(K) if k not in (strong, weak)]
            slot_of = {rest[0]: 0, rest[1]: 1, strong: 2, weak: 3}
            best = lambda g: int(max(g, key=lambda i: c.base[i]))
            targets = {'in_strong': best(groups[strong]), 'in_weak': best(groups[weak])}
            focus = {'in_strong': weak, 'in_weak': strong}
            layout.append(dict(seed=seed, assign=rule, groups=groups, hmin=hmin, slot_of=slot_of, targets=targets))
            for code in codes:
                shards, radio, track = [], {}, {}
                for s2 in snrs:
                    for al in ALIGNS:
                        rc = RadioConfig(sigma2=s2, eps=eps, align=al, target=eps, n_nom=N_CLIENTS / K, mux='code', code=code[0], L=code[1], delay_max=code[2])
                        add_system(shards, radio, track, f'{s2}|{al}', groups, slot_of, targets, rc, T)
                log(f'[interference] seed {seed} {RULE_KO[rule]} {cname(code)}: shard 모델 {len(shards)}개')
                W, leds, diags, _ = train(c.tr, shards, c.h, c.w0, seed, lambda s: radio[s], track_of=lambda s: track.get(s, ()), log=log, log_every=T // 4)
                P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
                for s2 in snrs:
                    for al in ALIGNS:
                        r, sr = measure(c, W, P, leds, diags, ix, f'{s2}|{al}', groups, slot_of, targets, hmin, focus,
                                        dict(seed=seed, assign=rule, code=cname(code), sigma2=s2, align=al))
                        rows += r; sysrows.append(sr)
                del W, P
        log(f'[interference] seed {seed} 끝')
    write_json(out / 'results.json', dict(rows=rows, systems=sysrows, layout=layout,
                                          setting=dict(K=K, T=T, eps=eps, seeds=list(cfg['seeds']), assigns=assigns, codes=[cname(x) for x in codes], snrs=snrs)))
    (out / 'REPORT_KO.md').write_text(report(rows, sysrows, cfg, assigns, codes, snrs), encoding='utf-8')
    return {}

def report(rows, sysrows, cfg, assigns, codes, snrs):
    leaky = [x for x in codes if not (x[0] == 'walsh' and x[2] == 0)]
    L = ['# interference — 같은 자원을 쓰는 shard 사이의 간섭, near-far, 언러닝 흔적', '',
         f'shard {K}개가 같은 자원에 동시에 보내고 서버가 코드로 분리한다. {cfg["T"]}라운드, seed {len(cfg["seeds"])}개, eps = {fmt(cfg["eps"])}.', '',
         '식: shard j 의 member i 가 shard k 의 평균 추정에 새는 양 = g_ik · (β_k/β_j) · x_i/n_k. β 는 member 도착 크기의 역수.', '',
         '| 전력 정렬 | 도착 크기 | β 비 | 잡음 |', '|---|---|---|---|',
         '| 최대 전력 | shard 마다 자기 최약 member 의 한계 | min_{S_j}|h| / min_{S_k}|h| | 가장 작음 |',
         '| 가장 약한 shard 에 맞춤 | 모든 shard 가 가장 약한 shard 의 한계 | 1 | 강한 shard 의 잡음이 커짐 |',
         '| 공통 수신 크기 | 모든 shard 가 공개 상수 (집계 오차 = eps) | 1 (못 맞추는 shard 만 예외) | 허용치까지 커짐, 전력 최소 |', '',
         '코드 배치: 가장 강한 shard = 코드 2, 가장 약한 shard = 코드 3. walsh 의 순환 칩 오차는 2↔3 사이에서만 간섭을 만든다. PN16 은 모든 쌍에 간섭이 있다.', '',
         '열: 수신 크기 비 = β_(흔적을 받는 shard)/β_(삭제 대상 shard). u 간섭/자기 = 삭제 대상 한 명이 새는 에너지 ÷ 그 shard 자기 신호 에너지. '
         '파라미터 차이 = 그 shard 모델의 ||W(u 있음) − W(u 없이 처음부터)|| / ||W(u 없이)||, 파라미터 난수 변동 = 같은 조건을 다른 배치·잡음으로 학습했을 때의 같은 양. '
         '예측 불일치 = 두 모델의 예측 label 이 다른 test 표본 비율 (작은 흔적에도 포화되므로 참고용).', '']
    hd = ['코드', '배정', '전력 정렬', '수신 크기 비', 'u 간섭/자기', '간섭/자기(전체)', '반복', '파라미터 차이', '파라미터 난수 변동', '차이÷난수', '예측 불일치', '예측 난수 변동']
    for tn, title in [('in_strong', '## 1. 증폭 방향: 가장 강한 shard 의 member 를 지울 때 가장 약한 shard 에 남는 흔적'),
                      ('in_weak', '## 2. 감쇠 방향: 가장 약한 shard 의 member 를 지울 때 가장 강한 shard 에 남는 흔적')]:
        L += [title, '']
        for s2 in snrs:
            tb = []
            for code in codes:
                for a in assigns:
                    for al in ALIGNS:
                        rs = select(rows, code=cname(code), assign=a, align=al, sigma2=s2, target=tn, role='focus')
                        if rs:
                            rl, ry = mean(rs, 'rel'), mean(rs, 'rel_yard')
                            tb.append([cko(code), RULE_KO[a], ALIGN_KO[al], fmt(mean(rs, 'beta_ratio')), fmt(mean(rs, 'uleak_to_own'), 4), fmt(mean(rs, 'leak_to_own'), 4),
                                       fmt(mean(rs, 'repeats'), 2), fmt(rl, 4), fmt(ry, 4), fmt(rl / ry if ry > 0 else float('nan'), 4), pct(mean(rs, 'trace')), pct(mean(rs, 'yard'))])
            L += [f'### {db(s2)}', '', table(hd, tb), '']
    L += ['## 3. 대조군', '']
    tb = []
    for s2 in snrs:
        for code in codes:
            if code[0] != 'walsh':
                continue
            rs = select(rows, code=cname(code), sigma2=s2) if code[2] == 0 else [r for r in select(rows, code=cname(code), sigma2=s2) if r['code_index'] in (0, 1)]
            what = '완전 직교, 모든 미삭제 shard' if code[2] == 0 else '코드 0·1 shard (간섭 없음)'
            tb.append([db(s2), cko(code), what, fmt(mean(rs, 'leak_to_own'), 4), fmt(mean(rs, 'rel'), 4), fmt(max([r['rel'] for r in rs], default=float('nan')), 4),
                       fmt(mean(rs, 'rel_yard'), 4), pct(mean(rs, 'trace'))])
    L += [table(['SNR', '코드', '대상', '간섭/자기', '파라미터 차이 평균', '파라미터 차이 최대', '파라미터 난수 변동', '예측 불일치'], tb), '',
          '대조군의 차이는 vmap 배치 계산의 부동소수점 차이가 만드는 바닥이다. 이보다 큰 차이만 간섭에서 온 것으로 본다.', '',
          '## 4. 비용', '',
          '코드 3가지 평균. 에너지 비 = 같은 seed·배정·코드의 최대 전력 대비. 슬롯 = 라운드당 슬롯 수 (반복 때문에 1보다 크면 그만큼 자원을 더 씀). '
          '앙상블 불일치 = 가장 강한 shard 의 member 를 지운 뒤 SISA 앙상블 (해당 shard 만 재학습) vs u 없이 전체를 처음부터 학습한 앙상블.', '']
    tb = []
    for s2 in snrs:
        for a in assigns:
            for al in ALIGNS:
                ss = select(sysrows, sigma2=s2, assign=a, align=al)
                if not ss:
                    continue
                er = [s['energy'] / b[0]['energy'] for s in ss for b in [select(sysrows, seed=s['seed'], assign=a, code=s['code'], sigma2=s2, align='maxpow')] if b]
                tb.append([db(s2), RULE_KO[a], ALIGN_KO[al], fmt(float(np.mean(er)) if er else float('nan'), 4), fmt(mean(ss, 'slots'), 2), fmt(mean(ss, 'beta_spread')),
                           fmt(mean(ss, 'mse'), 4), pct(mean(ss, 'acc_shard')), pct(mean(ss, 'acc_ens')), pct(mean(ss, 'ens_dis_in_strong'))])
    L += [table(['SNR', '배정', '전력 정렬', '에너지 비', '슬롯', 'shard 간 β 최대/최소', '평균 집계 오차', 'shard 평균 정확도', '앙상블 정확도', '앙상블 불일치'], tb), '',
          '## 5. 판정 (± 는 seed 짝 차이의 2×표준오차)', '']
    for s2 in snrs:
        floor = mean(select(rows, code='walsh4_d0', sigma2=s2), 'rel')
        L.append(f'### {db(s2)} (대조군 파라미터 차이 바닥 = {fmt(floor, 4)})')
        ok = lambda m, se, n: n > 1 and m > 2 * se and not (floor > m)
        for code in leaky:
            base = dict(code=cname(code), sigma2=s2, target='in_strong', role='focus')
            if not select(rows, **base):
                continue
            L.append(f'- {cko(code)}')
            if 'rank_chunk' in assigns and 'rank_rr' in assigns:
                ch, rr = select(rows, **base, assign='rank_chunk', align='maxpow'), select(rows, **base, assign='rank_rr', align='maxpow')
                x = mean(ch, 'uleak_to_own') / max(mean(rr, 'uleak_to_own'), 1e-30); y = (mean(ch, 'beta_ratio') / max(mean(rr, 'beta_ratio'), 1e-30)) ** 2
                L.append(f'  - 증폭 크기 (최대 전력, 정렬 ÷ 돌려 담기): u 간섭/자기 비 = {fmt(x)}, 수신 크기 비 제곱의 비 = {fmt(y)}')
                m, se, n = paired(rows, 'rel', base, dict(assign='rank_chunk', align='maxpow'), dict(assign='rank_rr', align='maxpow'))
                L.append(f'  - 파라미터 차이 (정렬·최대 전력 − 돌려 담기·최대 전력) = {pm(m, se, fmt)} (n={n}) → {"배정이 흔적을 키움" if ok(m, se, n) else "차이 확인 안 됨"}')
            if 'rank_chunk' in assigns:
                for al in ['weakest', 'common']:
                    m, se, n = paired(rows, 'rel', base, dict(assign='rank_chunk', align='maxpow'), dict(assign='rank_chunk', align=al))
                    L.append(f'  - 파라미터 차이 (정렬·최대 전력 − 정렬·{ALIGN_KO[al]}) = {pm(m, se, fmt)} (n={n}) → {"흔적이 줄어듦" if ok(m, se, n) else "차이 확인 안 됨"}')
            for al in ['weakest', 'common']:
                m, se, n = paired(rows, 'rel', base, dict(align='maxpow'), dict(align=al), keys=('seed', 'assign'))
                L.append(f'  - 파라미터 차이 (최대 전력 − {ALIGN_KO[al]}, 배정 전체) = {pm(m, se, fmt)} (n={n})')
            inv = dict(base, target='in_weak')
            for al in ['weakest', 'common']:
                m, se, n = paired(rows, 'rel', inv, dict(align=al), dict(align='maxpow'), keys=('seed', 'assign'))
                L.append(f'  - 감쇠 방향 파라미터 차이 ({ALIGN_KO[al]} − 최대 전력, 배정 전체) = {pm(m, se, fmt)} (n={n}). 양수면 맞춘 대가로 강한 shard 가 더 오염됨')
        for al in ['weakest', 'common']:
            m, se, n = paired(sysrows, 'acc_ens', dict(sigma2=s2), dict(align=al), dict(align='maxpow'), keys=('seed', 'assign', 'code'))
            L.append(f'- 비용: 앙상블 정확도 ({ALIGN_KO[al]} − 최대 전력) = {pm(m, se, pp)} (n={n}), 평균 집계 오차 {fmt(mean(select(sysrows, sigma2=s2, align=al), "mse"), 4)} '
                     f'(최대 전력 {fmt(mean(select(sysrows, sigma2=s2, align="maxpow"), "mse"), 4)}), 라운드당 슬롯 {fmt(mean(select(sysrows, sigma2=s2, align=al), "slots"), 2)}')
        L.append('')
    L += ['## 읽는 법', '',
          '- 가설이 맞다면 20dB 최대 전력에서 채널 정렬 자르기의 수신 크기 비·u 간섭/자기·파라미터 차이가 가장 크고, 도착 크기를 맞추면 (가장 약한 shard 에 맞춤, 공통 수신 크기) 수신 크기 비가 1이 되고 파라미터 차이가 크게 준다.',
          '- 가장 약한 shard 에 맞춤은 도착 크기를 같게 하는 선택 중 잡음이 가장 작다. 공통 수신 크기는 잡음을 허용치까지 키우는 대신 전력을 가장 적게 쓴다. 두 방식과 최대 전력의 정확도·집계 오차 차이가 도착 크기를 맞추는 대가다.',
          '- 감쇠 방향(2절)은 최대 전력에서 강한 shard 가 약한 shard 의 신호를 작게 받으므로 흔적이 작다. 도착 크기를 맞추면 이 감쇠가 사라져 흔적이 커질 수 있다 (간섭이 고르게 퍼짐).',
          '- -20dB 에서는 약한 shard 가 공통 수신 크기를 못 맞춰 최대 전력 + 반복으로 남는다. 반복 슬롯에는 다른 shard 가 보내지 않으므로 간섭이 희석된다.',
          '- 파라미터 난수 변동은 학습 잡음의 크기에 따라 달라진다 (공통 수신 크기는 집계 오차를 eps 까지 키우므로 난수 변동이 훨씬 크다). 그래서 전력 정렬 사이의 비교는 파라미터 차이 자체로 하고, 차이÷난수는 같은 전력 정렬 안에서만 비교한다.',
          '- 파라미터 차이는 파라미터 난수 변동과 대조군 바닥과 비교한다.', '']
    return '\n'.join(L)
