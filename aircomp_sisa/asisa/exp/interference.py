"""interference — 같은 자원에 동시에 보내는 shard 사이의 간섭과 near-far, 그리고 그것이 남기는 언러닝 흔적.

식 (radio.py, mux='code')
  shard j 의 member i 가 shard k 의 평균 추정에 새는 양 = g_ik (beta_k / beta_j) x_i / n_k
  최대 전력 정렬: beta_k / beta_j = min_{S_j}|h| / min_{S_k}|h| -> 채널이 약한 shard 가 (h_min 비)^2 배의 간섭 에너지를 받는다.
  공통 수신 크기: 모든 shard 가 공개 상수 beta0 (eps 와 명목 인원 N/K 로 정함) 로 도착 -> 비 = 1.
  u 의 신호가 다른 shard 로 새면 그 shard 도 u 에 의존하므로, u 의 shard 만 재학습해서는 정확한 언러닝이 되지 않는다.
설정
  배정: 채널 정렬 자르기 (약한 client 가 한 shard 에 몰림) / 채널 돌려 담기 (shard 마다 h_min 이 비슷) / 무작위
  전력 정렬: 최대 전력 / 공통 수신 크기
  코드: walsh L4 칩 오차 0 (완전 직교 = 대조군) / walsh L4 칩 오차 상한 0.3 / PN16
  SNR: 20dB (잡음 여유 있음) / -20dB (전력 상한에 걸려 반복이 필요함)
  코드 배치: 가장 강한 shard (h_min 최대) = 코드 2, 가장 약한 shard = 코드 3, 나머지 = 코드 0, 1.
    walsh 에 순환 칩 오차를 주면 간섭은 코드 2 <-> 3 사이에서만 생긴다 (check.py 에서 확인). 그래서 어느 배정·전력 규칙에서든
    두 shard 사이의 코드 간섭 g 는 같고 수신 크기 비만 달라진다. 코드 0·1 shard 는 같은 실행 안의 대조군이다.
  삭제 대상 2명: 가장 강한 shard 의 member (-> 가장 약한 shard 가 증폭된 흔적을 받는 방향),
                가장 약한 shard 의 member (-> 가장 강한 shard 가 감쇠된 흔적을 받는 방향).
    둘 다 자기 shard 에서 채널이 가장 좋은 member 라서, 지워도 그 shard 의 h_min 은 그대로다.
학습 경로 (system 하나 = 같은 자원을 쓰는 shard 4개)
  src: 원래 학습 / alt: 같은 system 을 다른 배치·잡음 난수로 (학습 난수 변동 기준) /
  ref: u 없이 처음부터 같은 system 으로 (src 와 배치·잡음이 같다) / rep: 삭제 후 SISA 재학습 (u 의 shard 만 단독으로)
측정: 흔적 = 미삭제 shard 의 src vs ref 예측 불일치, 난수 변동 = src vs alt 불일치, 수신 크기 비, u 간섭/자기 신호, 반복, 비용.
"""
import math
import numpy as np
from . import seed_context, db
from ..channel import partition, shard_of, RULE_KO
from ..config import N_CLIENTS
from ..fl import Shard, train, per_round
from ..radio import RadioConfig
from ..trainer import disagreement
from ..util import write_json, fmt, pct, pp, table, select, mean, paired, pm

NAME = 'interference'
K = 4
ASSIGN = ['rank_chunk', 'rank_rr', 'random_split']
ALIGNS = ['maxpow', 'common']
ALIGN_KO = {'maxpow': '최대 전력', 'common': '공통 수신 크기'}
CODES = [('walsh', 4, 0.0), ('walsh', 4, 0.3), ('pn', 16, 0.0)]
SNRS = [0.01, 100.0]
TARGETS = ['in_strong', 'in_weak']

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
            code_of = {rest[0]: 0, rest[1]: 1, strong: 2, weak: 3}
            best = lambda g: int(max(g, key=lambda i: c.base[i]))
            target = {'in_strong': best(groups[strong]), 'in_weak': best(groups[weak])}
            focus = {'in_strong': weak, 'in_weak': strong}
            layout.append(dict(seed=seed, assign=rule, groups=groups, hmin=hmin, code_of=code_of, target=target))
            for code in codes:
                shards, radio, track = [], {}, {}
                for s2 in snrs:
                    for al in ALIGNS:
                        rc = RadioConfig(sigma2=s2, eps=eps, align=al, target=eps, n_nom=N_CLIENTS / K, mux='code',
                                         code=code[0], L=code[1], delay_max=code[2])
                        tag = f'{s2}|{al}'
                        for kind, salt in [('src', ''), ('alt', 'alt')]:
                            radio[f'{tag}|{kind}'] = rc
                            shards += [Shard(f'{tag}|{kind}|{k}', g, f'{tag}|{kind}', slot=code_of[k], salt=salt, T=T) for k, g in enumerate(groups)]
                        track[f'{tag}|src'] = tuple(target.values())
                        for tn, u in target.items():
                            cu = shard_of(groups, u)
                            radio[f'{tag}|ref|{tn}'] = rc; radio[f'{tag}|rep|{tn}'] = rc
                            shards += [Shard(f'{tag}|ref|{tn}|{k}', [i for i in g if i != u], f'{tag}|ref|{tn}', slot=code_of[k], T=T) for k, g in enumerate(groups)]
                            shards.append(Shard(f'{tag}|rep|{tn}', [i for i in groups[cu] if i != u], f'{tag}|rep|{tn}', slot=code_of[cu], T=T))
                log(f'[interference] seed {seed} {RULE_KO[rule]} {cname(code)}: shard 모델 {len(shards)}개')
                W, leds, diags, _ = train(c.tr, shards, c.h, c.w0, seed, lambda s: radio[s], track_of=lambda s: track.get(s, ()),
                                          log=log, log_every=T // 4)
                P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
                pick = lambda n: c.ev.pick(P, ix[n])
                for s2 in snrs:
                    for al in ALIGNS:
                        tag = f'{s2}|{al}'
                        src = {k: pick(f'{tag}|src|{k}') for k in range(K)}; alt = {k: pick(f'{tag}|alt|{k}') for k in range(K)}
                        dg = {k: diags[ix[f'{tag}|src|{k}']] for k in range(K)}; ld = {k: leds[ix[f'{tag}|src|{k}']] for k in range(K)}
                        beta = {k: per_round(dg[k], 'beta') for k in range(K)}; nr = max(1, dg[0]['rounds'])
                        sysrow = dict(seed=seed, assign=rule, code=cname(code), sigma2=s2, align=al,
                                      energy=sum(l.energy for l in ld.values()) / nr, slots=per_round(dg[0], 'slots'),
                                      repeats={k: per_round(dg[k], 'repeats') for k in range(K)},
                                      mse=float(np.mean([l.mse_sum / max(1, l.rounds) for l in ld.values()])),
                                      beta_spread=max(beta.values()) / min(beta.values()),
                                      acc_shard=float(np.mean([c.ev.acc(p) for p in src.values()])), acc_ens=c.ev.ensemble(list(src.values()))['test'],
                                      max_power_ratio=max(l.max_power_ratio for l in ld.values()))
                        for tn, u in target.items():
                            cu = shard_of(groups, u)
                            ref = {k: pick(f'{tag}|ref|{tn}|{k}') for k in range(K)}
                            sisa = c.ev.ensemble([pick(f'{tag}|rep|{tn}') if k == cu else src[k] for k in range(K)]); full = c.ev.ensemble(list(ref.values()))
                            sysrow[f'ens_dis_{tn}'] = disagreement(sisa['ptest'], full['ptest'])
                            for k in range(K):
                                if k == cu:
                                    continue
                                d = dg[k]; own = max(d['own_energy'], 1e-30)
                                ws, wr = W[ix[f'{tag}|src|{k}']], W[ix[f'{tag}|ref|{tn}|{k}']]
                                rows.append(dict(seed=seed, assign=rule, code=cname(code), sigma2=s2, align=al, target=tn, client=u, shard=k, affected=cu,
                                                 code_index=code_of[k], role='focus' if k == focus[tn] else 'other',
                                                 beta_ratio=beta[k] / beta[cu], hmin_ratio=hmin[cu] / hmin[k],
                                                 own_energy=per_round(d, 'own_energy'), uleak_energy=d['u_leak'].get(u, 0.0) / max(1, d['rounds']),
                                                 leak_to_own=d['leak_energy'] / own, uleak_to_own=d['u_leak'].get(u, 0.0) / own,
                                                 noise_to_own=d['noise_energy'] / own, repeats=per_round(d, 'repeats'),
                                                 trace=disagreement(src[k]['test'], ref[k]['test']), yard=disagreement(src[k]['test'], alt[k]['test']),
                                                 rel=float((ws - wr).norm() / wr.norm()), acc_src=c.ev.acc(src[k]), acc_ref=c.ev.acc(ref[k])))
                        sysrows.append(sysrow)
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
         '식: shard j 의 member i 가 shard k 의 평균 추정에 새는 양 = g_ik · (β_k/β_j) · x_i/n_k. β 는 member 수신 크기의 역수. '
         '최대 전력이면 β_k/β_j = min_{S_j}|h| / min_{S_k}|h| (약한 shard 가 증폭된 간섭을 받음), 공통 수신 크기면 1 (전력 상한 때문에 못 맞추는 shard 만 예외).', '',
         '코드 배치: 가장 강한 shard = 코드 2, 가장 약한 shard = 코드 3. walsh 의 순환 칩 오차는 2↔3 사이에서만 간섭을 만든다. PN16 은 모든 쌍에 간섭이 있다.', '',
         '열: 수신 크기 비 = β_(흔적을 받는 shard)/β_(삭제 대상 shard). u 간섭/자기 = 삭제 대상 한 명이 새는 에너지 ÷ 그 shard 자기 신호 에너지. '
         '흔적 = 그 shard 모델의 (u 있음) vs (u 없이 처음부터) 예측 불일치. 난수 변동 = 같은 조건을 다른 배치·잡음으로 학습한 불일치. 반복 = 라운드당 평균 반복 수.', '']
    hd = ['코드', '배정', '전력 정렬', '수신 크기 비', 'u 간섭/자기', '간섭/자기(전체)', '반복', '흔적', '난수 변동', '흔적÷난수 변동', '정확도 차이(u있음−u없음)']
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
                            tr_, yd = mean(rs, 'trace'), mean(rs, 'yard')
                            tb.append([cko(code), RULE_KO[a], ALIGN_KO[al], fmt(mean(rs, 'beta_ratio')), fmt(mean(rs, 'uleak_to_own'), 4), fmt(mean(rs, 'leak_to_own'), 4),
                                       fmt(mean(rs, 'repeats'), 2), pct(tr_), pct(yd), fmt(tr_ / yd if yd > 0 else float('nan')), pp(mean(rs, 'acc_src') - mean(rs, 'acc_ref'))])
            L += [f'### {db(s2)}', '', table(hd, tb), '']
    L += ['## 3. 대조군', '']
    tb = []
    for s2 in snrs:
        for code in codes:
            if code[0] != 'walsh':
                continue
            rs = select(rows, code=cname(code), sigma2=s2) if code[2] == 0 else [r for r in select(rows, code=cname(code), sigma2=s2) if r['code_index'] in (0, 1)]
            what = '완전 직교, 모든 미삭제 shard' if code[2] == 0 else '코드 0·1 shard (간섭 없음)'
            tb.append([db(s2), cko(code), what, fmt(mean(rs, 'leak_to_own'), 4), pct(mean(rs, 'trace')), pct(max([r['trace'] for r in rs], default=float('nan'))), pct(mean(rs, 'yard'))])
    L += [table(['SNR', '코드', '대상', '간섭/자기', '흔적 평균', '흔적 최대', '난수 변동'], tb), '',
          '대조군의 흔적은 vmap 배치 계산의 부동소수점 차이가 만드는 바닥이다. 이보다 큰 흔적만 간섭에서 온 것으로 본다.', '',
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
        floor = mean(select(rows, code='walsh4_d0', sigma2=s2), 'trace')
        L.append(f'### {db(s2)} (대조군 흔적 바닥 = {pct(floor)})')
        for code in leaky:
            base = dict(code=cname(code), sigma2=s2, target='in_strong', role='focus')
            if not select(rows, **base):
                continue
            L.append(f'- {cko(code)}')
            ok = lambda m, se, n: n > 1 and m > 2 * se and not (floor > m)
            if 'rank_chunk' in assigns and 'rank_rr' in assigns:
                ch, rr = select(rows, **base, assign='rank_chunk', align='maxpow'), select(rows, **base, assign='rank_rr', align='maxpow')
                x = mean(ch, 'uleak_to_own') / max(mean(rr, 'uleak_to_own'), 1e-30); y = (mean(ch, 'beta_ratio') / max(mean(rr, 'beta_ratio'), 1e-30)) ** 2
                L.append(f'  - 증폭 크기 (최대 전력, 정렬 ÷ 돌려 담기): u 간섭/자기 비 = {fmt(x)}, 수신 크기 비 제곱의 비 = {fmt(y)} '
                         '(자기 신호와 u 의 update 크기도 shard 구성에 따라 달라 두 값이 같을 필요는 없다)')
                m, se, n = paired(rows, 'trace', base, dict(assign='rank_chunk', align='maxpow'), dict(assign='rank_rr', align='maxpow'))
                L.append(f'  - 흔적 차이 (정렬·최대 전력 − 돌려 담기·최대 전력) = {pm(m, se)} (n={n}) → {"배정이 흔적을 키움" if ok(m, se, n) else "차이 확인 안 됨"}')
            if 'rank_chunk' in assigns:
                m, se, n = paired(rows, 'trace', base, dict(assign='rank_chunk', align='maxpow'), dict(assign='rank_chunk', align='common'))
                L.append(f'  - 흔적 차이 (정렬·최대 전력 − 정렬·공통 수신 크기) = {pm(m, se)} (n={n}) → {"공통 수신 크기가 흔적을 줄임" if ok(m, se, n) else "차이 확인 안 됨"}')
            m, se, n = paired(rows, 'trace', base, dict(align='maxpow'), dict(align='common'), keys=('seed', 'assign'))
            L.append(f'  - 흔적 차이 (최대 전력 − 공통 수신 크기, 배정 전체) = {pm(m, se)} (n={n})')
        m, se, n = paired(sysrows, 'acc_ens', dict(sigma2=s2), dict(align='common'), dict(align='maxpow'), keys=('seed', 'assign', 'code'))
        L += [f'- 비용: 앙상블 정확도 (공통 수신 크기 − 최대 전력) = {pm(m, se, pp)} (n={n}), 라운드당 슬롯 = '
              f'최대 전력 {fmt(mean(select(sysrows, sigma2=s2, align="maxpow"), "slots"), 2)}, 공통 수신 크기 {fmt(mean(select(sysrows, sigma2=s2, align="common"), "slots"), 2)}', '']
    L += ['## 읽는 법', '',
          '- 가설이 맞다면 20dB 최대 전력에서 채널 정렬 자르기의 수신 크기 비·u 간섭/자기·흔적이 가장 크고, 같은 배정에 공통 수신 크기를 쓰면 수신 크기 비가 1, 흔적이 돌려 담기 수준으로 내려간다.',
          '- 감쇠 방향(2절)은 강한 shard 가 약한 shard 의 신호를 작게 받으므로 흔적이 작아야 한다.',
          '- -20dB 에서는 약한 shard 가 공통 수신 크기를 못 맞춰 최대 전력 + 반복으로 남는다. 그 shard 에는 증폭이 일부 남고 반복은 슬롯을 늘린다. '
          '공통 수신 크기는 반복 수를 늘리지 않는다 (못 맞추는 shard 는 최대 전력과 같은 횟수를 반복한다).',
          '- 흔적은 난수 변동과 대조군 바닥과 비교한다. 바닥보다 작은 차이는 해석하지 않는다.', '']
    return '\n'.join(L)
