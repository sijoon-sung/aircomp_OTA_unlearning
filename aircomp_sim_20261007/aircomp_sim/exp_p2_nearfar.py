"""P2 — shard 사이 near-far: 전력 정렬과 배정이 shard 간 간섭을 약한 shard 로 몰아주는가, 공통 수신 크기로 없앨 수 있는가.

가설 (cdma.py 의 식)
  shard j 의 member i 가 shard k 의 평균 추정에 새는 양 = g_ik * (beta_k / beta_j) * x_i / n_k
    beta_k: shard k 의 member 가 서버에 도착하는 크기의 역수 (수신 진폭 = x_i / beta_k)
  최대 전력 정렬: beta_k = C / (min_{S_k}|h| sqrt(PD))  ->  beta_k / beta_j = min_{S_j}|h| / min_{S_k}|h|
    채널이 약한 shard 는 다른 shard 의 신호를 (h_min 비)^2 배의 에너지로 받는다 (CDMA 의 near-far 와 같은 구조).
    u 의 신호가 다른 shard 로 새면 그 shard 도 u 에 의존하므로, u 를 지울 때 그 shard 도 재학습해야 정확하다.
  공통 수신 크기: 모든 shard 가 공개 상수 beta0 로 도착 (beta0 는 허용 집계 오차 eps 로 정함) -> 비 = 1.
    전력 상한 때문에 beta0 를 못 맞추는 shard 만 예외 (그 shard 는 최대 전력 + 반복).
설정
  배정 3가지: 채널 정렬 자르기 (약한 client 가 한 shard 에 몰림) / 채널 돌려 담기 (shard 마다 h_min 이 비슷) / 무작위 (P0-A 와 같음)
  전력 정렬 2가지: 최대 전력 정렬 / 공통 수신 크기 (목표 = eps, 명목 인원 N/K)
  코드 3가지: walsh L4 칩 오차 0 (완전 직교 대조군) / walsh L4 칩 오차 상한 0.3 / PN16
  SNR 2가지: 20dB (잡음 여유 있음) / -20dB (전력 상한에 걸려 반복이 필요한 영역)
  코드 배치: 가장 강한 shard(h_min 최대) = 코드 2, 가장 약한 shard = 코드 3, 나머지 두 shard = 코드 0, 1.
    walsh 에 순환 칩 오차를 주면 간섭은 코드 2 <-> 3 사이에서만 생긴다 (0, 1 번은 간섭 없음 = 같은 실행 안의 대조군).
    그래서 모든 배정과 전력 규칙에서 "강한 shard 와 약한 shard 사이의 코드 간섭"이 같고, 수신 크기 비만 달라진다.
  삭제 대상 2명: 가장 강한 shard 의 member (-> 가장 약한 shard 가 증폭된 흔적을 받는 방향)
                가장 약한 shard 의 member (-> 가장 강한 shard 가 감쇠된 흔적을 받는 방향)
    두 대상 모두 shard 안에서 채널이 가장 좋은 member 라서, 지워도 그 shard 의 h_min 은 그대로다.
측정 (미삭제 shard 마다)
  수신 크기 비 beta_k / beta_c, 간섭/자기 신호, u 간섭/자기 신호 (u 한 명이 새는 양), 반복 수,
  흔적 = 미삭제 shard 모델의 (u 있던 원래 학습) vs (u 없이 처음부터 같은 system 으로 학습) 예측 불일치 (칩 잡음·배치 짝 비교),
  난수 변동 = 같은 system 을 다른 잡음·배치 난수로 학습했을 때의 불일치. system 단위로 에너지, 슬롯 수, 집계 오차, 정확도.
"""
import math
import numpy as np
import torch
from .fl import ShardJob, load_split, channels, partition, shard_of, init_vector, key, N_CLIENTS, LocalTrainer, Evaluator, disagreement, write_json
from .cdma import CdmaConfig, run_rounds_cdma
from .exp_p0_problem import fmt, pct, pp, table

K = 4
ASSIGN = ['rank_chunk', 'rank_rr', 'random_split']
ASSIGN_KO = {'rank_chunk': '채널 정렬 자르기', 'rank_rr': '채널 돌려 담기', 'random_split': '무작위'}
CODES = [('walsh', 4, 0.0), ('walsh', 4, 0.3), ('pn', 16, 0.0)]
POWERS = ['maxpow', 'common']
POWER_KO = {'maxpow': '최대 전력 정렬', 'common': '공통 수신 크기'}
SNRS = [0.01, 100.0]
QUICK = dict(assign=['rank_chunk', 'rank_rr'], codes=[('walsh', 4, 0.3)], snrs=[0.01])
TARGET_KO = {'in_strong': '가장 강한 shard 의 member 삭제', 'in_weak': '가장 약한 shard 의 member 삭제'}

def cname(c):
    return f'{c[0]}{c[1]}_d{c[2]:g}'

def cko(c):
    return f'{c[0]} L{c[1]} 칩오차 {c[2]:g}'

def db(s2):
    return f'{round(10 * math.log10(1 / s2))}dB'

def run(cfg, log):
    out = cfg['out'] / 'p2_nearfar'; out.mkdir(parents=True, exist_ok=True)
    T = cfg['T']; eps = cfg['eps']
    assigns = QUICK['assign'] if cfg['quick'] else ASSIGN
    codes = QUICK['codes'] if cfg['quick'] else CODES
    snrs = QUICK['snrs'] if cfg['quick'] else SNRS
    rows, sysrows, layout = [], [], []
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device']); base, h = channels(seed, T)
        unit_delay = np.array([np.random.default_rng(key(seed, 'chipdelay', i)).uniform() for i in range(N_CLIENTS)])
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device']); ev = Evaluator(tr, data)
        acc = lambda p: float((p['test'].argmax(1) == data['ty']).float().mean())
        for rule in assigns:
            groups = partition(rule, range(N_CLIENTS), base, K, seed)
            hmin_g = [float(min(base[i] for i in g)) for g in groups]
            strong = int(np.argmax(hmin_g)); weak = int(np.argmin(hmin_g))
            rest = [k for k in range(K) if k not in (strong, weak)]
            code_of = {rest[0]: 0, rest[1]: 1, strong: 2, weak: 3}
            best = lambda g: int(max(g, key=lambda i: base[i]))
            targets = {'in_strong': best(groups[strong]), 'in_weak': best(groups[weak])}
            focus = {'in_strong': weak, 'in_weak': strong}
            layout.append(dict(seed=seed, assign=rule, groups=groups, hmin=hmin_g, strong=strong, weak=weak, code_of=code_of, targets=targets))
            for c in codes:
                delays = unit_delay * c[2]; jobs = []; sysconf = {}; track = {}
                for s2 in snrs:
                    for pw in POWERS:
                        cc = CdmaConfig(sigma2=s2, code=c[0], L=c[1], delay_max=c[2], align=pw, target=eps, n_nom=N_CLIENTS / K, eps=eps)
                        tag = f'{s2}|{pw}'
                        for sysn, salt in [('src', ''), ('alt', 'alt')]:
                            sysconf[f'{tag}|{sysn}'] = cc
                            for k, g in enumerate(groups):
                                jobs.append(ShardJob(f'{tag}|{sysn}|{k}', list(g), w0, T, system=f'{tag}|{sysn}', tape=code_of[k], salt=salt))
                        track[f'{tag}|src'] = tuple(targets.values())
                        for tn, u in targets.items():
                            cu = shard_of(groups, u)
                            sysconf[f'{tag}|ref|{tn}'] = cc
                            for k, g in enumerate(groups):
                                jobs.append(ShardJob(f'{tag}|ref|{tn}|{k}', [i for i in g if i != u], w0, T, system=f'{tag}|ref|{tn}', tape=code_of[k]))
                            sysconf[f'{tag}|rep|{tn}'] = cc
                            jobs.append(ShardJob(f'{tag}|rep|{tn}', [i for i in groups[cu] if i != u], w0, T, system=f'{tag}|rep|{tn}', tape=code_of[cu]))
                log(f'[P2] seed {seed} {rule} {cname(c)}: shard jobs={len(jobs)}')
                W, leds, diags = run_rounds_cdma(tr, jobs, h, seed, lambda n: sysconf[n], delays, log, max(1, T // 4),
                                                 track_of=lambda n: track.get(n, ()))
                P = ev.probs(W); ix = {j.name: k for k, j in enumerate(jobs)}
                pr = lambda n: {kk: v[ix[n]] for kk, v in P.items()}
                for s2 in snrs:
                    for pw in POWERS:
                        tag = f'{s2}|{pw}'
                        src = {k: pr(f'{tag}|src|{k}') for k in range(K)}; alt = {k: pr(f'{tag}|alt|{k}') for k in range(K)}
                        dsrc = {k: diags[ix[f'{tag}|src|{k}']] for k in range(K)}; lsrc = {k: leds[ix[f'{tag}|src|{k}']] for k in range(K)}
                        nr = max(1, dsrc[0]['rounds'])
                        mbeta = {k: dsrc[k]['beta'] / max(1, dsrc[k]['rounds']) for k in range(K)}
                        ens = ev.ensemble(list(src.values()))
                        sysrow = dict(seed=seed, assign=rule, code=cname(c), sigma2=s2, power=pw,
                                      energy=sum(l.energy for l in lsrc.values()) / nr, slots=dsrc[0]['slots'] / nr,
                                      ul_symbols=lsrc[0].ul_symbols / nr, repeats={k: dsrc[k]['repeats'] / nr for k in range(K)},
                                      mse=float(np.mean([lsrc[k].mse_sum / max(1, lsrc[k].rounds) for k in range(K)])),
                                      beta_spread=max(mbeta.values()) / min(mbeta.values()),
                                      acc_shard=float(np.mean([acc(src[k]) for k in range(K)])), acc_ens=ens['test'],
                                      max_power_ratio=max(l.max_power_ratio for l in lsrc.values()))
                        for tn, u in targets.items():
                            cu = shard_of(groups, u)
                            ref = {k: pr(f'{tag}|ref|{tn}|{k}') for k in range(K)}; rep = pr(f'{tag}|rep|{tn}')
                            sisa = ev.ensemble([rep if k == cu else src[k] for k in range(K)]); refe = ev.ensemble(list(ref.values()))
                            sysrow[f'ens_dis_{tn}'] = disagreement(sisa['_ptest'], refe['_ptest'])
                            sysrow[f'acc_sisa_{tn}'] = sisa['test']; sysrow[f'acc_ref_{tn}'] = refe['test']
                            for k in range(K):
                                if k == cu:
                                    continue
                                d = dsrc[k]; own = max(d['own_energy'], 1e-30)
                                wk = W[ix[f'{tag}|src|{k}']]; wr = W[ix[f'{tag}|ref|{tn}|{k}']]
                                rows.append(dict(seed=seed, assign=rule, code=cname(c), delay_max=c[2], sigma2=s2, power=pw, target=tn, client=u,
                                                 shard=k, affected=cu, code_index=code_of[k], role='focus' if k == focus[tn] else 'other',
                                                 hmin_ratio=hmin_g[cu] / hmin_g[k], beta_ratio=mbeta[k] / mbeta[cu],
                                                 own_energy=d['own_energy'] / max(1, d['rounds']), uleak_energy=d['u_leak'].get(u, 0.0) / max(1, d['rounds']),
                                                 leak_to_own=d['leak_energy'] / own, uleak_to_own=d['u_leak'].get(u, 0.0) / own,
                                                 leak_from_aff_to_own=d['leak_by_src'].get(code_of[cu], 0.0) / own,
                                                 noise_to_own=d['noise_energy'] / own, repeats=d['repeats'] / max(1, d['rounds']),
                                                 trace=disagreement(src[k]['test'], ref[k]['test']), yard=disagreement(src[k]['test'], alt[k]['test']),
                                                 rel=float((wk - wr).norm() / wr.norm()), acc_src=acc(src[k]), acc_ref=acc(ref[k])))
                        sysrows.append(sysrow)
                del W, P
        log(f'[P2] seed {seed} done')
    write_json(out / 'results.json', dict(rows=rows, systems=sysrows, layout=layout,
                                          setting=dict(K=K, T=T, eps=eps, seeds=list(cfg['seeds']), assigns=assigns, codes=[cname(c) for c in codes], snrs=snrs)))
    (out / 'REPORT_KO.md').write_text(report(rows, sysrows, cfg, assigns, codes, snrs), encoding='utf-8')
    return dict(name='P2')

# ---------------------------------------------------------------- 보고서
def _sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]

def _m(rs, f):
    return float(np.mean([r[f] for r in rs])) if rs else float('nan')

def _paired(rows, f, base, a, b, keys=('seed',)):
    """base 조건에서 a 와 b 를 같은 keys 끼리 짝지은 차이 (a - b) 의 평균, 표준오차, 짝 수."""
    A = {tuple(r[k] for k in keys): r[f] for r in _sel(rows, **{**base, **a})}
    B = {tuple(r[k] for k in keys): r[f] for r in _sel(rows, **{**base, **b})}
    d = [A[s] - B[s] for s in A if s in B]
    if not d:
        return float('nan'), float('nan'), 0
    se = float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float('nan')
    return float(np.mean(d)), se, len(d)

def _pm(m, se, f=pct):
    return f'{f(m)} ± {f(2 * se) if not math.isnan(se) else "-"}'

def report(rows, sysrows, cfg, assigns, codes, snrs):
    leaky = [c for c in codes if not (c[0] == 'walsh' and c[2] == 0)]
    L = ['# P2 — shard 사이 near-far: 전력 정렬과 배정에 따른 shard 간 간섭과 언러닝 흔적', '',
         f'shard {K}개가 같은 자원에 동시에 보내고 서버가 코드로 분리한다. {cfg["T"]}라운드, seed {len(cfg["seeds"])}개, 허용 집계 오차 eps = {cfg["eps"]:g}.', '',
         '식 (cdma.py): shard j 의 member i 가 shard k 의 평균 추정에 새는 양 = g_ik · (β_k/β_j) · x_i/n_k. '
         'β 는 member 의 수신 크기의 역수다. 최대 전력 정렬이면 β_k/β_j = min_{S_j}|h| / min_{S_k}|h| 라서 약한 shard 가 증폭된 간섭을 받고, '
         '공통 수신 크기면 β_k/β_j = 1 이다 (전력 상한 때문에 공통 크기를 못 맞추는 shard 만 예외).', '',
         '코드 배치: 가장 강한 shard = 코드 2, 가장 약한 shard = 코드 3. walsh 의 순환 칩 오차는 코드 2↔3 사이에서만 간섭을 만들므로, '
         '모든 배정·전력 규칙에서 두 shard 사이의 코드 간섭 g 는 같고 수신 크기 비만 달라진다. PN16 은 모든 쌍에 간섭이 있다.', '',
         '열 설명: "수신 크기 비" = β_(흔적을 받는 shard) / β_(삭제 대상 shard), 라운드 평균. "u 간섭/자기" = 삭제 대상 u 한 명이 그 shard 로 새는 에너지 ÷ 그 shard 자기 신호 에너지. '
         '"흔적" = 그 shard 모델의 (u 있음) vs (u 없이 처음부터) 예측 불일치. "난수 변동" = 같은 system 을 다른 잡음·배치로 학습한 불일치. "반복" = 그 shard 의 라운드당 평균 반복 수.', '']
    hd = ['코드', '배정', '전력 정렬', '수신 크기 비', 'u 간섭/자기', '간섭/자기(전체)', '반복', '흔적', '난수 변동', '흔적÷난수 변동', 'shard 정확도 차이(u있음−u없음)']
    for tn, title in [('in_strong', '## 1. 증폭 방향: 가장 강한 shard 의 member 를 지울 때, 가장 약한 shard 에 남는 흔적'),
                      ('in_weak', '## 2. 감쇠 방향: 가장 약한 shard 의 member 를 지울 때, 가장 강한 shard 에 남는 흔적')]:
        L += [title, '']
        for s2 in snrs:
            tb = []
            for c in codes:
                for a in assigns:
                    for pw in POWERS:
                        rs = _sel(rows, code=cname(c), assign=a, power=pw, sigma2=s2, target=tn, role='focus')
                        if not rs:
                            continue
                        tr_, yd = _m(rs, 'trace'), _m(rs, 'yard')
                        tb.append([cko(c), ASSIGN_KO[a], POWER_KO[pw], fmt(_m(rs, 'beta_ratio')), fmt(_m(rs, 'uleak_to_own'), 4), fmt(_m(rs, 'leak_to_own'), 4),
                                   fmt(_m(rs, 'repeats'), 2), pct(tr_), pct(yd), fmt(tr_ / yd if yd > 0 else float('nan'), 3), pp(_m(rs, 'acc_src') - _m(rs, 'acc_ref'))])
            L += [f'### {db(s2)}', '', table(hd, tb), '']
    # 같은 실행 안의 대조군
    L += ['## 3. 대조군', '']
    tb = []
    for s2 in snrs:
        for c in codes:
            if c[0] == 'walsh' and c[2] == 0:
                rs = _sel(rows, code=cname(c), sigma2=s2)
                tb.append([db(s2), cko(c) + ' (완전 직교)', '모든 미삭제 shard', fmt(_m(rs, 'leak_to_own'), 4), pct(_m(rs, 'trace')), pct(max([r['trace'] for r in rs], default=float('nan'))), pct(_m(rs, 'yard'))])
            elif c[0] == 'walsh':
                rs = [r for r in _sel(rows, code=cname(c), sigma2=s2) if r['code_index'] in (0, 1)]
                tb.append([db(s2), cko(c), '코드 0·1 shard (간섭 없음)', fmt(_m(rs, 'leak_to_own'), 4), pct(_m(rs, 'trace')), pct(max([r['trace'] for r in rs], default=float('nan'))), pct(_m(rs, 'yard'))])
    L += [table(['SNR', '코드', '대상', '간섭/자기', '흔적 평균', '흔적 최대', '난수 변동'], tb), '',
          '대조군의 흔적은 vmap 배치 계산의 부동소수점 차이가 만드는 바닥이다. 이보다 큰 흔적만 간섭에서 온 것으로 본다.', '']
    # 비용
    L += ['## 4. 비용: 전력 정렬 규칙과 배정에 따른 에너지·슬롯·집계 오차·정확도', '',
          '코드 3가지 평균. "에너지 비" = 같은 seed·배정·코드의 최대 전력 정렬 대비. "슬롯" = 라운드당 슬롯 수 (반복 때문에 1보다 크면 그만큼 자원을 더 씀). '
          '"앙상블 불일치" = 가장 강한 shard 의 member 를 지운 뒤 SISA 앙상블(해당 shard 만 재학습) vs u 없이 전체를 처음부터 학습한 앙상블.', '']
    tb = []
    for s2 in snrs:
        for a in assigns:
            for pw in POWERS:
                ss = _sel(sysrows, sigma2=s2, assign=a, power=pw)
                if not ss:
                    continue
                er = []
                for s in ss:
                    b = _sel(sysrows, seed=s['seed'], assign=a, code=s['code'], sigma2=s2, power='maxpow')
                    if b:
                        er.append(s['energy'] / b[0]['energy'])
                tb.append([db(s2), ASSIGN_KO[a], POWER_KO[pw], fmt(float(np.mean(er)) if er else float('nan'), 4), fmt(_m(ss, 'slots'), 2), fmt(_m(ss, 'beta_spread'), 3),
                           fmt(_m(ss, 'mse'), 4), pct(_m(ss, 'acc_shard')), pct(_m(ss, 'acc_ens')), pct(_m(ss, 'ens_dis_in_strong'))])
    L += [table(['SNR', '배정', '전력 정렬', '에너지 비', '슬롯', 'shard 간 β 최대/최소', '평균 집계 오차', 'shard 평균 정확도', '앙상블 정확도', '앙상블 불일치'], tb), '']
    # 판정
    L += ['## 5. 판정 (수치와 기준을 함께 적는다. ± 는 seed 짝 차이의 2×표준오차)', '']
    for s2 in snrs:
        floor = _m(_sel(rows, code='walsh4_d0', sigma2=s2), 'trace')
        L.append(f'### {db(s2)} (대조군 흔적 바닥 = {pct(floor)})')
        for c in leaky:
            base = dict(code=cname(c), sigma2=s2, target='in_strong', role='focus')
            if not _sel(rows, **base):
                continue
            L.append(f'- {cko(c)}')
            if 'rank_chunk' in assigns and 'rank_rr' in assigns:
                ch = _sel(rows, **base, assign='rank_chunk', power='maxpow'); rr = _sel(rows, **base, assign='rank_rr', power='maxpow')
                if ch and rr:
                    x = _m(ch, 'uleak_to_own') / max(_m(rr, 'uleak_to_own'), 1e-30)
                    y = _m(ch, 'beta_ratio') ** 2 / max(_m(rr, 'beta_ratio') ** 2, 1e-30)
                    L.append(f'  - 증폭 크기 (최대 전력, 정렬 ÷ 돌려 담기): u 간섭/자기 비 = {fmt(x)}, 수신 크기 비 제곱의 비 = {fmt(y)} '
                             f'(자기 신호 크기와 u 의 업데이트 크기도 shard 구성에 따라 달라서 두 값이 같을 필요는 없다. 식 자체는 단위 검증으로 확인함)')
                m, se, n = _paired(rows, 'trace', base, dict(assign='rank_chunk', power='maxpow'), dict(assign='rank_rr', power='maxpow'))
                L.append(f'  - 흔적 차이 (정렬·최대 전력 − 돌려 담기·최대 전력) = {_pm(m, se)} (n={n}) → '
                         f'{"배정이 흔적을 키움" if (n > 1 and m > 2 * se and m > floor) else "차이 확인 안 됨"}')
            if 'rank_chunk' in assigns:
                m, se, n = _paired(rows, 'trace', base, dict(assign='rank_chunk', power='maxpow'), dict(assign='rank_chunk', power='common'))
                L.append(f'  - 흔적 차이 (정렬·최대 전력 − 정렬·공통 수신 크기) = {_pm(m, se)} (n={n}) → '
                         f'{"공통 수신 크기가 흔적을 줄임" if (n > 1 and m > 2 * se and m > floor) else "차이 확인 안 됨"}')
            m, se, n = _paired(rows, 'trace', base, dict(power='maxpow'), dict(power='common'), keys=('seed', 'assign'))
            L.append(f'  - 흔적 차이 (최대 전력 − 공통 수신 크기, 배정 전체) = {_pm(m, se)} (n={n})')
        m, se, n = _paired(sysrows, 'acc_ens', dict(sigma2=s2), dict(power='common'), dict(power='maxpow'), keys=('seed', 'assign', 'code'))
        sl = {pw: _m(_sel(sysrows, sigma2=s2, power=pw), 'slots') for pw in POWERS}
        L.append(f'- 비용: 앙상블 정확도 (공통 수신 크기 − 최대 전력) = {_pm(m, se, pp)} (n={n}), 라운드당 슬롯 = 최대 전력 {fmt(sl["maxpow"], 2)}, 공통 수신 크기 {fmt(sl["common"], 2)}')
        L.append('')
    L += ['## 읽는 법', '',
          '- 가설이 맞다면 20dB 최대 전력 정렬에서 "채널 정렬 자르기"의 수신 크기 비가 가장 크고 (약한 shard 가 증폭된 간섭을 받음), '
          'u 간섭/자기와 흔적도 가장 크다. 같은 배정에 공통 수신 크기를 쓰면 수신 크기 비가 1이 되고 흔적이 돌려 담기 수준으로 내려간다.',
          '- u 간섭/자기의 비가 수신 크기 비 제곱의 비와 같으면, 흔적 차이가 near-far 식에서 나온 것이다.',
          '- 감쇠 방향(2절)은 반대로 강한 shard 가 약한 shard 의 신호를 작게 받으므로 흔적이 작아야 한다.',
          '- -20dB 에서는 약한 shard 가 공통 수신 크기를 못 맞춰 최대 전력 + 반복으로 남는다. 그 shard 에는 증폭이 일부 남고, 반복은 슬롯(자원)을 늘린다. '
          '배정이 약한 client 를 한 shard 에 몰면 반복이 필요한 shard 가 하나로 줄고, 돌려 담으면 모든 shard 가 약한 client 를 하나씩 갖는다.',
          '- 흔적의 크기는 "난수 변동"(같은 조건을 다른 난수로 학습했을 때의 차이)과 비교한다. 대조군 바닥보다 작은 차이는 해석하지 않는다.', '']
    return '\n'.join(L)
