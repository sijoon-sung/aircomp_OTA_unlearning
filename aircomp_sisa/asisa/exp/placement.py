"""placement — 같은 자원을 쓰는 shard 들을 어떻게 배정해야 하는가: 배정이 바꾸는 것들을 한 표에서 본다.

배정 규칙 (K=4): 무작위 / 채널 정렬 자르기 / 채널 돌려 담기 / 채널 고정 구간 / ID 해시
  (고정 구간과 해시는 각 client 의 자기 정보만으로 정해지므로 shard 크기가 고르지 않을 수 있다)
전력 정렬: 최대 전력 / 가장 약한 shard 에 맞춤 / 공통 수신 크기
고정: PN16 코드 (모든 shard 쌍에 간섭이 있음, shard k = 코드 k), 20dB.
삭제 대상: 2명 이상인 shard 마다 채널이 가장 좋은 member 1명 (지워도 그 shard 의 h_min 은 그대로).
배정이 바꾸는 것 (측정)
  1. shard 간 near-far: shard 별 h_min 의 흩어짐 -> beta 최대/최소, 가장 많이 오염되는 shard 의 간섭/자기
  2. 언러닝 흔적: 삭제 대상이 다른 shard 에 남기는 파라미터 차이 (평균·최대)와 그 난수 기준선
  3. 삭제 비용: 재학습 shard 의 송신 에너지와 로컬 학습 횟수
  4. 노출: 가장 작은 shard 의 인원 (resources 실험의 n 별 노출 표와 같이 읽는다)
  5. 삭제 안정성: 규칙을 u 없이 다시 적용하면 남은 client 사이의 "같은 shard 인가" 관계가 바뀌는 쌍의 비율 (모든 u 평균·최대, 학습 없이 계산).
     0 이 아니면 "u 가 처음부터 없었다면" 의 세계에서 다른 shard 의 구성도 달라져, u 의 shard 만 재학습해서는 정확하지 않다.
  6. 학습 에너지, 정확도
"""
import itertools
import numpy as np
from . import seed_context, db
from .deletion import add_system, measure
from ..channel import partition, RULE_KO
from ..config import N_CLIENTS
from ..fl import train
from ..radio import RadioConfig
from ..util import write_json, fmt, pct, table, select, mean

NAME = 'placement'
K = 4
SIGMA2 = .01
RULES = ['random_split', 'rank_chunk', 'rank_rr', 'bins', 'hash']
ALIGNS = ['maxpow', 'weakest', 'common']
ALIGN_KO = {'maxpow': '최대 전력', 'weakest': '가장 약한 shard 에 맞춤', 'common': '공통 수신 크기'}

def stability(rule, base, seed):
    """u 를 지우고 규칙을 다시 적용했을 때 남은 client 쌍 중 같은-shard 관계가 바뀌는 비율. 반환: (평균, 최대) over u."""
    full = partition(rule, range(N_CLIENTS), base, K, seed)
    where = {i: k for k, g in enumerate(full) for i in g}
    out = []
    for u in range(N_CLIENTS):
        rest = [i for i in range(N_CLIENTS) if i != u]
        again = partition(rule, rest, base, K, seed); w2 = {i: k for k, g in enumerate(again) for i in g}
        pairs = list(itertools.combinations(rest, 2))
        out.append(sum((where[a] == where[b]) != (w2[a] == w2[b]) for a, b in pairs) / len(pairs))
    return float(np.mean(out)), float(np.max(out))

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    T, eps = cfg['T'], cfg['eps']
    rules = ['rank_chunk', 'rank_rr', 'hash'] if cfg['quick'] else RULES
    rows, sysrows = [], []
    for seed in cfg['seeds']:
        c = seed_context(cfg, seed)
        for rule in rules:
            groups = partition(rule, range(N_CLIENTS), c.base, K, seed)
            live = [k for k, g in enumerate(groups) if g]
            hmin = [float(min(c.base[i] for i in g)) if g else float('nan') for g in groups]
            slot_of = {k: k for k in live}
            targets = {f't{k}': int(max(groups[k], key=lambda i: c.base[i])) for k in live if len(groups[k]) >= 2}
            st_mean, st_max = stability(rule, c.base, seed)
            shards, radio, track = [], {}, {}
            for al in ALIGNS:
                rc = RadioConfig(sigma2=SIGMA2, eps=eps, align=al, target=eps, n_nom=N_CLIENTS / K, mux='code', code='pn', L=16)
                add_system(shards, radio, track, al, groups, slot_of, targets, rc, T)
            log(f'[placement] seed {seed} {RULE_KO[rule]}: shard 크기 {[len(g) for g in groups]}, shard 모델 {len(shards)}개')
            W, leds, diags, _ = train(c.tr, shards, c.h, c.w0, seed, lambda s: radio[s], track_of=lambda s: track.get(s, ()), log=log, log_every=T // 4)
            P = c.ev.probs(W); ix = {s.name: j for j, s in enumerate(shards)}
            for al in ALIGNS:
                r, sr = measure(c, W, P, leds, diags, ix, al, groups, slot_of, targets, hmin, None, dict(seed=seed, rule=rule, align=al))
                sr.update(stab_mean=st_mean, stab_max=st_max, min_size=min(len(g) for g in groups), n_targets=len(targets),
                          hmin_spread=max(hmin[k] for k in live) / min(hmin[k] for k in live),
                          del_energy=float(np.mean([sr[f'del_energy_{t}'] for t in targets])), del_compute=float(np.mean([sr[f'del_compute_{t}'] for t in targets])),
                          del_acc=float(np.mean([sr[f'del_acc_{t}'] for t in targets])), ens_dis=float(np.mean([sr[f'ens_dis_{t}'] for t in targets])),
                          rel_mean=float(np.mean([x['rel'] for x in r])), rel_max=float(np.max([x['rel'] for x in r])),
                          rel_yard=float(np.mean([x['rel_yard'] for x in r])), trace=float(np.mean([x['trace'] for x in r])))
                rows += r; sysrows.append(sr)
            del W, P
        log(f'[placement] seed {seed} 끝')
    write_json(out / 'results.json', dict(rows=rows, systems=sysrows, setting=dict(K=K, T=T, eps=eps, seeds=list(cfg['seeds']), rules=rules)))
    L = ['# placement — shard 배정이 바꾸는 것들 (같은 자원, PN16 코드, 20dB)', '',
         f'{cfg["T"]}라운드, seed {len(cfg["seeds"])}개. 삭제 대상은 2명 이상인 shard 마다 채널이 가장 좋은 member 1명.', '',
         '열: h_min 비 = shard 별 최약 채널의 최대/최소 (near-far 의 원인). 최대 간섭/자기 = 가장 많이 오염되는 shard 의 간섭 에너지 ÷ 자기 신호. '
         '파라미터 차이 = 삭제 대상이 다른 shard 모델에 남긴 ||W(u 있음) − W(u 없이 처음부터)|| / ||W(u 없이)|| (모든 대상·받는 shard 평균과 최대), 난수 변동 = 같은 양의 학습 난수 기준선. '
         '삭제 에너지·계산 = 재학습 shard 하나의 송신 에너지와 로컬 학습 횟수. 최소 인원 = 가장 작은 shard (노출은 resources 실험의 n 별 표로 읽는다). '
         '안정성 = 규칙을 u 없이 다시 적용했을 때 남은 client 쌍의 같은-shard 관계가 바뀌는 비율 (0 이면 u 를 지워도 다른 shard 구성이 그대로).', '']
    for al in ALIGNS:
        tb = []
        for rule in rules:
            ss = select(sysrows, rule=rule, align=al)
            if not ss:
                continue
            tb.append([RULE_KO[rule], str(ss[0]['sizes']), fmt(mean(ss, 'min_size'), 1), fmt(mean(ss, 'hmin_spread'), 2), fmt(mean(ss, 'beta_spread'), 2),
                       fmt(mean(ss, 'max_leak_to_own'), 4), fmt(mean(ss, 'rel_mean'), 4), fmt(mean(ss, 'rel_max'), 4), fmt(mean(ss, 'rel_yard'), 4),
                       fmt(mean(ss, 'del_energy')), fmt(mean(ss, 'del_compute'), 0), fmt(mean(ss, 'energy')), pct(mean(ss, 'stab_mean')), pct(mean(ss, 'stab_max')),
                       pct(mean(ss, 'acc_ens')), pct(mean(ss, 'del_acc'))])
        L += [f'## {ALIGN_KO[al]}', '',
              table(['배정', 'shard 크기(첫 seed)', '최소 인원', 'h_min 비', 'β 최대/최소', '최대 간섭/자기', '파라미터 차이 평균', '파라미터 차이 최대', '파라미터 난수 변동',
                     '삭제 에너지', '삭제 로컬 학습', '학습 에너지/라운드', '안정성 평균', '안정성 최대', '앙상블 정확도', '삭제 후 정확도'], tb), '']
    L += ['## 배정과 흔적의 관계', '']
    for al in ALIGNS:
        ss = select(sysrows, align=al)
        if len(ss) > 2:
            x = np.log([s['hmin_spread'] for s in ss]); y = np.log([max(s['max_leak_to_own'], 1e-30) for s in ss]); z = [s['rel_max'] for s in ss]
            L.append(f'- {ALIGN_KO[al]}: log(h_min 비) 와 log(최대 간섭/자기) 의 상관 {np.corrcoef(x, y)[0, 1]:.2f}, log(h_min 비) 와 최대 파라미터 차이의 상관 {np.corrcoef(x, z)[0, 1]:.2f} '
                     f'(seed × 배정 {len(ss)}개)')
    L += ['', '## 읽는 법', '',
          '- 최대 전력에서는 h_min 비가 큰 배정(채널 정렬 자르기, 고정 구간)일수록 가장 오염되는 shard 의 간섭이 커야 한다 (상관이 양수). 도착 크기를 맞추면 이 관계가 약해져야 한다.',
          '- 안정성은 규칙의 성질이다. 고정 구간과 ID 해시는 0 이고, 무작위·정렬·돌려 담기는 0 보다 크다. 안정적인 규칙은 shard 크기가 고르지 않아 최소 인원(노출)과 삭제 비용이 나빠질 수 있다.',
          '- 파라미터 난수 변동은 학습 잡음의 크기에 따라 달라진다 (공통 수신 크기는 집계 오차를 eps 까지 키우므로 난수 변동이 훨씬 크다). 그래서 전력 정렬 사이의 비교는 파라미터 차이 자체로 하고, 차이÷난수는 같은 전력 정렬 안에서만 비교한다.',
          '- 좋은 배정은 h_min 비가 작고(오염 균등), 안정성이 0 이고, 최소 인원이 충분하고, 삭제 비용이 작은 것이다. 표에서 이 네 가지가 한 규칙에서 동시에 만족되는지, 어디서 서로 부딪히는지를 본다.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
