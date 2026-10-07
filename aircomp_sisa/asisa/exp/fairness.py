"""fairness (C) — 남의 삭제 비용을 누가 내는가.

중앙 SISA 에서 재학습 비용은 서버가 낸다. FL 에서는 u 의 shard 에 남은 사람들이 자기 계산과 송신 에너지로 u 의 삭제를 치른다.
u 본인은 떠나므로 아무것도 내지 않는다. 삭제가 특정 shard 에 몰리면 그 shard 사람들만 계속 손해를 본다.
lifecycle (E2b) 과 같은 사건 흐름 (160 라운드 학습 후 20 라운드마다 삭제 70% / 참여 30%) 을 사람별로 다시 센다.
  낸 비용 (client i) = i 가 다시 돈 라운드 수의 합 (계산·송신 에너지 모두 이에 비례)
시나리오
  uniform     삭제 요청자가 현재 참여자 중 무작위
  one_shard   삭제 요청자가 한 묶음 (ID 해시 값 0 인 사람들) 에서만 나옴 — 같은 사람 집합을 모든 방법에 씀
방법: shard 없음 / 채널 돌려 담기 / ID 해시 / 해시 + 병합 / 해시 + 병합 + slicing
측정: 전체 낸 비용, 삭제를 요청한 적 없는 사람이 낸 비율, 한 사람 최대 / 평균, 지니 계수, 상위 5명이 낸 비율.
"""
import math
import numpy as np
from .lifecycle import assign, active, N_MIN, T0, GAP, EVENTS, P_DEL, SLICES, METHOD_KO
from ..config import N_CLIENTS
from ..util import key, write_json, fmt, pct, table, select, mean

NAME = 'fairness'
METHODS = ['full', 'rank_rr', 'hash', 'hash_merge', 'hash_merge_slice']
SCENARIOS = ['uniform', 'one_shard']
SCEN_KO = {'uniform': '무작위 요청자', 'one_shard': '한 묶음에서만 요청'}
DRAWS = 200

def retrain_groups(before, after, entry, t_now, slicing):
    """lifecycle.retrain_cost 와 같은 규칙, 다만 (shard 구성원, 다시 도는 라운드) 를 돌려준다."""
    out, bset = [], set(before)
    for g in after:
        if g in bset:
            continue
        old = max(before, key=lambda b: len(b & g), default=frozenset())
        added, removed = (g - old, old - g) if old & g else (g, frozenset())
        if not removed and all(entry[i] == t_now for i in added):
            start = t_now
        elif slicing:
            start = min(entry[i] for i in (added | removed))
        else:
            start = 0
        out.append((g, t_now - start))
    return out

def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    n = len(x); return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))

def simulate(method, scen, seed, base0):
    K = 4; slicing = 'slice' in method
    rng = np.random.default_rng(key(seed, 'lifecycle_events'))
    base = dict(enumerate(base0)); ids = list(range(N_CLIENTS))
    entry = {i: ((key(seed, i, 'slice') % SLICES) * (T0 // SLICES) if slicing else 0) for i in ids}
    pool0 = [i for i in ids if key(seed, i, 'hash') % K == 0]            # one_shard 시나리오의 요청자 집합 (방법과 무관하게 같은 사람들)
    order = sorted(ids, key=lambda i: (entry[i], i))
    state = active(method, assign(method, order, base, seed, K))
    paid = {i: 0.0 for i in ids}; requested = set(); nid = N_CLIENTS
    for e in range(EVENTS):
        t_now = T0 + (e + 1) * GAP
        cand = [i for i in order if (scen == 'uniform' or i in pool0)]
        if rng.random() < P_DEL and len(order) > 4 and cand:
            u = int(rng.choice(cand)); order = [i for i in order if i != u]; requested.add(u)
        else:
            v = nid; nid += 1; base[v] = float(10 ** (rng.uniform(-20, 0) / 20)); entry[v] = t_now; order = order + [v]; paid[v] = 0.0
        new = active(method, assign(method, order, base, seed, K))
        for g, r in retrain_groups(state, new, entry, t_now, slicing):
            for i in g:
                paid[i] += r
        state = new
    v = np.array([paid[i] for i in paid]); nonreq = sum(paid[i] for i in paid if i not in requested)
    top = np.sort(v)[::-1][:5].sum()
    return dict(method=method, scen=scen, seed=seed, total=float(v.sum()), nonrequester_share=float(nonreq / v.sum()) if v.sum() else float('nan'),
                max_over_mean=float(v.max() / v.mean()) if v.mean() else float('nan'), gini=gini(v), top5_share=float(top / v.sum()) if v.sum() else float('nan'),
                zero_payers=float((v == 0).mean()), n_requested=len(requested))

def run(cfg, log):
    out = cfg['out'] / NAME; out.mkdir(parents=True, exist_ok=True)
    methods = ['full', 'hash', 'hash_merge'] if cfg['quick'] else METHODS
    draws = 20 if cfg['quick'] else DRAWS
    sims = []
    for sd in list(cfg['seeds']) + [3 * 10 ** 6 + d for d in range(draws)]:
        base0 = 10 ** (np.random.default_rng(key(sd, 'slow_channel')).uniform(-20, 0, N_CLIENTS) / 20)
        sims += [simulate(m, sc, sd, base0) for m in methods for sc in SCENARIOS]
    log(f'[fairness] 계산 끝 (사건 흐름 {len(cfg["seeds"]) + draws}개 × 방법 {len(methods)} × 시나리오 {len(SCENARIOS)})')
    write_json(out / 'results.json', dict(sims=sims, setting=dict(T0=T0, gap=GAP, events=EVENTS, p_del=P_DEL, n_min=N_MIN, draws=draws, methods=methods)))
    L = ['# fairness (C) — 남의 삭제 비용을 누가 내는가', '',
         f'초기 client {N_CLIENTS}명, {T0} 라운드 학습 후 {GAP} 라운드마다 사건 {EVENTS}개 (삭제 {int(P_DEL * 100)}%). 사건 흐름 {len(cfg["seeds"]) + draws}개 평균. '
         '낸 비용 = 그 사람이 다시 돈 라운드 수 (계산과 송신 에너지가 이에 비례). 삭제 요청자 본인은 떠나므로 내지 않는다.', '']
    for scen in SCENARIOS:
        L += [f'## {SCEN_KO[scen]}', '']
        tb = []
        for m in methods:
            s = select(sims, method=m, scen=scen)
            tb.append([METHOD_KO[m], fmt(mean(s, 'total'), 0), pct(mean(s, 'nonrequester_share')), fmt(mean(s, 'max_over_mean'), 2), fmt(mean(s, 'gini'), 3),
                       pct(mean(s, 'top5_share')), pct(mean(s, 'zero_payers'))])
        L += [table(['방법', '전체 낸 라운드', '요청한 적 없는 사람이 낸 비율', '최대 / 평균', '지니', '상위 5명 비율', '한 번도 안 낸 사람'], tb), '']
    L += ['## 읽는 법', '',
          '- shard 없음은 모두가 모든 삭제를 같이 치르므로 지니가 0 에 가깝고 최대/평균이 1 이다. 비용은 크지만 고르게 든다.',
          '- 해시 계열은 전체 비용이 작은 대신 u 의 shard 사람들만 낸다. 삭제가 한 묶음에 몰리면 (one_shard) 그 사람들의 최대/평균과 상위 5명 비율이 커져야 한다.',
          '- "요청한 적 없는 사람이 낸 비율" 이 100% 에 가까우면, 삭제 비용은 전부 요청하지 않은 사람이 내는 것이다. 요청자에게 비용을 지우는 설계 '
          '(예: 요청자의 마지막 송신이 재학습 첫 라운드를 대신 보냄) 가 있는지가 다음 질문이다.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return {}
