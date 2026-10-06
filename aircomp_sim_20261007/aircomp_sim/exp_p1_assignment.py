"""P1 — shard 배정에서 학습 비용과 삭제 비용의 트레이드오프가 실제로 나타나는가.

배정 규칙 (K=4):
  random_split  무작위 / rank_chunk 채널 정렬 자르기 / rank_rr 채널 돌려 담기 / bins 채널 고정 구간 / hash ID 해시
  lam{λ}        학습 비용 T 와 기대 삭제 비용 U 의 가중합 (T/T0 + λ U/U0) 을 국소 탐색으로 최소화 (λ=0 은 학습만)
  epsopt0.01    T 가 최적의 1% 이내인 배정 중 U 최소
전송 규칙: 최대 전력(maxpow) / 전력 낮춤(minen). 비용은 실제 장부(심볼·에너지·계산·시간)로 잰다.
삭제: 채널 순위 0/25/50/75/100% 위치의 client 5명을 각각 삭제, 해당 shard 를 처음부터 재학습.
판정: 규칙별 (T, U) 산점에서 T 가 늘 때 U 가 주는 교환이 있는가, 손익분기 삭제 수, 정확도.
"""
import math
import numpy as np
import torch
from .fl import (ShardJob, run_rounds, load_split, channels, partition, shard_of, init_vector, key, N_CLIENTS,
                 LocalTrainer, Evaluator, write_json)
from .radio import RadioConfig, Ledger
from .exp_p0_problem import fmt, pct, pp, table
import sys
from sisa_arch.x1_lifecycle import designed_partitions, compute_times

K = 4
RULES = ['random_split', 'rank_chunk', 'rank_rr', 'bins', 'hash', 'lam0', 'lam0.5', 'lam2', 'epsopt0.01']
RULE_KO = {'random_split': '무작위', 'rank_chunk': '채널 정렬 자르기', 'rank_rr': '채널 돌려 담기', 'bins': '채널 고정 구간', 'hash': 'ID 해시',
           'lam0': '학습 최적(λ=0)', 'lam0.5': 'λ=0.5', 'lam2': 'λ=2', 'epsopt0.01': 'T+1% 내 U 최소'}

def run(cfg, log):
    out = cfg['out'] / 'p1_assignment'; out.mkdir(parents=True, exist_ok=True)
    T = cfg['T']; s2 = .01; modes = ['maxpow', 'minen']
    rules = ['random_split', 'rank_rr', 'lam0', 'lam2'] if cfg['quick'] else RULES
    rows = []
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device']); base, h = channels(seed, T); c_t = compute_times(seed)
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device']); ev = Evaluator(tr, data)
        order = np.argsort(base); dels = [int(order[int(round(q * (N_CLIENTS - 1)))]) for q in [0, .25, .5, .75, 1.0]]
        jobs = []; parts = {}
        for m in modes:
            designed = designed_partitions(base, c_t, m, cfg['eps'], seed)
            rc = RadioConfig(sigma2=s2, eps=cfg['eps'], mode=m)
            for r in rules:
                groups = designed[r] if r in designed else partition(r, range(N_CLIENTS), base, K, seed)
                groups = [list(g) for g in groups]; parts[(m, r)] = groups
                for c, g in enumerate(groups):
                    if g:
                        jobs.append(ShardJob(f'{m}|{r}|src|{c}', g, w0, T, system=f'{m}|{r}', tape=c, radio=rc))
                for u in dels:
                    cu = shard_of(groups, u); rest = [i for i in groups[cu] if i != u]
                    if rest:
                        jobs.append(ShardJob(f'{m}|{r}|del|{u}', rest, w0, T, system=f'{m}|{r}|del{u}', tape=cu, radio=rc))
        log(f'[P1] seed {seed}: shard jobs={len(jobs)}')
        W, _, leds, _ = run_rounds(tr, jobs, h, seed, RadioConfig(), log, max(1, T // 4))
        P = ev.probs(W); ix = {j.name: k for k, j in enumerate(jobs)}
        for m in modes:
            for r in rules:
                groups = parts[(m, r)]
                Tl = Ledger()
                for c, g in enumerate(groups):
                    if g: Tl.add(leds[ix[f'{m}|{r}|src|{c}']])
                Ul = Ledger(); nd = 0; dacc = []
                src = {c: {k: v[ix[f'{m}|{r}|src|{c}']] for k, v in P.items()} for c, g in enumerate(groups) if g}
                acc = ev.ensemble(list(src.values()))['test']
                for u in dels:
                    cu = shard_of(groups, u)
                    if f'{m}|{r}|del|{u}' not in ix:
                        continue
                    Ul.add(leds[ix[f'{m}|{r}|del|{u}']]); nd += 1
                    pl = [{k: v[ix[f'{m}|{r}|del|{u}']] for k, v in P.items()} if c == cu else src[c] for c in src]
                    dacc.append(ev.ensemble(pl)['test'])
                rows.append(dict(seed=seed, mode=m, rule=r, sizes=[len(g) for g in groups], acc=acc, del_acc=float(np.mean(dacc)),
                                 T=Tl.asdict(), U={k: v / max(1, nd) for k, v in Ul.asdict().items()}, n_del=nd))
        log(f'[P1] seed {seed} done')
    write_json(out / 'results.json', dict(rows=rows, dels_rule='채널 순위 0/25/50/75/100%', eps=cfg['eps']))

    # ---------------- 분석: 규칙별 평균, lam0 대비 변화, 손익분기, 파레토
    L = ['# P1 — shard 배정의 학습·삭제 비용 트레이드오프 (실제 장부)', '',
         f'N=20, K={K}, {cfg["T"]}라운드, 20dB, seed {len(cfg["seeds"])}개, eps={cfg["eps"]}. 삭제 대상은 채널 순위 0/25/50/75/100% 위치 5명(각각 삭제, 평균).', '']
    for m in modes:
        L += [f'## {"최대 전력" if m == "maxpow" else "전력 낮춤"}', '']
        tb = []; pts = []
        base_rows = {s: next((r for r in rows if r['seed'] == s and r['mode'] == m and r['rule'] == 'lam0'), None) for s in cfg['seeds']}
        for r in rules:
            sel = [x for x in rows if x['mode'] == m and x['rule'] == r]
            if not sel: continue
            Te = np.mean([x['T']['energy'] for x in sel]); Ue = np.mean([x['U']['energy'] for x in sel])
            Ts = np.mean([x['T']['ul_symbols'] for x in sel]); Us = np.mean([x['U']['ul_symbols'] for x in sel])
            dT, dU, q = [], [], []
            for x in sel:
                b = base_rows[x['seed']]
                if b is None: continue
                dT.append(x['T']['energy'] / b['T']['energy'] - 1); dU.append(x['U']['energy'] / b['U']['energy'] - 1)
                gain = b['U']['energy'] - x['U']['energy']; cost = x['T']['energy'] - b['T']['energy']
                q.append(cost / gain if gain > 0 else (0.0 if cost <= 0 else float('inf')))
            pts.append((r, Te, Ue))
            tb.append([RULE_KO[r], str(sel[0]['sizes']), fmt(Te), fmt(Ue), fmt(Ts), fmt(Us), fmt(np.mean([x['U']['compute_calls'] for x in sel]), 0),
                       pct(np.mean(dT)) if dT else '-', pct(np.mean(dU)) if dU else '-', fmt(float(np.median(q)), 2) if q else '-',
                       pct(np.mean([x['acc'] for x in sel])), pct(np.mean([x['del_acc'] for x in sel]))])
        L.append(table(['배정', 'shard 크기(첫 seed)', '학습 에너지 T', '삭제 에너지 U', '학습 UL 심볼', '삭제 UL 심볼', '삭제 계산',
                        'T 변화(vs λ=0)', 'U 변화(vs λ=0)', '손익분기 삭제 수', '정확도', '삭제 후 정확도'], tb))
        # 파레토: T 오름차순으로 U 가 감소하는 쌍이 있는가
        pts_sorted = sorted(pts, key=lambda p: p[1]); front = []
        best_u = float('inf')
        for name, te, ue in pts_sorted:
            if ue < best_u - 1e-9:
                front.append(name); best_u = ue
        tradeoff = len(front) >= 2
        L += ['', f'파레토 전선(T 오름차순에서 U 가 더 낮아지는 배정): {", ".join(RULE_KO[n] for n in front)}. '
                  f'전선에 배정이 2개 이상이면 학습 비용을 더 써서 삭제 비용을 줄이는 교환이 실제로 존재한다 → {"있음" if tradeoff else "없음"}.', '']
    L += ['손익분기 삭제 수: λ=0(학습만 최적화) 대비 학습에서 더 쓴 에너지를 삭제 에너지 절감으로 회수하는 데 필요한 삭제 횟수. 0 이면 학습도 더 싸진 것, inf 면 회수 불가.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return dict(name='P1')
