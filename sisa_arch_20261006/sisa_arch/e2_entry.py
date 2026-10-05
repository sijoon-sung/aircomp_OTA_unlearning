"""E2 — ② 채널 순서로 참여 시점 정하기 (AirComp 에서의 slicing).

바꾸는 것: shard 안의 참여 순서 {무작위, 강한 채널 먼저, 약한 채널 먼저(반대 대조), 자기 채널 구간으로 결정}.
           참여 일정 [0,0,s,2s,3s] 는 고정(s=20, 민감도 s=40). 기준선: 전원 동시 참여.
고정: shard 배정, 데이터, 채널 값, 라운드 수.
측정(CPU, 채널 draw 다수): 초기 학습 비용, client 20명 각각의 삭제 비용(평균·최악). 지표는 송신 에너지, UL 채널 사용량, 전체 자원.
측정(GPU, seed 5개): 초기·삭제 후 앙상블 정확도. 1회 전송(R1)이라 조건마다 잡음이 실제로 다르다.
추가: 배정 {무작위, 채널 순위 자르기} x 순서 상호작용, 단계적 삭제 replay 와 처음부터 학습한 reference 의 일치 확인.
"""
import numpy as np
import torch
from .common import (load_split, channels, partition, shard_of, entry_schedule, init_vector, write_json,
                     N_CLIENTS, ORDERS, ORDER_KO, RULE_KO)
from .trainer import LocalTrainer, Job, run_jobs, Evaluator, split_probs
from . import costmodel
from .report import table, fmt, pct, pp, verdict_line

K = 4
ASSIGNS = ['random_split', 'rank_chunk']
COST_KEYS = ['energy', 'ul_uses', 'total_re']

def regimes(eps):
    out = []
    for s2 in [1.0, .1, .01]:
        out += [dict(mode='min_energy', sigma2=s2, eps=eps), dict(mode='fixed_mse', sigma2=s2, eps=eps),
                dict(mode='fixed_mse', sigma2=s2, eps=1e-4), dict(mode='R1', sigma2=s2)]
    return out

def schedule_all(h, groups, entries, T):
    """초기 학습 shard 들 + 20명 삭제 각각의 라운드 통계 목록. 반환 (stats, 'src' 개수)."""
    stats = []
    for c, g in enumerate(groups):
        stats.append(costmodel.schedule_stats(h, g, entries[c], 0, T))
    for u in range(N_CLIENTS):
        c = shard_of(groups, u)
        stats.append(costmodel.schedule_stats(h, groups[c], entries[c], entries[c][u], T, exclude=(u,)))
    return stats

def cpu_part(cfg, log):
    M = cfg['draws']; T = cfg['T']; regs = regimes(cfg['eps'])
    combos = [(a, sp, o) for a in ASSIGNS for sp in [20, 40] for o in ORDERS if not (o == 'all0' and sp == 40)]
    # arr[combo, regime, metric(src, del_mean, del_max) x COST_KEYS, draw]
    arr = np.zeros((len(combos), len(regs), 3, len(COST_KEYS), M), dtype=np.float64)
    for d in range(M):
        sd = f'mc{d}'; base, h = channels(sd, T)
        for a in ASSIGNS:
            groups = partition(a, range(N_CLIENTS), base, K, sd)
            for ci, (aa, sp, o) in enumerate(combos):
                if aa != a:
                    continue
                entries = [entry_schedule(g, base, o, sd, c, T, sp) for c, g in enumerate(groups)]
                stats = schedule_all(h, groups, entries, T)
                for ri, rg in enumerate(regs):
                    mc = costmodel.multi_cost(stats, rg)
                    for mi, k in enumerate(COST_KEYS):
                        v = mc[k]
                        arr[ci, ri, 0, mi, d] = v[:K].sum()
                        arr[ci, ri, 1, mi, d] = v[K:].mean()
                        arr[ci, ri, 2, mi, d] = v[K:].max()
        if (d + 1) % max(1, M // 5) == 0:
            log(f'[E2-a] draws {d + 1}/{M}')
    return combos, regs, arr

def analyze(combos, regs, arr):
    """강한 순 vs 무작위, 약한 순 vs 무작위 비교 (draw 별 짝 비교)."""
    cidx = {c: i for i, c in enumerate(combos)}
    res = []
    for a in ASSIGNS:
        for sp in [20, 40]:
            base_c = cidx[(a, 20, 'all0')]
            for ri, rg in enumerate(regs):
                for mi, k in enumerate(COST_KEYS):
                    s0 = arr[base_c, ri, 0, mi]
                    if not np.all(s0 > 0):
                        continue
                    row = dict(assign=a, spacing=sp, regime=costmodel.regime_name(rg), metric=k)
                    get = lambda o, part: arr[cidx[(a, sp, o)], ri, part, mi]
                    inc = {o: get(o, 0) / s0 - 1 for o in ['random', 'strong', 'weak', 'stable_thr']}
                    dm = {o: get(o, 1) for o in ['random', 'strong', 'weak', 'stable_thr']}
                    dmax = {o: get(o, 2) for o in ['random', 'strong', 'weak', 'stable_thr']}
                    d0 = arr[base_c, ri, 1, mi]
                    for o in inc:
                        row[f'init_increase_{o}'] = float(np.mean(inc[o]))
                        row[f'del_mean_ratio_{o}'] = float(np.mean(dm[o] / d0))
                        row[f'del_max_ratio_{o}'] = float(np.mean(dmax[o] / arr[base_c, ri, 2, mi]))
                    row['p_strong_init_lt_random'] = float(np.mean(inc['strong'] < inc['random'] - 1e-12))
                    row['p_strong_del_lt_random'] = float(np.mean(dm['strong'] < dm['random'] - 1e-9 * dm['random']))
                    row['p_weak_init_gt_random'] = float(np.mean(inc['weak'] > inc['random'] + 1e-12))
                    row['p_weak_del_gt_random'] = float(np.mean(dm['weak'] > dm['random'] + 1e-9 * dm['random']))
                    row['strong_init_increase_reduction'] = float(1 - np.mean(inc['strong']) / np.mean(inc['random'])) if np.mean(inc['random']) > 0 else float('nan')
                    row['strong_del_reduction'] = float(1 - np.mean(dm['strong']) / np.mean(dm['random']))
                    res.append(row)
    return res

def gpu_part(cfg, log):
    T = cfg['T']; quick = cfg['quick']
    assigns = ['random_split'] if quick else ASSIGNS
    orders = ['all0', 'random', 'strong'] if quick else ORDERS
    s2s = [.01] if quick else [.01, 1.0]
    src_rows, del_rows, exact = [], [], []
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device'])
        base, h = channels(seed, T)
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device'])
        ev = Evaluator(tr, data)
        combos = [(a, o, s2) for a in assigns for o in orders for s2 in s2s]
        groups = {a: partition(a, range(N_CLIENTS), base, K, seed) for a in assigns}
        ent = {(a, o): [entry_schedule(g, base, o, seed, c, T) for c, g in enumerate(groups[a])] for a in assigns for o in orders}
        # pass 1: 초기 학습(진입 시점 체크포인트 저장)
        jobs = []
        for a, o, s2 in combos:
            for c, g in enumerate(groups[a]):
                e = ent[(a, o)][c]
                jobs.append(Job(f'src|{a}|{o}|{s2}|{c}', c, w0, e, 0, T, dict(mode='R1', sigma2=s2), save_at=tuple(sorted(set(e.values())))))
        log(f'[E2-b] seed {seed} pass1 jobs={len(jobs)}')
        W, ck, rec = run_jobs(tr, jobs, h, seed, log_every=max(1, T // 4), logger=log)
        ix = {j.name: k for k, j in enumerate(jobs)}
        P = ev.probs(W)
        src_p = {}
        for a, o, s2 in combos:
            pl = [split_probs(P, ix[f'src|{a}|{o}|{s2}|{c}']) for c in range(K)]
            src_p[(a, o, s2)] = pl
            s = ev.ensemble(pl)
            src_rows.append(dict(seed=seed, assign=a, order=o, sigma2=s2, test=s['test'], worst_recall=s['test_worst_recall'],
                                 mean_mse=float(np.mean([rec[ix[f'src|{a}|{o}|{s2}|{c}']]['mean_mse'] for c in range(K)]))))
        # pass 2: 삭제 replay (진입 직전 체크포인트에서 시작)
        jobs2 = []
        for a, o, s2 in combos:
            for u in range(N_CLIENTS):
                c = shard_of(groups[a], u); e = ent[(a, o)][c]; t0 = e[u]
                w_start = ck[ix[f'src|{a}|{o}|{s2}|{c}']][t0]
                rest = {i: v for i, v in e.items() if i != u}
                jobs2.append(Job(f'del|{a}|{o}|{s2}|{u}', c, w_start, rest, t0, T, dict(mode='R1', sigma2=s2)))
        # 일치 확인: 가장 먼저 / 가장 늦게 들어온 client 를 처음부터 제외하고 학습한 reference
        a0, o0 = assigns[0], ('strong' if 'strong' in orders else orders[-1])
        chk = []
        for c, g in enumerate(groups[a0][:1]):
            e = ent[(a0, o0)][c]
            for u in [min(e, key=lambda i: (e[i], i)), max(e, key=lambda i: (e[i], i))]:
                chk.append(u)
                jobs2.append(Job(f'ref|{a0}|{o0}|{s2s[0]}|{u}', c, w0, {i: v for i, v in e.items() if i != u}, 0, T,
                                 dict(mode='R1', sigma2=s2s[0])))
        log(f'[E2-b] seed {seed} pass2 jobs={len(jobs2)}')
        W2, _, rec2 = run_jobs(tr, jobs2, h, seed, log_every=max(1, T // 4), logger=log)
        ix2 = {j.name: k for k, j in enumerate(jobs2)}
        P2 = ev.probs(W2)
        for a, o, s2 in combos:
            accs, worst = [], []
            for u in range(N_CLIENTS):
                c = shard_of(groups[a], u)
                pl = [split_probs(P2, ix2[f'del|{a}|{o}|{s2}|{u}']) if k == c else src_p[(a, o, s2)][k] for k in range(K)]
                r = ev.ensemble(pl); accs.append(r['test']); worst.append(r['test_worst_recall'])
            del_rows.append(dict(seed=seed, assign=a, order=o, sigma2=s2, del_test_mean=float(np.mean(accs)),
                                 del_test_min=float(np.min(accs)), del_worst_recall_mean=float(np.mean(worst))))
        for u in chk:
            a_ = W2[ix2[f'ref|{a0}|{o0}|{s2s[0]}|{u}']]; b_ = W2[ix2[f'del|{a0}|{o0}|{s2s[0]}|{u}']]
            exact.append(dict(seed=seed, client=u, entry=ent[(a0, o0)][shard_of(groups[a0], u)][u], maxabs=float((a_ - b_).abs().max())))
        log(f'[E2-b] seed {seed} done')
    return src_rows, del_rows, exact

def run(cfg, log):
    out = cfg['out'] / 'e2_entry'; out.mkdir(parents=True, exist_ok=True)
    combos, regs, arr = cpu_part(cfg, log)
    np.savez_compressed(out / 'cpu_costs.npz', arr=arr.astype(np.float32))
    ana = analyze(combos, regs, arr)
    src_rows, del_rows, exact = gpu_part(cfg, log)
    write_json(out / 'results.json', dict(combos=combos, regimes=[costmodel.regime_name(r) for r in regs], analysis=ana,
                                           gpu_source=src_rows, gpu_delete=del_rows, exactness=exact, eps=cfg['eps']))

    # ---------------- 판정
    # 주 조건: 잡음 여유가 있으면 전력을 낮추는 min_energy 의 에너지. 반복 전송이 실제로 필요한 설정(G0)이면
    # eps 기준 fixed_mse 의 UL 사용량도 함께 본다. 전력 상한으로 그냥 보내는 R1/fixed_mse(여유 있음)의 에너지는 비교 대상이 아니다.
    prim = [r for r in ana if r['assign'] == 'random_split' and r['spacing'] == 20 and
            ((r['metric'] == 'energy' and r['regime'].startswith('min_energy')) or
             (cfg.get('repetition_matters') and r['metric'] == 'ul_uses' and r['regime'].startswith('fixed_mse') and f"eps{cfg['eps']:g}" in r['regime']))]
    def h2a_ok(r):
        return (r['p_strong_init_lt_random'] >= .8 and r['p_strong_del_lt_random'] >= .8
                and r['p_weak_init_gt_random'] >= .8 and r['p_weak_del_gt_random'] >= .8)
    h2a = bool(prim) and all(h2a_ok(r) for r in prim)
    # 정확도: 강한 순 vs 무작위
    def accdiff(rows, field, a='random_split'):
        out_ = {}
        for s2 in sorted(set(r['sigma2'] for r in rows)):
            d = []
            for seed in sorted(set(r['seed'] for r in rows)):
                g = {r['order']: r[field] for r in rows if r['seed'] == seed and r['sigma2'] == s2 and r['assign'] == a}
                if 'strong' in g and 'random' in g:
                    d.append(g['random'] - g['strong'])
            if d:
                out_[s2] = d
        return out_
    ds, dd = accdiff(src_rows, 'test'), accdiff(del_rows, 'del_test_mean')
    h2b = None
    if ds:
        h2b = all(np.mean(v) <= .01 and max(v) <= .02 for v in ds.values()) and all(np.mean(v) <= .01 and max(v) <= .02 for v in dd.values())
    ex_max = max([e['maxabs'] for e in exact] or [float('nan')])

    L = ['# E2 — 채널 순서로 참여 시점 정하기', '',
         f'K={K} shard x 5명, 일정 [0,0,s,2s,3s] (s=20, 민감도 40), {cfg["T"]}라운드. CPU 해석은 채널 {cfg["draws"]}가지, '
         f'GPU 는 seed {len(cfg["seeds"])}개. eps={fmt(cfg["eps"])} (G0 결과), 비교용으로 이전 실험의 1e-4 도 함께 계산했다.', '',
         '## 비용: 강한 순 vs 무작위 vs 약한 순 (무작위 배정, s=20)', '',
         '"초기 증가"는 전원 동시 참여 대비 초기 학습 비용 증가율, "삭제 비"는 전원 동시 참여 대비 기대 삭제 비용 비율이다. '
         'P(...)는 채널 draw 중 그 관계가 성립한 비율이다.', '']
    tb = []
    for r in ana:
        if r['assign'] != 'random_split' or r['spacing'] != 20:
            continue
        tb.append([r['regime'], r['metric'], pct(r['init_increase_random']), pct(r['init_increase_strong']), pct(r['init_increase_weak']),
                   fmt(r['del_mean_ratio_random']), fmt(r['del_mean_ratio_strong']), fmt(r['del_mean_ratio_weak']),
                   fmt(r['p_strong_init_lt_random']), fmt(r['p_strong_del_lt_random']), fmt(r['p_weak_del_gt_random'])])
    L.append(table(['조건', '지표', '초기 증가 무작위', '초기 증가 강한 순', '초기 증가 약한 순', '삭제 비 무작위', '삭제 비 강한 순',
                    '삭제 비 약한 순', 'P(강한 순 초기<무작위)', 'P(강한 순 삭제<무작위)', 'P(약한 순 삭제>무작위)'], tb))
    L += ['', '## 상호작용: 배정 방식에 따라 순서 효과가 달라지는가 (에너지, s=20)', '']
    tb = []
    for r in ana:
        if r['spacing'] == 20 and r['metric'] == 'energy' and r['regime'].startswith('min_energy'):
            tb.append([RULE_KO[r['assign']], r['regime'], pct(r['strong_init_increase_reduction']).lstrip('+'), pct(r['strong_del_reduction']).lstrip('+'),
                       pct(r['init_increase_stable_thr']), fmt(r['del_mean_ratio_stable_thr'])])
    L.append(table(['배정', '조건', '강한 순의 초기 증가분 감소', '강한 순의 기대 삭제 비용 감소', '채널 구간 판본 초기 증가', '채널 구간 판본 삭제 비'], tb))
    L += ['', '"초기 증가분 감소"는 1 - (강한 순의 초기 증가)/(무작위의 초기 증가) 다. 100% 를 넘으면 강한 순이 전원 동시 참여보다도 초기 비용이 작다는 뜻이다.']
    L += ['', '## 정확도 (GPU, 1회 전송)', '']
    tb = []
    keys = sorted(set((r['assign'], r['order'], r['sigma2']) for r in src_rows))
    for a, o, s2 in keys:
        s = [r for r in src_rows if (r['assign'], r['order'], r['sigma2']) == (a, o, s2)]
        d = [r for r in del_rows if (r['assign'], r['order'], r['sigma2']) == (a, o, s2)]
        tb.append([RULE_KO[a], ORDER_KO[o], f'{s2:g}', fmt(np.mean([x['test'] for x in s]), 4), fmt(np.mean([x['del_test_mean'] for x in d]), 4),
                   fmt(np.mean([x['del_test_min'] for x in d]), 4), fmt(np.mean([x['mean_mse'] for x in s]))])
    L.append(table(['배정', '순서', '잡음 분산', '초기 test', '삭제 후 test 평균', '삭제 후 test 최저', '초기 학습 평균 집계 오차'], tb))
    L += ['', '## 판정', '',
          verdict_line('H2a 강한 순이 초기 증가와 기대 삭제 비용을 모두 줄이고, 약한 순은 반대', h2a,
                       '주 조건(무작위 배정, s=20, min_energy 의 에너지' + (', eps 기준 fixed_mse 의 UL 사용량' if cfg.get('repetition_matters') else '') +
                       ')의 모든 SNR 에서 채널 draw 80% 이상 성립해야 지지. 대상 행: ' +
                       ', '.join(f"{r['regime']}/{r['metric']}={'O' if h2a_ok(r) else 'X'}" for r in prim)),
          verdict_line('H2b 강한 순의 정확도 손실이 무작위 대비 평균 1pp, seed 별 2pp 이하', h2b,
                       '초기: ' + ', '.join(f'잡음 {k:g} 평균 {pp(np.mean(v))} 최대 {pp(max(v))}' for k, v in ds.items()) +
                       ' / 삭제: ' + ', '.join(f'잡음 {k:g} 평균 {pp(np.mean(v))} 최대 {pp(max(v))}' for k, v in dd.items())),
          f'- 단계적 삭제 replay 와 처음부터 학습한 reference 의 파라미터 최대 차이: {fmt(ex_max)} '
          '(같은 계산이므로 부동소수 오차 수준이어야 한다)', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return dict(name='E2', verdicts=[('H2a', h2a), ('H2b', h2b)])
