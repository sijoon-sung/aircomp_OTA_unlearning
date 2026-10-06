"""X1 — 수명 비용과 손익분기: 언러닝을 고려한 배정은 평상시 학습을 얼마나 손해 보고, 삭제 몇 번이면 회수되는가.

흩어져 있던 결과(10/3 K 비교, E1 배정 비용, E3 K 비교, CPU 언러닝 항 검증)를 G0 의 eps*=10 기준으로 한곳에 모은다.

(A) CPU, N=20, 채널 draw 다수
  - 전송 규칙 2가지: 최대 전력(maxpow, eps 기준 반복 포함) / 잡음 한계까지 전력 낮춤(minen)
  - K in {1,2,4,5,10} (무작위 배정): 초기 학습 비용 T, 삭제 1회 기대 비용 U, 계산량, 재학습 지연
  - K=4 배정 방법: 무작위 / 채널 정렬 자르기 / 채널 돌려 담기 / 언러닝 고려(lambda 별) / 학습 eps-최적 집합 안에서 언러닝 최소
  - 수명 비용 T + q U, 손익분기 삭제 수 q* = (T_a - T_b) / (U_b - U_a)
  - 재학습 지연: 기기별 계산 시간(로그정규, 채널과 독립). 학습 지연은 전체 최댓값이라 배정과 무관하다.
(B) GPU, seed 5개: 위 구성의 초기·삭제 후 정확도 (명목 20dB)
"""
import math
import numpy as np
from .common import (load_split, channels, slow_channel, partition, shard_of, init_vector, write_json, key, N_CLIENTS, D_PARAMS)
from .trainer import LocalTrainer, Job, run_jobs, Evaluator, split_probs
from . import costmodel
from .report import table, fmt, pct, pp, verdict_line

KS = [1, 2, 4, 5, 10]
SIGMA2 = .01
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
EPS_OPT = [0.01, 0.05]
METHODS = ['random_split', 'rank_chunk', 'rank_rr']
METHOD_KO = {'random_split': '무작위', 'rank_chunk': '채널 정렬 자르기', 'rank_rr': '채널 돌려 담기'}

def modes(eps):
    return {'maxpow': dict(mode='fixed_mse', sigma2=SIGMA2, eps=eps), 'minen': dict(mode='min_energy', sigma2=SIGMA2, eps=eps)}
MODE_KO = {'maxpow': '최대 전력', 'minen': '전력 낮춤'}

def compute_times(seed):
    return np.array([np.random.default_rng(key(seed, 'compute', i)).lognormal(0, 0.5) for i in range(N_CLIENTS)])

# ---------------- 정적 채널 기반 탐색용 비용 (라운드당)
def g_energy(mode, h, g, eps):
    g = list(g); n = len(g)
    if n == 0:
        return 0.0
    hs = h[g]; m1 = SIGMA2 / (n * n * hs.min() ** 2)
    if mode == 'maxpow':
        R = max(1.0, math.ceil(m1 / eps - 1e-9)); mse = m1 / R
    else:
        R = 1.0 if m1 <= eps else math.ceil(m1 / eps - 1e-9); mse = eps if m1 <= eps else m1 / R
    return D_PARAMS * SIGMA2 * float((1 / hs ** 2).sum()) / (n * n * mse)

class Scorer:
    def __init__(self, mode, h, c, eps):
        self.mode, self.h, self.c, self.eps = mode, h, c, eps; self.memo = {}
    def group(self, g):
        k = tuple(sorted(g))
        if k not in self.memo:
            T = g_energy(self.mode, self.h, k, self.eps)
            U = sum(g_energy(self.mode, self.h, [x for x in k if x != u], self.eps) for u in k)
            S = sum(max(self.c[x] for x in k if x != u) for u in k) if len(k) > 1 else 0.0
            self.memo[k] = (T, U, S)
        return self.memo[k]
    def part(self, groups):
        t = u = s = 0.0
        for g in groups:
            a, b, cc = self.group(g); t += a; u += b; s += cc
        return t, u / N_CLIENTS, s / N_CLIENTS

def local_search(sc, start, obj, passes=25):
    groups = [list(g) for g in start]; best = obj(*sc.part(groups))
    for _ in range(passes):
        improved = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                for i in list(groups[a]):
                    for j in list(groups[b]):
                        ga = [x for x in groups[a] if x != i] + [j]; gb = [x for x in groups[b] if x != j] + [i]
                        cand = groups[:a] + [ga] + groups[a + 1:b] + [gb] + groups[b + 1:]
                        v = obj(*sc.part(cand))
                        if v < best - 1e-12:
                            best, groups, improved = v, cand, True
                            break
                    if improved: break
                if improved: break
            if improved: break
        if not improved:
            break
    return [sorted(g) for g in groups]

def designed_partitions(h_slow, c, mode, eps, sd, K=4):
    """반환: {이름: 배정}. 탐색은 정적 채널로, 보고 비용은 페이딩 포함 해석식으로 다시 계산."""
    sc = Scorer(mode, h_slow, c, eps); allc = list(range(N_CLIENTS))
    out = {m: partition(m, allc, h_slow, K, sd) for m in METHODS}
    T0, U0, _ = sc.part(out['random_split'])
    starts = [out['rank_rr'], out['random_split'], out['rank_chunk']]
    for lam in LAMBDAS:
        cands = [local_search(sc, s, lambda t, u, s_: t / T0 + lam * u / U0) for s in starts]
        out[f'lam{lam:g}'] = min(cands, key=lambda p: (lambda t, u, s_: t / T0 + lam * u / U0)(*sc.part(p)))
    Tstar = sc.part(out['lam0'])[0]
    for e in EPS_OPT:   # 학습 비용이 최적의 (1+e) 이내인 배정 중 언러닝 비용 최소
        pen = lambda t, u, s_, e=e: u / U0 + 1e3 * max(0.0, t / Tstar - 1 - e)
        cands = [local_search(sc, s, pen) for s in [out['lam0']] + starts]
        out[f'epsopt{e:g}'] = min(cands, key=lambda p: pen(*sc.part(p)))
    # 재학습 지연만 고려 (학습 비용을 같은 eps 범위 안에서)
    pen = lambda t, u, s_: s_ + 1e3 * max(0.0, t / Tstar - 1 - EPS_OPT[0])
    out['latopt'] = min([local_search(sc, s, pen) for s in [out['lam0']] + starts], key=lambda p: pen(*sc.part(p)))
    return out

def full_cost(h, c, groups, rg, T):
    """페이딩 포함: 초기 학습 비용, 삭제 1회 기대 비용(20명 균등), 계산, 재학습 지연."""
    stats_src = [costmodel.schedule_stats(h, g, {i: 0 for i in g}, 0, T) for g in groups if g]
    stats_del = []
    for u in range(N_CLIENTS):
        g = groups[shard_of(groups, u)]
        stats_del.append(costmodel.schedule_stats(h, g, {i: 0 for i in g}, 0, T, exclude=(u,)))
    ms = costmodel.multi_cost(stats_src, rg); md = costmodel.multi_cost(stats_del, rg)
    lat = []
    for u in range(N_CLIENTS):
        rest = [x for x in groups[shard_of(groups, u)] if x != u]
        lat.append(T * (max(c[x] for x in rest) if rest else 0.0))
    return dict(T_energy=float(ms['energy'].sum()), T_uses=float(ms['ul_uses'].sum()), T_compute=float(ms['compute_calls'].sum()),
                T_latency=float(T * c.max()),
                U_energy=float(md['energy'].mean()), U_uses=float(md['ul_uses'].mean()), U_compute=float(md['compute_calls'].mean()),
                U_latency=float(np.mean(lat)), sizes=[len(g) for g in groups])

def cpu_part(cfg, log):
    M = cfg['draws_x1']; T = cfg['T']; eps = cfg['eps']; md = modes(eps)
    rows = []
    for d in range(M):
        sd = f'mc{d}'; base, h = channels(sd, T); c = compute_times(sd)
        for mname, rg in md.items():
            for K in KS:
                g = partition('random_split', range(N_CLIENTS), base, K, sd) if K > 1 else [list(range(N_CLIENTS))]
                rows.append(dict(draw=d, mode=mname, K=K, method='random_split', **full_cost(h, c, g, rg, T)))
            if d < cfg['search_draws']:
                parts = designed_partitions(base, c, mname, eps, sd)
                for name, g in parts.items():
                    if name == 'random_split':
                        continue
                    rows.append(dict(draw=d, mode=mname, K=4, method=name, **full_cost(h, c, g, rg, T)))
        if (d + 1) % max(1, M // 5) == 0:
            log(f'[X1-a] draws {d + 1}/{M}')
    return rows

def summarize(rows, search_draws):
    S = {}
    def mean(sel, f):
        return float(np.mean([r[f] for r in sel])) if sel else float('nan')
    for mode in ['maxpow', 'minen']:
        # K 비교 (무작위 배정, 모든 draw)
        kt = {}
        for K in KS:
            sel = [r for r in rows if r['mode'] == mode and r['K'] == K and r['method'] == 'random_split']
            kt[K] = {f: mean(sel, f) for f in ['T_energy', 'U_energy', 'T_uses', 'U_uses', 'T_compute', 'U_compute', 'U_latency']}
        # 단일 모델 대비 손익분기 (draw 별 계산 후 중앙값)
        be = {}
        for K in KS[1:]:
            q = []
            for d in sorted(set(r['draw'] for r in rows)):
                a = next((r for r in rows if r['draw'] == d and r['mode'] == mode and r['K'] == K and r['method'] == 'random_split'), None)
                b = next((r for r in rows if r['draw'] == d and r['mode'] == mode and r['K'] == 1), None)
                if a and b:
                    dT = a['T_energy'] - b['T_energy']; dU = b['U_energy'] - a['U_energy']
                    q.append(dT / dU if dU > 0 else (0.0 if dT <= 0 else float('inf')))
            be[K] = dict(median=float(np.median(q)), p_never=float(np.mean([math.isinf(x) for x in q])),
                         p_free=float(np.mean([x <= 0 for x in q])))
        # K=4 배정 방법 비교 (탐색 draw 만, draw 별 짝 비교)
        draws = sorted(set(r['draw'] for r in rows if r['mode'] == mode and r['method'] == 'lam0'))
        mt = {}
        names = METHODS + [f'lam{l:g}' for l in LAMBDAS] + [f'epsopt{e:g}' for e in EPS_OPT] + ['latopt']
        for name in names:
            dT, dU, dL, q = [], [], [], []
            for d in draws:
                a = next(r for r in rows if r['draw'] == d and r['mode'] == mode and r['K'] == 4 and r['method'] == name)
                b = next(r for r in rows if r['draw'] == d and r['mode'] == mode and r['K'] == 4 and r['method'] == 'lam0')
                dT.append(a['T_energy'] / b['T_energy'] - 1); dU.append(a['U_energy'] / b['U_energy'] - 1)
                dL.append(a['U_latency'] / b['U_latency'] - 1)
                gainU = b['U_energy'] - a['U_energy']; costT = a['T_energy'] - b['T_energy']
                q.append(costT / gainU if gainU > 0 else (0.0 if costT <= 0 else float('inf')))
            mt[name] = dict(T_change=float(np.mean(dT)), U_change=float(np.mean(dU)), lat_change=float(np.mean(dL)),
                            breakeven_median=float(np.median(q)), p_breakeven_le1=float(np.mean([x <= 1 for x in q])),
                            p_breakeven_le10=float(np.mean([x <= 10 for x in q])))
        S[mode] = dict(K=kt, breakeven_vs_K1=be, methods=mt)
    return S

def gpu_part(cfg, log, designs):
    T = cfg['T']; eps = cfg['eps']; md = modes(eps); rows = []
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device']); base, h = channels(seed, T); c = compute_times(seed)
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device']); ev = Evaluator(tr, data)
        rand4 = partition('random_split', range(N_CLIENTS), base, 4, seed)
        dels = [g[0] for g in rand4]          # 모든 구성에서 같은 4명을 삭제
        confs = []
        for mname in md:
            for K in KS:
                confs.append((mname, f'K{K}', partition('random_split', range(N_CLIENTS), base, K, seed) if K > 1 else [list(range(N_CLIENTS))]))
            parts = designs(base, c, mname, eps, seed)
            for name in ['rank_chunk', 'rank_rr', 'lam0', 'lam1', 'epsopt0.01']:
                confs.append((mname, name, parts[name]))
        jobs = []
        for mname, name, groups in confs:
            rg = md[mname]
            for ci, g in enumerate(groups):
                if g:
                    jobs.append(Job(f'{mname}|{name}|src|{ci}', ci, w0, {i: 0 for i in g}, 0, T, rg))
            for u in dels:
                ci = shard_of(groups, u); rest = [i for i in groups[ci] if i != u]
                if rest:
                    jobs.append(Job(f'{mname}|{name}|del{u}', ci, w0, {i: 0 for i in rest}, 0, T, rg))
        log(f'[X1-b] seed {seed}: jobs={len(jobs)}')
        W, _, rec = run_jobs(tr, jobs, h, seed, log_every=max(1, T // 4), logger=log)
        P = ev.probs(W); ix = {j.name: k for k, j in enumerate(jobs)}
        for mname, name, groups in confs:
            src = {ci: split_probs(P, ix[f'{mname}|{name}|src|{ci}']) for ci, g in enumerate(groups) if g}
            s = ev.ensemble(list(src.values()))
            da = []
            for u in dels:
                ci = shard_of(groups, u)
                pl = [src[k] for k in src if k != ci]
                if f'{mname}|{name}|del{u}' in ix:
                    pl.append(split_probs(P, ix[f'{mname}|{name}|del{u}']))
                da.append(ev.ensemble(pl)['test'])
            rows.append(dict(seed=seed, mode=mname, config=name, test=s['test'], worst_recall=s['test_worst_recall'],
                             del_test=float(np.mean(da)), sizes=[len(g) for g in groups]))
        log(f'[X1-b] seed {seed} done')
    return rows

def run(cfg, log):
    out = cfg['out'] / 'x1_lifecycle'; out.mkdir(parents=True, exist_ok=True)
    cfg.setdefault('draws_x1', 300); cfg.setdefault('search_draws', 100)
    rows = cpu_part(cfg, log)
    S = summarize(rows, cfg['search_draws'])
    acc = gpu_part(cfg, log, designed_partitions)
    write_json(out / 'results.json', dict(summary=S, cpu_rows=rows, gpu_rows=acc, eps=cfg['eps'], sigma2=SIGMA2))

    L = ['# X1 — 수명 비용과 손익분기', '',
         f'N=20, 명목 20dB, eps={fmt(cfg["eps"])}, {cfg["T"]}라운드. CPU 채널 {cfg["draws_x1"]}가지(배정 탐색은 {cfg["search_draws"]}가지), GPU seed {len(cfg["seeds"])}개. '
         '초기 학습 비용 T 는 모든 shard 의 초기 학습, U 는 client 1명 삭제(해당 shard 를 처음부터 재학습)의 20명 평균이다. 에너지는 정규화 단위.', '']
    for mode in ['maxpow', 'minen']:
        s = S[mode]
        L += [f'## {MODE_KO[mode]}', '', '### shard 수 K (무작위 배정)', '',
              table(['K', '초기 에너지 T', '삭제 1회 에너지 U', 'U/T', '삭제 계산(로컬 호출)', '재학습 지연', '단일 모델 대비 손익분기 삭제 수(중앙값)', '회수 불가 비율'],
                    [[K, fmt(v['T_energy']), fmt(v['U_energy']), fmt(v['U_energy'] / v['T_energy'], 3), fmt(v['U_compute'], 0), fmt(v['U_latency'], 0),
                      fmt(s['breakeven_vs_K1'][K]['median'], 2) if K > 1 else '-', pct(s['breakeven_vs_K1'][K]['p_never']).lstrip('+') if K > 1 else '-']
                     for K, v in s['K'].items()]), '',
              '### K=4 배정 방법 (학습 최적 배정 lam0 대비, draw 별 짝 비교 평균)', '',
              table(['방법', '초기 에너지 변화', '삭제 에너지 변화', '재학습 지연 변화', '손익분기 삭제 수(중앙값)', '1회 이내 회수 비율', '10회 이내 회수 비율'],
                    [[METHOD_KO.get(n, n), pct(v['T_change']), pct(v['U_change']), pct(v['lat_change']), fmt(v['breakeven_median'], 2),
                      pct(v['p_breakeven_le1']).lstrip('+'), pct(v['p_breakeven_le10']).lstrip('+')] for n, v in s['methods'].items()]), '']
    L += ['## 정확도 (GPU)', '']
    keys = sorted(set((r['mode'], r['config']) for r in acc))
    L.append(table(['전송 규칙', '구성', '초기 test', '삭제 후 test', 'shard 크기(첫 seed)'],
                   [[MODE_KO[m], METHOD_KO.get(cn, cn), fmt(np.mean([r['test'] for r in acc if (r['mode'], r['config']) == (m, cn)]), 4),
                     fmt(np.mean([r['del_test'] for r in acc if (r['mode'], r['config']) == (m, cn)]), 4),
                     next(r['sizes'] for r in acc if (r['mode'], r['config']) == (m, cn))] for m, cn in keys]))
    L += ['', '손익분기 삭제 수: 학습에서 더 쓴 비용을 삭제 비용 절감으로 회수하는 데 필요한 삭제 횟수. 0 이하면 학습 비용도 줄어든 것(공짜), inf 면 회수 불가.', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return dict(name='X1', verdicts=[])
