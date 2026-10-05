"""E3 — ③ 연속 삭제를 고려한 shard 크기 설계.

바꾸는 것: shard 수 K (N=20 이므로 크기 10/5/4/3~4/2), 연속 삭제 수 L, 합치기 규칙 {합치지 않음, 최소 인원 미만이면 가장 작은 shard 와 합침}.
고정: 채널 draw 와 삭제 순서를 모든 K 가 함께 쓴다(짝 비교).
측정(CPU): 누적 통신 비용(에너지, UL 사용량), 누적 계산량(로컬 학습 호출), 최소 인원 미달 shard 수, 합치기 횟수.
         단일 모델 재학습(K=1)을 분모로 한 점수 J = 0.5 x 통신비 + 0.5 x 계산비 (이전 SISA 실험의 선택 점수와 같은 형태).
측정(GPU): L 번 삭제 후 앙상블 정확도.
"""
import numpy as np
from .common import load_split, channels, partition, init_vector, write_json, N_CLIENTS, LOCAL_STEPS
from .trainer import LocalTrainer, Job, run_jobs, Evaluator, split_probs
from . import costmodel
from .report import table, fmt, pct, verdict_line

KS = [2, 4, 5, 6, 10]
L_MAX = 10
VARIANTS = [('none', 3), ('merge', 3), ('merge', 2)]   # (규칙, 최소 인원). none 의 미달은 n_min 3 기준으로 센다
METRICS = ['energy', 'ul_uses', 'compute_calls']

def regimes(eps):
    out = []
    for s2 in [1.0, .1, .01]:
        out += [dict(mode='min_energy', sigma2=s2, eps=eps), dict(mode='fixed_mse', sigma2=s2, eps=eps),
                dict(mode='fixed_mse', sigma2=s2, eps=1e-4)]
    return out

def simulate(groups0, seq, policy, n_min, L):
    groups = [sorted(g) for g in groups0 if g]
    steps = []
    viol0 = sum(1 for g in groups if 0 < len(g) < n_min)
    for u in seq[:L]:
        ci = next(k for k, g in enumerate(groups) if u in g)
        g = [i for i in groups[ci] if i != u]
        retrain, merged = None, False
        if not g:
            groups.pop(ci)
        elif policy == 'merge' and len(g) < n_min and len(groups) > 1:
            others = [k for k in range(len(groups)) if k != ci]
            pk = min(others, key=lambda k: (len(groups[k]), k))
            new = sorted(g + groups[pk])
            groups = [gg for k, gg in enumerate(groups) if k not in (ci, pk)] + [new]
            retrain, merged = new, True
        else:
            groups[ci] = g; retrain = g
        steps.append((retrain, sum(1 for gg in groups if 0 < len(gg) < n_min), merged))
    return steps, viol0, groups

def cpu_part(cfg, log):
    M = cfg['draws_e3']; Q = cfg['seqs']; T = cfg['T']; regs = regimes(cfg['eps'])
    nK = len(KS) + 1   # 마지막 = K1(단일 모델)
    cum = np.zeros((nK, len(VARIANTS), len(regs), len(METRICS), L_MAX))
    viol = np.zeros((nK, len(VARIANTS), L_MAX)); merges = np.zeros((nK, len(VARIANTS), L_MAX)); viol0 = np.zeros((nK, len(VARIANTS)))
    count = 0
    for d in range(M):
        sd = f'mc{d}'; base, h = channels(sd, T)
        plans = []   # (kidx, vidx, q, steps)
        for q in range(Q):
            seq = list(np.random.default_rng(np.random.SeedSequence([d, q, 7])).permutation(N_CLIENTS))
            for ki, K in enumerate(KS + [1]):
                g0 = partition('random_split', range(N_CLIENTS), base, K, sd) if K > 1 else [list(range(N_CLIENTS))]
                for vi, (pol, nmin) in enumerate(VARIANTS):
                    if K == 1 and vi > 0:
                        steps, v0 = plans[-1][3], plans[-1][4]
                    else:
                        steps, v0, _ = simulate(g0, seq, pol if K > 1 else 'none', nmin, L_MAX)
                    plans.append((ki, vi, q, steps, v0))
        uniq = {}
        for p in plans:
            for st in p[3]:
                if st[0] is not None:
                    uniq.setdefault(tuple(st[0]), None)
        keys = list(uniq)
        stats = [costmodel.schedule_stats(h, list(g), {i: 0 for i in g}, 0, T) for g in keys]
        gidx = {g: k for k, g in enumerate(keys)}
        costs = np.zeros((len(regs), len(METRICS), len(keys)))
        for ri, rg in enumerate(regs):
            mc = costmodel.multi_cost(stats, rg)
            costs[ri, 0] = mc['energy']; costs[ri, 1] = mc['ul_uses']; costs[ri, 2] = mc['compute_calls'] * LOCAL_STEPS
        for ki, vi, q, steps, v0 in plans:
            run = np.zeros((len(regs), len(METRICS)))
            for l, (rt, vv, mg) in enumerate(steps):
                if rt is not None:
                    run += costs[:, :, gidx[tuple(rt)]]
                cum[ki, vi, :, :, l] += run
                viol[ki, vi, l] += float(vv > 0); merges[ki, vi, l] += float(mg)
            viol0[ki, vi] += float(v0 > 0)
        count += Q
        if (d + 1) % max(1, M // 5) == 0:
            log(f'[E3-a] draws {d + 1}/{M}')
    cum /= count; viol /= count; viol0 /= count; merges = np.cumsum(merges / count, axis=-1)
    return regs, cum, viol, viol0, merges

def gpu_part(cfg, log):
    T = cfg['T']; Ks = [4] if cfg['quick'] else [2, 4, 5, 10]; Ls = [0, 3] if cfg['quick'] else [0, 3, 6, 10]
    vars_ = [('none', 3), ('merge', 3)]
    rows = []
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device']); base, h = channels(seed, T)
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device']); ev = Evaluator(tr, data)
        seq = list(np.random.default_rng(np.random.SeedSequence([seed, 11])).permutation(N_CLIENTS))
        jobs, plan, seen = [], [], {}
        for K in Ks:
            g0 = partition('random_split', range(N_CLIENTS), base, K, seed)
            for pol, nmin in vars_:
                for L in Ls:
                    _, _, groups = simulate(g0, seq, pol, nmin, L)
                    names = []
                    for g in groups:
                        kk = tuple(g)
                        if kk not in seen:
                            seen[kk] = f'g{len(seen)}'
                            jobs.append(Job(seen[kk], len(seen), w0, {i: 0 for i in g}, 0, T, dict(mode='R1', sigma2=.01)))
                        names.append(seen[kk])
                    plan.append((K, pol, nmin, L, names, [len(g) for g in groups]))
        log(f'[E3-b] seed {seed}: jobs={len(jobs)}')
        W, _, _ = run_jobs(tr, jobs, h, seed, log_every=max(1, T // 4), logger=log)
        P = ev.probs(W); ix = {j.name: k for k, j in enumerate(jobs)}
        for K, pol, nmin, L, names, sizes in plan:
            r = ev.ensemble([split_probs(P, ix[n]) for n in names])
            rows.append(dict(seed=seed, K=K, policy=pol, n_min=nmin, L=L, test=r['test'], worst_recall=r['test_worst_recall'], sizes=sizes))
        log(f'[E3-b] seed {seed} done')
    return rows

def run(cfg, log):
    out = cfg['out'] / 'e3_lifetime'; out.mkdir(parents=True, exist_ok=True)
    regs, cum, viol, viol0, merges = cpu_part(cfg, log)
    np.savez_compressed(out / 'cpu_lifetime.npz', cum=cum, viol=viol, viol0=viol0, merges=merges)
    rows = gpu_part(cfg, log)
    Kall = KS + [1]
    # J 점수: K1 대비
    J = {}
    for vi in range(len(VARIANTS)):
        for ri in range(len(regs)):
            for ci, comm in enumerate(['energy', 'ul_uses']):
                mi = METRICS.index(comm); ci_ = METRICS.index('compute_calls')
                ref_comm = cum[-1, 0, ri, mi]; ref_comp = cum[-1, 0, ri, ci_]
                with np.errstate(divide='ignore', invalid='ignore'):
                    jj = 0.5 * cum[:-1, vi, ri, mi] / ref_comm + 0.5 * cum[:-1, vi, ri, ci_] / ref_comp
                J[(vi, ri, comm)] = jj   # [len(KS), L_MAX]
    write_json(out / 'results.json', dict(Ks=Kall, variants=VARIANTS, regimes=[costmodel.regime_name(r) for r in regs], L_max=L_MAX,
                                           cum=cum, violation_rate=viol, initial_violation=viol0, merges=merges,
                                           J={f'{VARIANTS[k[0]]}|{costmodel.regime_name(regs[k[1]])}|{k[2]}': v for k, v in J.items()},
                                           gpu_rows=rows, eps=cfg['eps']))
    # 판정: merge(3) 규칙, 처음부터 최소 인원을 만족하는 K 만 대상
    vi = 1; feasible = [k for k, K in enumerate(KS) if viol0[k, vi] == 0]
    changes, details = 0, []
    prim = [ri for ri, rg in enumerate(regs) if rg['eps'] == cfg['eps']]
    for ri in prim:
        jj = J[(vi, ri, 'energy')]
        if not np.all(np.isfinite(jj[feasible])):
            jj = J[(vi, ri, 'ul_uses')]
        a1 = feasible[int(np.argmin(jj[feasible, 0]))]; aL = feasible[int(np.argmin(jj[feasible, -1]))]
        regret = jj[a1, -1] / jj[aL, -1] - 1
        changes += int(a1 != aL)
        details.append((costmodel.regime_name(regs[ri]), KS[a1], KS[aL], regret))
    h3 = bool(prim) and changes * 2 >= len(prim)

    L = ['# E3 — 연속 삭제를 고려한 shard 크기', '',
         f'N=20, 연속 삭제 최대 {L_MAX}번, CPU 는 채널 {cfg["draws_e3"]}가지 x 삭제 순서 {cfg["seqs"]}가지, GPU 는 seed {len(cfg["seeds"])}개. '
         '각 삭제는 해당 shard 를 처음부터 다시 학습한다(정확 언러닝). 합치기 규칙은 shard 가 최소 인원 미만이 되면 가장 작은 다른 shard 와 합쳐 처음부터 학습한다.', '',
         '## 누적 비용 (merge, 최소 3명; eps 기준 조건)', '']
    for ri in prim:
        tb = []
        for k, K in enumerate(Kall):
            v = 1 if K > 1 else 0
            tb.append([K, fmt(cum[k, v, ri, 0, 0]), fmt(cum[k, v, ri, 0, -1]), fmt(cum[k, v, ri, 1, 0]), fmt(cum[k, v, ri, 1, -1]),
                       fmt(cum[k, v, ri, 2, 0]), fmt(cum[k, v, ri, 2, -1]),
                       fmt(J[(1, ri, 'energy')][k, 0]) if K > 1 else '1.000', fmt(J[(1, ri, 'energy')][k, -1]) if K > 1 else '1.000',
                       pct(viol0[k, v]).lstrip('+'), pct(viol[k, v, -1]).lstrip('+'), fmt(merges[k, v, -1], 2)])
        L += [f'### {costmodel.regime_name(regs[ri])}', '',
              table(['K', '에너지 L=1', f'에너지 L={L_MAX}', 'UL L=1', f'UL L={L_MAX}', '계산 L=1', f'계산 L={L_MAX}', 'J L=1', f'J L={L_MAX}',
                     '처음부터 미달', f'L={L_MAX} 미달 확률', '누적 합치기'], tb), '']
    L += ['## 합치지 않을 때 최소 인원 미달 확률 (L 별)', '',
          table(['K'] + [f'L={l + 1}' for l in range(L_MAX)], [[K] + [pct(viol[k, 0, l]).lstrip('+') for l in range(L_MAX)] for k, K in enumerate(KS)]), '',
          '## 정확도 (GPU)', '']
    tb = []
    for K in sorted(set(r['K'] for r in rows)):
        for pol in ['none', 'merge']:
            line = [K, pol]
            for Lv in sorted(set(r['L'] for r in rows)):
                rr = [r['test'] for r in rows if r['K'] == K and r['policy'] == pol and r['L'] == Lv]
                line.append(fmt(np.mean(rr), 4) if rr else '-')
            tb.append(line)
    L.append(table(['K', '규칙'] + [f'L={Lv}' for Lv in sorted(set(r['L'] for r in rows))], tb))
    L += ['', '## 판정', '',
          verdict_line('H3 연속 삭제 수에 따라 J 가 가장 작은 K 가 달라진다', h3,
                       f'eps 기준 조건 {len(prim)}개 중 {changes}개에서 바뀜. ' +
                       '; '.join(f'{n}: L=1 최적 K={a}, L={L_MAX} 최적 K={b}, L=1 기준으로 고르면 L={L_MAX} 에서 J {pct(rg)}' for n, a, b, rg in details)), '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return dict(name='E3', verdicts=[('H3', h3)])
