"""E1 — ① 삭제해도 다른 client 의 소속이 바뀌지 않는 배정 규칙.

(a) CPU 전수 계산: 채널 draw 마다 20명을 한 명씩 지우고, 같은 규칙을 남은 19명에 다시 적용했을 때
    다른 client 의 shard 가 바뀌는지 센다. 규칙: 무작위 순열 후 자르기 / 채널 순위 자르기 / 채널 순위 돌려 담기 /
    채널 고정 구간 / ID 해시.
(b) GPU: 소속이 바뀌는 규칙에서 "해당 shard 만 재학습(SISA)" 결과와 "규칙을 포함해 처음부터 학습(전체 알고리즘
    reference)" 결과의 예측 차이를, 같은 reference 를 다른 난수 tape 로 학습한 차이(난수 변동 기준)와 비교한다.
(c) 고정 구간 규칙의 대가: shard 크기 불균등에 따른 정확도와 비용.
"""
import math
import numpy as np
from .common import (load_split, channels, partition, shard_of, init_vector, write_json, N_CLIENTS, RULES, RULE_KO)
from .trainer import LocalTrainer, Job, run_jobs, Evaluator, split_probs, disagreement, kl
from . import costmodel
from .report import table, fmt, pct, pp, verdict_line

K = 4
STABLE = {'bins', 'hash'}

def regimes(eps):
    out = []
    for s2 in [1.0, .1, .01]:
        for mode in ['min_energy', 'fixed_mse']:
            out.append(dict(mode=mode, sigma2=s2, eps=eps))
        out.append(dict(mode='fixed_mse', sigma2=s2, eps=1e-4))
    return out

def cpu_part(cfg, log):
    M = cfg['draws']; T = cfg['T']; eps = cfg['eps']
    stab = {r: dict(changed_any=0, changed_clients=0, total=0, max_changed=0) for r in RULES}
    sizes = {r: dict(min_le2=0, empty=0, n=0) for r in RULES}
    regs = regimes(eps); cost = {r: {costmodel.regime_name(rg): dict(src=[], dele=[]) for rg in regs} for r in RULES}
    for d in range(M):
        sd = f'mc{d}'
        base, h = channels(sd, T)
        allc = list(range(N_CLIENTS))
        for r in RULES:
            full = partition(r, allc, base, K, sd)
            szs = [len(g) for g in full]
            sizes[r]['n'] += 1; sizes[r]['min_le2'] += int(min(szs) <= 2); sizes[r]['empty'] += int(min(szs) == 0)
            for u in allc:
                minus = partition(r, [i for i in allc if i != u], base, K, sd)
                ch = sum(shard_of(full, i) != shard_of(minus, i) for i in allc if i != u)
                st = stab[r]; st['total'] += 1; st['changed_any'] += int(ch > 0); st['changed_clients'] += ch
                st['max_changed'] = max(st['max_changed'], ch)
            if d < cfg['cost_draws']:
                for rg in regs:
                    nm = costmodel.regime_name(rg); acc = cost[r][nm]
                    src = {k: 0.0 for k in costmodel.METRICS}
                    for g in full:
                        if g:
                            c = costmodel.schedule_cost(h, g, {i: 0 for i in g}, 0, T, rg)
                            for k in src: src[k] += c[k]
                    acc['src'].append(src)
                    dl = []
                    for u in allc:
                        g = full[shard_of(full, u)]
                        c = costmodel.schedule_cost(h, g, {i: 0 for i in g}, 0, T, rg, exclude=(u,))
                        dl.append(c)
                    acc['dele'].append({k: float(np.mean([c[k] for c in dl])) for k in costmodel.METRICS})
        if (d + 1) % max(1, M // 5) == 0:
            log(f'[E1-a] draws {d + 1}/{M}')
    stab_out = {r: dict(frac_deletions_changed=v['changed_any'] / v['total'], mean_changed=v['changed_clients'] / v['total'],
                        max_changed=v['max_changed']) for r, v in stab.items()}
    size_out = {r: dict(p_min_size_le2=v['min_le2'] / v['n'], p_empty=v['empty'] / v['n']) for r, v in sizes.items()}
    cost_out = {}
    for r in RULES:
        cost_out[r] = {}
        for nm, acc in cost[r].items():
            cost_out[r][nm] = {part: {k: float(np.mean([x[k] for x in acc[part]])) for k in costmodel.METRICS}
                               for part in ('src', 'dele')}
    return stab_out, size_out, cost_out

def gpu_part(cfg, log):
    T = cfg['T']; rules = ['rank_chunk', 'bins'] if cfg['quick'] else ['random_split', 'rank_chunk', 'rank_rr', 'bins', 'hash']
    rg = dict(mode='R1', sigma2=.01)
    rows, accs = [], []
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device'])
        base, h = channels(seed, T)
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device'])
        allc = list(range(N_CLIENTS))
        order = sorted(allc, key=lambda i: base[i])
        U = [order[0], order[len(order) // 2], order[-1]][: (1 if cfg['quick'] else 3)]
        jobs = []; meta = {}
        for r in rules:
            full = partition(r, allc, base, K, seed)
            for c, g in enumerate(full):
                if g:
                    jobs.append(Job(f'{r}|src|{c}', c, w0, {i: 0 for i in g}, 0, T, rg))
            for u in U:
                cu = shard_of(full, u); rest = [i for i in full[cu] if i != u]
                if rest:
                    jobs.append(Job(f'{r}|sisa|{u}', cu, w0, {i: 0 for i in rest}, 0, T, rg))
                minus = partition(r, [i for i in allc if i != u], base, K, seed)
                for c, g in enumerate(minus):
                    if g:
                        jobs.append(Job(f'{r}|ref|{u}|{c}', c, w0, {i: 0 for i in g}, 0, T, rg))
                        jobs.append(Job(f'{r}|alt|{u}|{c}', c, w0, {i: 0 for i in g}, 0, T, rg, salt='alt'))
                meta[(r, u)] = dict(full=full, minus=minus, cu=cu,
                                    changed=int(sum(shard_of(full, i) != shard_of(minus, i) for i in allc if i != u)))
        log(f'[E1-b] seed {seed}: jobs={len(jobs)}')
        W, _, rec = run_jobs(tr, jobs, h, seed, log_every=max(1, T // 4), logger=log)
        ev = Evaluator(tr, data); P = ev.probs(W); ix = {j.name: k for k, j in enumerate(jobs)}
        for r in rules:
            full = meta[(r, U[0])]['full']
            src = {c: split_probs(P, ix[f'{r}|src|{c}']) for c, g in enumerate(full) if g}
            s = ev.ensemble(list(src.values()))
            accs.append(dict(seed=seed, rule=r, test=s['test'], worst_recall=s['test_worst_recall'],
                             sizes=[len(g) for g in full]))
            for u in U:
                m = meta[(r, u)]; cu = m['cu']
                plist = [src[c] for c in src if c != cu]
                if f'{r}|sisa|{u}' in ix:
                    plist.append(split_probs(P, ix[f'{r}|sisa|{u}']))
                sisa = ev.ensemble(plist)
                refs = [split_probs(P, ix[f'{r}|ref|{u}|{c}']) for c, g in enumerate(m['minus']) if g]
                alts = [split_probs(P, ix[f'{r}|alt|{u}|{c}']) for c, g in enumerate(m['minus']) if g]
                ref = ev.ensemble(refs); alt = ev.ensemble(alts)
                maxabs = None
                if r in STABLE:   # 같은 계산이어야 함: shard 별 파라미터 최대 차이
                    diffs = []
                    for c, g in enumerate(m['minus']):
                        if not g:
                            continue
                        a = W[ix[f'{r}|ref|{u}|{c}']]
                        b = W[ix[f'{r}|sisa|{u}']] if c == cu else W[ix[f'{r}|src|{c}']]
                        diffs.append(float((a - b).abs().max()))
                    maxabs = max(diffs) if diffs else 0.0
                rows.append(dict(seed=seed, rule=r, u=u, changed=m['changed'],
                                 dis_sisa_ref=disagreement(sisa['_ptest'], ref['_ptest']),
                                 dis_ref_alt=disagreement(ref['_ptest'], alt['_ptest']),
                                 kl_ref_sisa=kl(ref['_ptest'], sisa['_ptest']),
                                 acc_sisa=sisa['test'], acc_ref=ref['test'], maxabs_stable=maxabs))
        log(f'[E1-b] seed {seed} done')
    return rows, accs

def run(cfg, log):
    out = cfg['out'] / 'e1_stable'; out.mkdir(parents=True, exist_ok=True)
    stab, size, cost = cpu_part(cfg, log)
    rows, accs = gpu_part(cfg, log)
    write_json(out / 'results.json', dict(stability=stab, sizes=size, cost=cost, gpu_rows=rows, gpu_acc=accs, eps=cfg['eps']))

    # 판정
    h1a = all(stab[r]['frac_deletions_changed'] == 0 for r in STABLE) and all(stab[r]['frac_deletions_changed'] > 0 for r in RULES if r not in STABLE)
    unstable_rows = [x for x in rows if x['rule'] not in STABLE and x['changed'] > 0]
    if unstable_rows:
        frac_exceed = np.mean([x['dis_sisa_ref'] > x['dis_ref_alt'] for x in unstable_rows])
        ratio = np.mean([x['dis_sisa_ref'] for x in unstable_rows]) / max(1e-12, np.mean([x['dis_ref_alt'] for x in unstable_rows]))
        h1b = bool(frac_exceed >= .8)
    else:
        frac_exceed = ratio = float('nan'); h1b = None
    stable_max = max([x['maxabs_stable'] for x in rows if x['maxabs_stable'] is not None] or [0.0])
    acc_by = {r: np.mean([a['test'] for a in accs if a['rule'] == r]) for r in set(a['rule'] for a in accs)}
    ref_rule = 'rank_chunk' if 'rank_chunk' in acc_by else None
    h1c = None
    if 'bins' in acc_by and ref_rule:
        loss = acc_by[ref_rule] - acc_by['bins']; h1c = bool(loss <= .01)

    L = ['# E1 — 삭제해도 배정이 바뀌지 않는 규칙', '',
         f'K={K}, client 20명. CPU 계산은 채널 {cfg["draws"]}가지 x 삭제 20건, GPU 는 seed {len(cfg["seeds"])}개, {cfg["T"]}라운드, 명목 20dB 1회 전송.', '',
         '## (a) 한 명을 지웠을 때 다른 client 의 소속이 바뀌는가', '',
         table(['규칙', '소속이 바뀐 삭제 비율', '평균 바뀐 인원', '최대 바뀐 인원', '가장 작은 shard<=2명 확률', '빈 shard 확률'],
               [[RULE_KO[r], pct(stab[r]['frac_deletions_changed']).lstrip('+'), fmt(stab[r]['mean_changed']), stab[r]['max_changed'],
                 pct(size[r]['p_min_size_le2']).lstrip('+'), pct(size[r]['p_empty']).lstrip('+')] for r in RULES]), '',
         '고정 구간과 해시는 각 client 의 소속이 자기 정보로만 정해지므로 0 이 나와야 한다(구현 검증). '
         '순위 기반과 무작위 순열 방식은 한 명이 빠지면 경계가 밀린다.', '',
         '## (b) 소속이 바뀌는 규칙에서 SISA 결과와 전체 알고리즘 reference 의 차이', '']
    if rows:
        tb = []
        for r in sorted(set(x['rule'] for x in rows)):
            rr = [x for x in rows if x['rule'] == r]
            tb.append([RULE_KO[r], fmt(np.mean([x['changed'] for x in rr]), 1), pct(np.mean([x['dis_sisa_ref'] for x in rr])).lstrip('+'),
                       pct(np.mean([x['dis_ref_alt'] for x in rr])).lstrip('+'), fmt(np.mean([x['kl_ref_sisa'] for x in rr]), 4),
                       pp(np.mean([x['acc_sisa'] - x['acc_ref'] for x in rr])),
                       fmt(max([x['maxabs_stable'] for x in rr if x['maxabs_stable'] is not None] or [float('nan')]))])
        L.append(table(['규칙', '평균 바뀐 인원', 'SISA-reference 예측 불일치', 'reference-다른 tape 불일치(난수 기준)',
                        'KL(ref||SISA)', '정확도 차이(SISA-ref)', 'stable 규칙 파라미터 최대 차이'], tb))
    L += ['', '## (c) 규칙별 초기 학습 정확도', '',
          table(['규칙', '평균 test 정확도', 'shard 크기 예(seed 별)'],
                [[RULE_KO[r], fmt(acc_by[r], 4), ' / '.join(str(a['sizes']) for a in accs if a['rule'] == r)] for r in acc_by]), '',
          '## 비용 (해석식, 채널 draw 평균, 무작위 순열 방식 대비 비율)', '']
    tb = []
    for rg in regimes(cfg['eps']):
        nm = costmodel.regime_name(rg); b = cost['random_split'][nm]
        for r in RULES:
            c = cost[r][nm]
            tb.append([nm, RULE_KO[r], fmt(c['src']['energy'] / b['src']['energy'] if b['src']['energy'] else float('nan'), 4),
                       fmt(c['src']['ul_uses'] / b['src']['ul_uses'], 4), fmt(c['dele']['energy'] / b['dele']['energy'] if b['dele']['energy'] else float('nan'), 4),
                       fmt(c['dele']['ul_uses'] / b['dele']['ul_uses'], 4)])
    L.append(table(['조건', '규칙', '초기 에너지 비', '초기 UL 사용 비', '삭제 에너지 비', '삭제 UL 사용 비'], tb))
    L += ['', '수식 예측: 잡음 여유가 있는 영역(min_energy 이고 반복이 필요 없는 경우)에서는 크기가 같은 배정(무작위, 순위 자르기, 순위 돌려 담기)의 '
          '초기 에너지가 정확히 같아야 한다. 위 표의 비율이 1.000 이면 예측이 맞은 것이다.', '',
          '## 판정', '',
          verdict_line('H1a 고정 구간·해시는 소속 변화 0, 나머지는 >0', h1a, '위 (a) 표'),
          verdict_line('H1b 소속 변화가 난수 변동보다 큰 예측 차이를 만든다', h1b,
                       f'SISA-ref 불일치가 난수 기준보다 큰 경우 비율 {fmt(frac_exceed)} (기준 0.8), 평균 비 {fmt(ratio)}'),
          verdict_line('H1c 고정 구간의 정확도 손실 1pp 이하(순위 자르기 대비)', h1c,
                       f'정확도 {", ".join(f"{RULE_KO[r]} {acc_by[r]:.4f}" for r in acc_by)}'),
          f'- stable 규칙에서 SISA 와 reference 의 파라미터 최대 차이: {fmt(stable_max)} (배치 크기에 따른 부동소수 오차 수준이면 같은 계산으로 본다)', '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return dict(name='E1', verdicts=[('H1a', h1a), ('H1b', h1b), ('H1c', h1c)])
