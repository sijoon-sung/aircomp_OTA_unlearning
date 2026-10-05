"""E4 — ④ 삭제 중 빈 대역을 삭제 shard 에 주기 (삭제 처리 지연).

바꾸는 것: 삭제 shard 에 주는 대역 배수 b {1,2,4} (K=4 블록 중), 전력 제약 {심볼당 상한, client 총 전력 상한}, 명목 SNR {0,10,20dB}.
고정: 배정(무작위 4x5), 채널, 삭제 대상(20명 각각), 라운드 수.
측정(CPU): 삭제 완료 지연(정규화 시간, 반복 포함, UL 과 UL+DL), 송신 에너지, 채널 사용량.
           다른 shard 가 계속 학습하는 경우, 그 shard 들이 같은 시간 동안 잃는 라운드.
측정(GPU): 총 전력 제약 + 1회 전송이면 대역을 넓힐수록 잡음이 b 배 -> 삭제 후 정확도.
수식 예측: 총 전력 제약에서 반복이 필요한 영역(MSE1 x b > eps)이면 R 이 b 배로 늘어 지연 이득이 사라진다.
          잡음 여유가 있는 영역에서는 지연이 1/b 로 준다.
"""
import numpy as np
from .common import load_split, channels, partition, shard_of, init_vector, write_json, N_CLIENTS
from .trainer import LocalTrainer, Job, run_jobs, Evaluator, split_probs
from . import costmodel
from .report import table, fmt, pct, pp, verdict_line

K = 4
BWS = [1, 2, 4]

def regimes(eps):
    out = []
    for s2 in [1.0, .1, .01]:
        for power in ['symbol', 'total']:
            out += [dict(mode='fixed_mse', sigma2=s2, eps=eps, power=power), dict(mode='fixed_mse', sigma2=s2, eps=1e-4, power=power)]
    return out

def cpu_part(cfg, log):
    M = cfg['draws']; T = cfg['T']; regs = regimes(cfg['eps'])
    # [regime, bw, metric(ul_time, total_time, energy, ul_uses, maxR), draw] — 20명 삭제 평균
    arr = np.zeros((len(regs), len(BWS), 5, M))
    slack = np.zeros((len(regs), len(BWS), M))
    for d in range(M):
        sd = f'mc{d}'; base, h = channels(sd, T)
        groups = partition('random_split', range(N_CLIENTS), base, K, sd)
        stats = []
        for u in range(N_CLIENTS):
            g = groups[shard_of(groups, u)]
            stats.append(costmodel.schedule_stats(h, g, {i: 0 for i in g}, 0, T, exclude=(u,)))
        for ri, rg in enumerate(regs):
            for bi, b in enumerate(BWS):
                mc = costmodel.multi_cost(stats, dict(rg, bw=b))
                arr[ri, bi, 0, d] = mc['ul_time'].mean(); arr[ri, bi, 1, d] = (mc['ul_time'] + mc['dl_time']).mean()
                arr[ri, bi, 2, d] = mc['energy'].mean(); arr[ri, bi, 3, d] = mc['ul_uses'].mean(); arr[ri, bi, 4, d] = mc['max_R'].mean()
                slack[ri, bi, d] = float(np.mean(mc['max_R'] <= 1))
        if (d + 1) % max(1, M // 5) == 0:
            log(f'[E4-a] draws {d + 1}/{M}')
    return regs, arr, slack

def gpu_part(cfg, log):
    T = cfg['T']; s2s = [.01] if cfg['quick'] else [.01, 1.0]; bws = [1, 4] if cfg['quick'] else BWS
    rows = []
    for seed in cfg['seeds']:
        data = load_split(cfg['raw'], seed, cfg['device']); base, h = channels(seed, T)
        tr = LocalTrainer(data, seed, cfg['device'], cfg['chunk']); w0 = init_vector(seed, cfg['device']); ev = Evaluator(tr, data)
        groups = partition('random_split', range(N_CLIENTS), base, K, seed)
        dels = [g[0] for g in groups]
        jobs = []
        for c, g in enumerate(groups):
            jobs.append(Job(f'src|{c}', c, w0, {i: 0 for i in g}, 0, T, dict(mode='R1', sigma2=.01)))
        for s2 in s2s:
            for b in bws:
                for mode, eps in [('R1', None), ('fixed_mse', cfg['eps'])]:
                    for u in dels:
                        c = shard_of(groups, u)
                        jobs.append(Job(f'del|{s2}|{b}|{mode}|{u}', c, w0, {i: 0 for i in groups[c] if i != u}, 0, T,
                                        dict(mode=mode, sigma2=s2, eps=eps, bw=b, power='total')))
        log(f'[E4-b] seed {seed}: jobs={len(jobs)}')
        W, _, rec = run_jobs(tr, jobs, h, seed, log_every=max(1, T // 4), logger=log)
        P = ev.probs(W); ix = {j.name: k for k, j in enumerate(jobs)}
        src = [split_probs(P, ix[f'src|{c}']) for c in range(K)]
        for s2 in s2s:
            for b in bws:
                for mode in ['R1', 'fixed_mse']:
                    accs, mses, times = [], [], []
                    for u in dels:
                        c = shard_of(groups, u); n = f'del|{s2}|{b}|{mode}|{u}'
                        accs.append(ev.ensemble([split_probs(P, ix[n]) if k == c else src[k] for k in range(K)])['test'])
                        mses.append(rec[ix[n]]['mean_mse']); times.append(rec[ix[n]]['ul_time'] + rec[ix[n]]['dl_time'])
                    rows.append(dict(seed=seed, sigma2=s2, bw=b, mode=mode, del_test=float(np.mean(accs)),
                                     mean_mse=float(np.mean(mses)), time=float(np.mean(times))))
        log(f'[E4-b] seed {seed} done')
    return rows

def run(cfg, log):
    out = cfg['out'] / 'e4_bandwidth'; out.mkdir(parents=True, exist_ok=True)
    regs, arr, slack = cpu_part(cfg, log)
    rows = gpu_part(cfg, log)
    names = [costmodel.regime_name(r) for r in regs]
    summ = []
    for ri, rg in enumerate(regs):
        t1 = arr[ri, 0, 0]
        for bi, b in enumerate(BWS):
            sp = t1 / arr[ri, bi, 0]; spt = arr[ri, 0, 1] / arr[ri, bi, 1]
            summ.append(dict(regime=names[ri], power=rg['power'], sigma2=rg['sigma2'], eps=rg['eps'], bw=b,
                             speedup_ul=float(np.mean(sp)), speedup_total=float(np.mean(spt)),
                             p_speedup_ge_90pct=float(np.mean(sp >= .9 * b)), p_slack=float(np.mean(slack[ri, bi])),
                             energy_ratio=float(np.mean(arr[ri, bi, 2] / arr[ri, 0, 2])), mean_maxR=float(np.mean(arr[ri, bi, 4])),
                             others_slowdown=(K - 1) / (K - b) if b < K else float('inf')))
    write_json(out / 'results.json', dict(summary=summ, gpu_rows=rows, eps=cfg['eps']))
    # 판정: (1) 심볼당 상한에서는 지연이 b 배 준다(예측 검증), (2) 총 전력 상한에서는 이득이 잡음 여유와 일치
    sym = [s for s in summ if s['power'] == 'symbol' and s['bw'] > 1]
    h4a = all(abs(s['speedup_ul'] - s['bw']) / s['bw'] < .05 for s in sym)
    # (2) 총 전력 상한 + eps 반복 규칙: b=최대일 때 지연 감소가 b 의 절반 이상 남는가 (SNR 별)
    bmax = max(BWS)
    tot = {s['sigma2']: s for s in summ if s['power'] == 'total' and s['bw'] == bmax and s['eps'] == cfg['eps']}
    h4b = all(s['speedup_ul'] >= bmax / 2 for s in tot.values()) if tot else None
    # (3) 총 전력 상한 + 1회 전송: 대역을 넓혀 잡음이 b 배가 돼도 정확도 손실 1pp 이하인가
    acc = {}
    for r in rows:
        acc.setdefault((r['sigma2'], r['mode'], r['bw']), []).append(r['del_test'])
    loss = {}
    for (s2, mode, b), v in acc.items():
        if mode == 'R1' and b == max(r['bw'] for r in rows) and (s2, mode, 1) in acc:
            loss[s2] = float(np.mean(acc[(s2, mode, 1)]) - np.mean(v))
    h4c = all(v <= .01 for v in loss.values()) if loss else None
    L = ['# E4 — 삭제 중 빈 대역을 삭제 shard 에 주기', '',
         f'K={K} 블록, 삭제 대상 20명 평균, CPU 채널 {cfg["draws"]}가지, GPU seed {len(cfg["seeds"])}개. 시간 단위는 기본 블록에서 심볼 1개를 보내는 시간이다. '
         'UL+DL 지연에는 디지털 모델 broadcast(32 bit/파라미터, 2 bit/심볼)를 포함했다.', '',
         '## 대역 배수에 따른 삭제 지연 감소', '',
         table(['조건', 'b', 'UL 지연 감소(배)', 'UL+DL 지연 감소(배)', 'b 의 90% 이상 달성 비율', '반복 불필요 비율', '에너지 비', '평균 최대 R', '다른 shard 감속(배)'],
               [[s['regime'], s['bw'], fmt(s['speedup_ul'], 2), fmt(s['speedup_total'], 2), fmt(s['p_speedup_ge_90pct'], 2), fmt(s['p_slack'], 2),
                 fmt(s['energy_ratio'], 3), fmt(s['mean_maxR'], 1), fmt(s['others_slowdown'], 2) if np.isfinite(s['others_slowdown']) else '중단'] for s in summ]), '',
         '다른 shard 감속: 나머지 K-1 개 shard 가 계속 학습하면 남은 K-b 블록을 나눠 써야 하므로 그만큼 느려진다(b=K 이면 멈춤). '
         '학습이 끝난 뒤의 삭제라면 이 비용은 없다.', '',
         '## 정확도 (총 전력 상한, GPU)', '',
         table(['잡음 분산', '모드', 'b', '삭제 후 test 평균'], [[f'{k[0]:g}', k[1], k[2], fmt(np.mean(v), 4)] for k, v in sorted(acc.items())]), '',
         '## 판정', '',
         verdict_line('H4a 심볼당 전력 상한이면 대역 b 배에 지연이 1/b (구현·수식 검증)', h4a, ''),
         verdict_line(f'H4b 총 전력 상한 + eps 반복 규칙에서 b={bmax} 일 때 UL 지연 감소가 {bmax / 2:g}배 이상', h4b,
                      ', '.join(f"잡음 {k:g}: {fmt(v['speedup_ul'], 2)}배" for k, v in sorted(tot.items()))),
         verdict_line('H4c 총 전력 상한 + 1회 전송에서 대역을 최대로 넓혀도 삭제 후 정확도 손실 1pp 이하', h4c,
                      ', '.join(f'잡음 {k:g}: 손실 {pp(v)}' for k, v in sorted(loss.items()))), '']
    (out / 'REPORT_KO.md').write_text('\n'.join(L), encoding='utf-8')
    return dict(name='E4', verdicts=[('H4a', h4a), ('H4b', h4b), ('H4c', h4c)])
